"""Daily self-training ML model on broad NSE market data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from dhan_client import get_nse_equity_universe, select_training_symbols
from market_data import fetch_benchmark, fetch_history
from ml_features import (
    MARKET_FEATURE_NAMES,
    extract_features_at,
    features_to_array,
    is_setup_like_day,
    label_forward_outcome,
)

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


def load_universe(root: Path, config: dict) -> list[str]:
    """Full NSE equity universe from Dhan instrument master (no hardcoded list)."""
    ml_cfg = config.get("ml", {})
    try:
        all_symbols = get_nse_equity_universe(root, config)
    except Exception:
        logger.exception("Failed to load universe from Dhan")
        return []

    max_sym = int(ml_cfg.get("max_symbols", 250))
    return select_training_symbols(all_symbols, max_sym)


def _model_paths(root: Path, config: dict) -> tuple[Path, Path]:
    model_dir = root / config.get("ml", {}).get("model_dir", "models")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / "market_model.joblib", model_dir / "train_meta.json"


def is_model_stale(root: Path, config: dict) -> bool:
    ml_cfg = config.get("ml", {})
    if not ml_cfg.get("enabled", True):
        return False

    _, meta_path = _model_paths(root, config)
    if not meta_path.exists():
        return True

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        trained_at = datetime.fromisoformat(meta["trained_at"])
    except (json.JSONDecodeError, KeyError, ValueError):
        return True

    max_age = timedelta(hours=float(ml_cfg.get("retrain_if_older_hours", 20)))
    return datetime.now(IST) - trained_at.astimezone(IST) > max_age


def _build_training_samples(
    df: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    config: dict,
) -> tuple[list[np.ndarray], list[int]]:
    ml_cfg = config.get("ml", {})
    target_pct = float(ml_cfg.get("target_pct", 5.0))
    stop_pct = float(ml_cfg.get("stop_pct", 3.0))
    horizon = int(ml_cfg.get("label_horizon", 10))
    setup_only = ml_cfg.get("train_on_setup_days_only", True)

    X: list[np.ndarray] = []
    y: list[int] = []

    start = 55
    end = len(df) - horizon - 1
    for idx in range(start, end):
        if setup_only and not is_setup_like_day(df, idx):
            continue
        label = label_forward_outcome(df, idx, target_pct, stop_pct, horizon)
        if label is None:
            continue
        feats = extract_features_at(df, idx, benchmark)
        if feats is None:
            continue
        X.append(features_to_array(feats))
        y.append(label)

    return X, y


def train_market_model(root: Path, config: dict) -> dict | None:
    """Train on whole-market universe; returns training metadata."""
    ml_cfg = config.get("ml", {})
    if not ml_cfg.get("enabled", True):
        return None

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib
    except ImportError:
        logger.error("scikit-learn required for ML training — pip install scikit-learn")
        return None

    symbols = load_universe(root, config)
    if not symbols:
        logger.error("No symbols in universe — cannot train")
        return None

    history_days = int(ml_cfg.get("history_days", 365))
    batch_size = int(ml_cfg.get("batch_size", 40))

    logger.info("ML training: fetching %s symbols (%s days history)...", len(symbols), history_days)

    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    symbols_loaded = 0

    benchmark = fetch_benchmark(history_days, config)

    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        history, _ = fetch_history(chunk, days=history_days, config=config)
        for sym, df in history.items():
            if df is None or len(df) < 80:
                continue
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            # Align benchmark length
            bench_slice = benchmark
            X, y = _build_training_samples(df, bench_slice, config)
            if X:
                all_X.extend(X)
                all_y.extend(y)
                symbols_loaded += 1
        logger.info("ML training progress: %s/%s symbols", min(i + batch_size, len(symbols)), len(symbols))

    min_samples = int(ml_cfg.get("min_samples", 800))
    if len(all_y) < min_samples:
        logger.warning(
            "Only %s training samples (need %s) — keeping existing model",
            len(all_y),
            min_samples,
        )
        return None

    X_arr = np.vstack(all_X)
    y_arr = np.array(all_y, dtype=int)

    logger.info(
        "ML dataset: %s samples from %s symbols | win rate %.1f%%",
        len(y_arr),
        symbols_loaded,
        y_arr.mean() * 100,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, random_state=42, stratify=y_arr
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.85,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy = float(accuracy_score(y_test, y_pred))
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = 0.5

    model_path, meta_path = _model_paths(root, config)
    joblib.dump({"model": model, "feature_names": MARKET_FEATURE_NAMES}, model_path)

    meta = {
        "trained_at": datetime.now(IST).isoformat(),
        "samples": len(y_arr),
        "symbols_loaded": symbols_loaded,
        "universe_size": len(symbols),
        "win_rate": round(float(y_arr.mean()), 4),
        "test_accuracy": round(accuracy, 4),
        "test_auc": round(auc, 4),
        "feature_names": MARKET_FEATURE_NAMES,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("ML model saved — accuracy %.1f%% AUC %.3f", accuracy * 100, auc)
    return meta


def ensure_daily_model(root: Path, config: dict) -> dict | None:
    """Train if stale; return latest metadata."""
    ml_cfg = config.get("ml", {})
    if not ml_cfg.get("enabled", True):
        return None
    if not ml_cfg.get("auto_train_daily", True):
        return _load_meta(root, config)

    if is_model_stale(root, config):
        logger.info("ML model stale — starting daily market training...")
        return train_market_model(root, config) or _load_meta(root, config)
    return _load_meta(root, config)


def _load_meta(root: Path, config: dict) -> dict | None:
    _, meta_path = _model_paths(root, config)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
