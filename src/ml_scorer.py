"""Load daily-trained market model and score live candidates."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml_features import MARKET_FEATURE_NAMES, extract_features_at, features_to_array

logger = logging.getLogger(__name__)

_loaded_bundle: dict | None = None
_loaded_path: str | None = None


def _load_model(root: Path, config: dict):
    global _loaded_bundle, _loaded_path

    model_path = root / config.get("ml", {}).get("model_dir", "models") / "market_model.joblib"
    path_str = str(model_path.resolve())

    if _loaded_bundle is not None and _loaded_path == path_str and model_path.exists():
        return _loaded_bundle

    if not model_path.exists():
        return None

    try:
        import joblib

        bundle = joblib.load(model_path)
        _loaded_bundle = bundle
        _loaded_path = path_str
        return bundle
    except Exception:
        logger.exception("Failed to load ML model from %s", model_path)
        return None


def predict_probability(
    df: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    root: Path,
    config: dict,
) -> float | None:
    """
    Return P(swing target hit before stop) from daily-trained market model.
    Returns None if model unavailable.
    """
    ml_cfg = config.get("ml", {})
    if not ml_cfg.get("enabled", True):
        return None

    bundle = _load_model(root, config)
    if bundle is None:
        return None

    model = bundle.get("model")
    if model is None:
        return None

    idx = len(df) - 1
    feats = extract_features_at(df, idx, benchmark)
    if feats is None:
        return None

    X = features_to_array(feats).reshape(1, -1)
    try:
        prob = float(model.predict_proba(X)[0, 1])
        return round(prob, 4)
    except Exception:
        logger.exception("ML prediction failed")
        return None


def model_available(root: Path, config: dict) -> bool:
    return _load_model(root, config) is not None


def reset_cache() -> None:
    global _loaded_bundle, _loaded_path
    _loaded_bundle = None
    _loaded_path = None
