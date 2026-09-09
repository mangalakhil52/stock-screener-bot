"""NSE market data via NSEPython — full equity universe and OHLCV (no API key)."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

_UNIVERSE_CACHE: list[str] | None = None
_NSE_MODULE = None


def _load_nse_module():
    """Use server edition on Linux/CI; fall back to local edition on Windows."""
    global _NSE_MODULE
    if _NSE_MODULE is not None:
        return _NSE_MODULE
    try:
        import nsepythonserver as nse

        _NSE_MODULE = nse
        return nse
    except ImportError:
        import nsepython as nse

        _NSE_MODULE = nse
        return nse


def _nse_cfg(config: dict) -> dict:
    return config.get("nse", {})


def _universe_cache_path(root: Path, config: dict) -> Path:
    rel = _nse_cfg(config).get("universe_cache", "data/cache/nse_universe.json")
    return root / rel


def _ohlcv_cache_path(root: Path, config: dict, symbol: str) -> Path:
    rel = _nse_cfg(config).get("ohlcv_cache_dir", "data/cache/ohlcv")
    return root / rel / f"{symbol.upper()}.json"


def _equity_date_range(days: int) -> tuple[str, str]:
    end = datetime.now(IST).date()
    start = end - timedelta(days=int(days * 1.6) + 15)
    return start.strftime("%d-%m-%Y"), end.strftime("%d-%m-%Y")


def _index_date_range(days: int) -> tuple[str, str]:
    end = datetime.now(IST).date()
    start = end - timedelta(days=int(days * 1.6) + 15)
    return start.strftime("%d-%b-%Y"), end.strftime("%d-%b-%Y")


def _normalize_equity_df(raw: pd.DataFrame) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None

    required = {
        "CH_OPENING_PRICE": "open",
        "CH_TRADE_HIGH_PRICE": "high",
        "CH_TRADE_LOW_PRICE": "low",
        "CH_CLOSING_PRICE": "close",
        "CH_TOT_TRADED_QTY": "volume",
    }
    if not all(col in raw.columns for col in required):
        return None

    idx = pd.to_datetime(raw["CH_TIMESTAMP"].values, errors="coerce")
    frame = pd.DataFrame(
        {out: pd.to_numeric(raw[col].values, errors="coerce") for col, out in required.items()},
        index=idx,
    )
    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.dropna(subset=["close"])
    return frame if not frame.empty else None


def _normalize_index_df(raw: pd.DataFrame) -> pd.DataFrame | None:
    if raw is None or raw.empty:
        return None

    date_col = "HistoricalDate" if "HistoricalDate" in raw.columns else None
    if date_col is None:
        return None

    col_map = {
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
    }
    if not all(col in raw.columns for col in col_map):
        return None

    idx = pd.to_datetime(raw[date_col].values, errors="coerce")
    frame = pd.DataFrame(
        {out: pd.to_numeric(raw[col].values, errors="coerce") for col, out in col_map.items()},
        index=idx,
    )
    frame["volume"] = 0.0
    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.dropna(subset=["close"])
    return frame if not frame.empty else None


def _read_ohlcv_cache(path: Path, max_age_h: float) -> pd.DataFrame | None:
    if not path.exists():
        return None
    age_h = (time.time() - path.stat().st_mtime) / 3600
    if age_h >= max_age_h:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload["data"], index=pd.to_datetime(payload["index"]))
        return frame.sort_index()
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return None


def _write_ohlcv_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index": [ts.isoformat() for ts in frame.index],
        "data": frame.reset_index(drop=True).to_dict(orient="list"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def get_nse_equity_universe(root: Path, config: dict) -> list[str]:
    """All NSE equity symbols via NSEPython `nse_eq_symbols()` (no cap)."""
    global _UNIVERSE_CACHE
    if _UNIVERSE_CACHE is not None:
        return _UNIVERSE_CACHE

    cache_path = _universe_cache_path(root, config)
    max_age_h = float(_nse_cfg(config).get("universe_cache_hours", 24))

    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < max_age_h:
            try:
                symbols = json.loads(cache_path.read_text(encoding="utf-8"))
                if symbols:
                    _UNIVERSE_CACHE = [str(s).upper() for s in symbols]
                    return _UNIVERSE_CACHE
            except (json.JSONDecodeError, OSError):
                logger.debug("Universe cache unreadable — refreshing")

    nse = _load_nse_module()
    logger.info("Fetching NSE equity symbol list via NSEPython...")
    try:
        symbols = [str(s).strip().upper() for s in nse.nse_eq_symbols() if str(s).strip()]
    except Exception:
        logger.exception("Failed to fetch NSE equity symbols")
        if cache_path.exists():
            symbols = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            return []

    symbols = sorted(set(symbols))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(symbols), encoding="utf-8")
    _UNIVERSE_CACHE = symbols
    logger.info("NSE equity universe: %s symbols", len(symbols))
    return symbols


def fetch_symbol_history(
    symbol: str,
    days: int,
    root: Path,
    config: dict,
) -> pd.DataFrame | None:
    sym = symbol.strip().upper()
    cache_h = float(_nse_cfg(config).get("ohlcv_cache_hours", 24))
    cache_path = _ohlcv_cache_path(root, config, sym)
    cached = _read_ohlcv_cache(cache_path, cache_h)
    if cached is not None and len(cached) >= min(days, 30):
        return cached.tail(days).copy()

    delay = float(_nse_cfg(config).get("request_delay_sec", 0.2))
    if delay > 0:
        time.sleep(delay)

    series = str(_nse_cfg(config).get("equity_series", "EQ"))
    start_date, end_date = _equity_date_range(days)
    nse = _load_nse_module()

    try:
        raw = nse.equity_history(sym, series, start_date, end_date)
    except Exception:
        logger.debug("NSEPython equity_history failed for %s", sym, exc_info=True)
        return cached.tail(days).copy() if cached is not None else None

    frame = _normalize_equity_df(raw)
    if frame is None or frame.empty:
        return cached.tail(days).copy() if cached is not None else None

    _write_ohlcv_cache(cache_path, frame)
    return frame.tail(days).copy()


def fetch_nifty_benchmark(days: int, root: Path, config: dict) -> pd.DataFrame | None:
    index_name = str(_nse_cfg(config).get("benchmark_index", "NIFTY 50"))
    cache_path = _ohlcv_cache_path(root, config, "BENCHMARK_NIFTY50")
    cache_h = float(_nse_cfg(config).get("ohlcv_cache_hours", 24))
    cached = _read_ohlcv_cache(cache_path, cache_h)
    if cached is not None and len(cached) >= min(days, 30):
        return cached.tail(days).copy()

    delay = float(_nse_cfg(config).get("request_delay_sec", 0.2))
    if delay > 0:
        time.sleep(delay)

    start_date, end_date = _index_date_range(days)
    nse = _load_nse_module()

    try:
        raw = nse.index_history(index_name, start_date, end_date)
    except Exception:
        logger.debug("NSEPython index_history failed for %s", index_name, exc_info=True)
        # Fallback: liquid Nifty ETF
        return fetch_symbol_history("NIFTYBEES", days, root, config)

    frame = _normalize_index_df(raw)
    if frame is None or len(frame) < 30:
        return fetch_symbol_history("NIFTYBEES", days, root, config)

    _write_ohlcv_cache(cache_path, frame)
    logger.info("Loaded %s benchmark (%s bars)", index_name, len(frame))
    return frame.tail(days).copy()
