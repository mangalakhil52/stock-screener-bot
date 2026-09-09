"""Fetch OHLCV history via NSEPython for advanced signal analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from nse_client import fetch_nifty_benchmark, fetch_symbol_history

logger = logging.getLogger(__name__)

_benchmark_cache: dict[tuple[int, str], pd.DataFrame | None] = {}


def _root_from_config(config: dict | None) -> Path:
    if config and config.get("_root"):
        return Path(config["_root"])
    return Path(__file__).resolve().parent.parent


def _get_benchmark(days: int, root: Path, cfg: dict) -> pd.DataFrame | None:
    root_key = str(root.resolve())
    key = (days, root_key)
    if key not in _benchmark_cache:
        _benchmark_cache[key] = fetch_nifty_benchmark(days, root, cfg)
    return _benchmark_cache[key]


def fetch_history(
    symbols: list[str],
    days: int = 90,
    config: dict | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Batch-fetch OHLCV for symbols plus Nifty benchmark from NSE."""
    if not symbols:
        return {}, None

    root = _root_from_config(config)
    cfg = config or {}
    benchmark = _get_benchmark(days, root, cfg)

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        sym = symbol.strip().upper()
        try:
            frame = fetch_symbol_history(sym, days, root, cfg)
            if frame is not None and not frame.empty:
                result[sym] = frame
        except Exception:
            logger.debug("No history for %s", sym, exc_info=True)

    if benchmark is None:
        logger.warning("Could not load Nifty benchmark from NSE")

    return result, benchmark


def fetch_benchmark(days: int = 365, config: dict | None = None) -> pd.DataFrame | None:
    """Download Nifty benchmark OHLCV only."""
    root = _root_from_config(config)
    return _get_benchmark(days, root, config or {})
