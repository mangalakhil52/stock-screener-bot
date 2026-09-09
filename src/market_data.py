"""Fetch OHLCV history via Indian API (NSE) for advanced signal analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from indian_api_client import (
    IndianApiCredentialsError,
    fetch_nifty_benchmark,
    fetch_symbol_history,
)

logger = logging.getLogger(__name__)


def _root_from_config(config: dict | None) -> Path:
    if config and config.get("_root"):
        return Path(config["_root"])
    return Path(__file__).resolve().parent.parent


def fetch_history(
    symbols: list[str],
    days: int = 90,
    config: dict | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Batch-fetch OHLCV for symbols plus Nifty benchmark from Indian API."""
    if not symbols:
        return {}, None

    root = _root_from_config(config)
    cfg = config or {}

    try:
        benchmark = fetch_nifty_benchmark(days, root, cfg)
    except IndianApiCredentialsError as exc:
        logger.error("%s", exc)
        return {}, None

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        sym = symbol.strip().upper()
        try:
            frame = fetch_symbol_history(sym, days, root, cfg)
            if frame is not None and not frame.empty:
                result[sym] = frame
        except IndianApiCredentialsError:
            raise
        except Exception:
            logger.debug("No history for %s", sym, exc_info=True)

    if benchmark is None:
        logger.warning("Could not load Nifty benchmark from Indian API")

    return result, benchmark


def fetch_benchmark(days: int = 365, config: dict | None = None) -> pd.DataFrame | None:
    """Download Nifty benchmark OHLCV only."""
    root = _root_from_config(config)
    try:
        return fetch_nifty_benchmark(days, root, config or {})
    except IndianApiCredentialsError as exc:
        logger.error("%s", exc)
        return None
