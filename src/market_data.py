"""Fetch and cache OHLCV history for advanced signal analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_NSE_SUFFIX = ".NS"
_BENCHMARK = "^NSEI"


def to_nse_ticker(symbol: str) -> str:
    return f"{symbol.upper()}{_NSE_SUFFIX}"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]
    return df.rename(columns=str.lower)


def fetch_history(
    symbols: list[str],
    days: int = 90,
    benchmark: str = _BENCHMARK,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame | None]:
    """Batch-download OHLCV for symbols plus Nifty benchmark."""
    if not symbols:
        return {}, None

    tickers = [to_nse_ticker(s) for s in symbols]
    period = "6mo" if days > 60 else "3mo"

    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance not installed — advanced analysis disabled")
        return {}, None

    all_tickers = tickers + [benchmark]
    try:
        raw = yf.download(
            all_tickers,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
    except Exception:
        logger.exception("Failed to download market data")
        return {}, None

    if raw is None or raw.empty:
        return {}, None

    result: dict[str, pd.DataFrame] = {}
    benchmark_df: pd.DataFrame | None = None

    if len(all_tickers) == 1:
        frame = _flatten_columns(raw.copy())
        frame.index = pd.to_datetime(frame.index)
        if all_tickers[0] == benchmark:
            benchmark_df = frame.tail(days).copy()
        else:
            result[symbols[0]] = frame.tail(days).copy()
        return result, benchmark_df

    for symbol, ticker in zip(symbols, tickers):
        try:
            sub = raw.xs(ticker, axis=1, level=1).copy()
            sub = _flatten_columns(sub).dropna(how="all")
            if not sub.empty:
                sub.index = pd.to_datetime(sub.index)
                result[symbol] = sub.tail(days).copy()
        except (KeyError, ValueError):
            try:
                sub = raw[ticker].copy()
                sub = _flatten_columns(sub).dropna(how="all")
                if not sub.empty:
                    result[symbol] = sub.tail(days).copy()
            except (KeyError, ValueError):
                logger.debug("No history for %s", symbol)

    try:
        bench = raw.xs(benchmark, axis=1, level=1).copy()
        benchmark_df = _flatten_columns(bench).tail(days).copy()
        benchmark_df.index = pd.to_datetime(benchmark_df.index)
    except (KeyError, ValueError):
        try:
            benchmark_df = _flatten_columns(raw[benchmark].copy()).tail(days).copy()
        except (KeyError, ValueError):
            logger.warning("Could not load Nifty benchmark data")

    return result, benchmark_df
