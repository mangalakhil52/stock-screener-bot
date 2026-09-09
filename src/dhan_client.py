"""Dhan API client — instrument master and OHLCV (replaces yfinance / hardcoded universe)."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

# Nifty 50 index in Dhan compact master (SEM_TRADING_SYMBOL == "NIFTY")
NIFTY_SECURITY_ID = "13"
NIFTY_SYMBOL = "NIFTY"

_INSTRUMENT_CACHE: pd.DataFrame | None = None
_INSTRUMENT_CACHE_PATH: Path | None = None
_INSTRUMENT_CACHE_MTIME: float | None = None

_dhan_client = None


class DhanCredentialsError(RuntimeError):
    """Raised when DHAN_CLIENT_ID or DHAN_ACCESS_TOKEN are missing."""


def get_credentials() -> tuple[str, str]:
    client_id = os.environ.get("DHAN_CLIENT_ID", "").strip()
    access_token = os.environ.get("DHAN_ACCESS_TOKEN", "").strip()
    if not client_id or not access_token:
        raise DhanCredentialsError(
            "Dhan API credentials missing. Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in "
            ".env (local) or GitHub Actions secrets (CI). See .env.example and README."
        )
    return client_id, access_token


def get_dhan_client():
    """Lazy singleton Dhan REST client (requires credentials for historical data)."""
    global _dhan_client
    if _dhan_client is not None:
        return _dhan_client

    from dhanhq import DhanContext, dhanhq

    client_id, access_token = get_credentials()
    _dhan_client = dhanhq(DhanContext(client_id, access_token))
    return _dhan_client


def _instrument_cache_file(root: Path, config: dict) -> Path:
    dhan_cfg = config.get("dhan", {})
    rel = dhan_cfg.get("instrument_cache", "data/cache/dhan_instruments.parquet")
    return root / rel


def load_instrument_master(root: Path, config: dict, force_refresh: bool = False) -> pd.DataFrame:
    """
    Load NSE/BSE instrument master from Dhan (public CSV via SDK).
    Cached on disk for 24h to avoid re-downloading ~200k rows every run.
    """
    global _INSTRUMENT_CACHE, _INSTRUMENT_CACHE_PATH, _INSTRUMENT_CACHE_MTIME

    cache_path = _instrument_cache_file(root, config)
    max_age_h = float(config.get("dhan", {}).get("instrument_cache_hours", 24))

    if not force_refresh and cache_path.exists():
        mtime = cache_path.stat().st_mtime
        age_h = (time.time() - mtime) / 3600
        if (
            _INSTRUMENT_CACHE is not None
            and _INSTRUMENT_CACHE_PATH == cache_path
            and _INSTRUMENT_CACHE_MTIME == mtime
            and age_h < max_age_h
        ):
            return _INSTRUMENT_CACHE
        if age_h < max_age_h:
            try:
                df = pd.read_parquet(cache_path)
                _INSTRUMENT_CACHE = df
                _INSTRUMENT_CACHE_PATH = cache_path
                _INSTRUMENT_CACHE_MTIME = mtime
                return df
            except Exception:
                logger.debug("Could not read instrument cache — refreshing")

    from dhanhq import dhanhq

    logger.info("Downloading Dhan instrument master (compact)...")
    df = dhanhq.fetch_security_list("compact")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:
        csv_path = cache_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        logger.info("Saved instrument cache as CSV: %s", csv_path)

    _INSTRUMENT_CACHE = df
    _INSTRUMENT_CACHE_PATH = cache_path
    _INSTRUMENT_CACHE_MTIME = cache_path.stat().st_mtime if cache_path.exists() else time.time()
    return df


def _nse_cash_mask(df: pd.DataFrame, series: list[str]) -> pd.Series:
    return (
        (df["SEM_EXM_EXCH_ID"] == "NSE")
        & (df["SEM_INSTRUMENT_NAME"] == "EQUITY")
        & (df["SEM_SEGMENT"] == "E")
        & (df["SEM_SERIES"].isin(series))
    )


def get_nse_equity_universe(root: Path, config: dict) -> list[str]:
    """All NSE cash equities from Dhan master (EQ / BE / SM per config)."""
    dhan_cfg = config.get("dhan", {})
    series = dhan_cfg.get("universe_series", ["EQ", "BE", "SM"])
    master = load_instrument_master(root, config)
    subset = master.loc[_nse_cash_mask(master, series), "SEM_TRADING_SYMBOL"]
    symbols = sorted({str(s).strip().upper() for s in subset if pd.notna(s) and str(s).strip()})
    logger.info("NSE equity universe from Dhan: %s symbols (series %s)", len(symbols), series)
    return symbols


def select_training_symbols(symbols: list[str], max_symbols: int) -> list[str]:
    """Rotate daily through full universe when training cap < market size."""
    if max_symbols <= 0 or len(symbols) <= max_symbols:
        return symbols
    day = datetime.now(IST).toordinal()
    start = day % len(symbols)
    rotated = symbols[start:] + symbols[:start]
    return rotated[:max_symbols]


def symbol_to_security_id(symbol: str, root: Path, config: dict) -> str | None:
    master = load_instrument_master(root, config)
    sym = symbol.strip().upper()
    dhan_cfg = config.get("dhan", {})
    series = dhan_cfg.get("universe_series", ["EQ", "BE", "SM"])
    match = master.loc[
        _nse_cash_mask(master, series) & (master["SEM_TRADING_SYMBOL"].str.upper() == sym),
        "SEM_SMST_SECURITY_ID",
    ]
    if match.empty:
        # Fallback: any NSE equity row (e.g. symbol only in other series)
        match = master.loc[
            (master["SEM_EXM_EXCH_ID"] == "NSE")
            & (master["SEM_INSTRUMENT_NAME"] == "EQUITY")
            & (master["SEM_TRADING_SYMBOL"].str.upper() == sym),
            "SEM_SMST_SECURITY_ID",
        ]
    if match.empty:
        return None
    return str(int(match.iloc[0]))


def _response_to_ohlcv(response: dict, dhan) -> pd.DataFrame | None:
    if not response or response.get("status") != "success":
        remarks = response.get("remarks") if response else None
        logger.debug("Dhan historical failed: %s", remarks)
        return None

    data = response.get("data") or {}
    required = ("open", "high", "low", "close", "volume", "timestamp")
    if not all(k in data and data[k] for k in required):
        return None

    try:
        if hasattr(dhan, "convert_to_date_time"):
            index = pd.to_datetime([dhan.convert_to_date_time(ts) for ts in data["timestamp"]])
        else:
            index = pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert(IST)
    except Exception:
        index = pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert(IST)

    frame = pd.DataFrame(
        {
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
        },
        index=index,
    )
    frame.index = frame.index.normalize()
    return frame.sort_index()


def fetch_daily_history(
    security_id: str,
    instrument_type: str,
    exchange_segment: str,
    days: int,
    config: dict,
) -> pd.DataFrame | None:
    """Fetch daily OHLCV for one security_id via Dhan historical API."""
    dhan = get_dhan_client()
    to_date = datetime.now(IST).date()
    # Buffer calendar days for weekends/holidays
    from_date = to_date - timedelta(days=int(days * 1.6) + 30)

    delay = float(config.get("dhan", {}).get("request_delay_sec", 0.05))
    if delay > 0:
        time.sleep(delay)

    try:
        response = dhan.historical_daily_data(
            str(security_id),
            exchange_segment,
            instrument_type,
            from_date.isoformat(),
            to_date.isoformat(),
            expiry_code=0,
            oi=False,
        )
    except DhanCredentialsError:
        raise
    except Exception:
        logger.exception("Dhan historical_daily_data error for security_id=%s", security_id)
        return None

    frame = _response_to_ohlcv(response, dhan)
    if frame is None or frame.empty:
        return None
    return frame.tail(days).copy()


def fetch_symbol_history(symbol: str, days: int, root: Path, config: dict) -> pd.DataFrame | None:
    from dhanhq import dhanhq as dhan_mod

    security_id = symbol_to_security_id(symbol, root, config)
    if not security_id:
        logger.debug("No Dhan security_id for symbol %s", symbol)
        return None
    return fetch_daily_history(
        security_id,
        "EQUITY",
        dhan_mod.NSE,
        days,
        config,
    )


def fetch_nifty_benchmark(days: int, root: Path, config: dict) -> pd.DataFrame | None:
    from dhanhq import dhanhq as dhan_mod

    return fetch_daily_history(
        NIFTY_SECURITY_ID,
        "INDEX",
        dhan_mod.INDEX,
        days,
        config,
    )
