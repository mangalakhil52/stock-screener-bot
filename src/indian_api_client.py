"""Indian Stock Market API client — NSE universe and OHLCV via indianapi.in."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")

DEFAULT_BASE_URL = "https://stock.indianapi.in"
DEFAULT_STOCK_LIST_URL = "https://analyst.indianapi.in/static/all_stocks.json"

_UNIVERSE_CACHE: list[str] | None = None


class IndianApiCredentialsError(RuntimeError):
    """Raised when INDIAN_API_KEY is missing."""


def get_api_key() -> str:
    key = os.environ.get("INDIAN_API_KEY", "").strip()
    if not key:
        raise IndianApiCredentialsError(
            "Indian API key missing. Set INDIAN_API_KEY in .env (local) or GitHub Actions "
            "secrets (CI). Get a free key at https://indianapi.in/indian-stock-market"
        )
    return key


def get_base_url(config: dict) -> str:
    env_url = os.environ.get("INDIAN_API_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    return str(config.get("indian_api", {}).get("base_url", DEFAULT_BASE_URL)).rstrip("/")


def _api_cfg(config: dict) -> dict:
    return config.get("indian_api", {})


def _request(
    path: str,
    config: dict,
    params: dict | None = None,
    timeout: int = 30,
) -> dict | list | None:
    api_cfg = _api_cfg(config)
    delay = float(api_cfg.get("request_delay_sec", 0.05))
    if delay > 0:
        time.sleep(delay)

    url = f"{get_base_url(config)}{path}"
    headers = {"X-API-Key": get_api_key()}
    try:
        response = requests.get(url, headers=headers, params=params or {}, timeout=timeout)
        if response.status_code == 429:
            logger.warning("Indian API rate limit hit for %s", path)
            return None
        if response.status_code == 401:
            raise IndianApiCredentialsError(
                "Invalid Indian API key or plan URL mismatch. Free/Hobby plans use "
                "https://stock.indianapi.in — set INDIAN_API_BASE_URL if you upgraded."
            )
        response.raise_for_status()
        return response.json()
    except IndianApiCredentialsError:
        raise
    except requests.RequestException:
        logger.exception("Indian API request failed: %s", path)
        return None


def _universe_cache_path(root: Path, config: dict) -> Path:
    rel = _api_cfg(config).get("universe_cache", "data/cache/nse_universe.json")
    return root / rel


def _load_stock_list(root: Path, config: dict) -> list[dict]:
    cache_path = _universe_cache_path(root, config)
    max_age_h = float(_api_cfg(config).get("universe_cache_hours", 24))
    list_url = str(_api_cfg(config).get("stock_list_url", DEFAULT_STOCK_LIST_URL))

    if cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < max_age_h:
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                logger.debug("Universe cache unreadable — refreshing")

    logger.info("Downloading NSE stock list from Indian API...")
    try:
        response = requests.get(list_url, timeout=60)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        logger.exception("Failed to download stock list from %s", list_url)
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        return []

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def get_nse_equity_universe(root: Path, config: dict) -> list[str]:
    """All NSE-listed symbols from Indian API stock master (no cap)."""
    global _UNIVERSE_CACHE
    if _UNIVERSE_CACHE is not None:
        return _UNIVERSE_CACHE

    rows = _load_stock_list(root, config)
    symbols: list[str] = []
    seen: set[str] = set()
    for row in rows:
        code = str(row.get("nse-code", "")).strip().upper()
        if code and code not in seen:
            seen.add(code)
            symbols.append(code)

    symbols.sort()
    _UNIVERSE_CACHE = symbols
    logger.info("NSE equity universe from Indian API: %s symbols", len(symbols))
    return symbols


def days_to_period(days: int) -> str:
    if days <= 31:
        return "1m"
    if days <= 183:
        return "6m"
    if days <= 365:
        return "1yr"
    if days <= 365 * 3:
        return "3yr"
    if days <= 365 * 5:
        return "5yr"
    if days <= 365 * 10:
        return "10yr"
    return "max"


def _parse_historical_payload(payload: dict | list | None) -> pd.DataFrame | None:
    if not payload or not isinstance(payload, dict):
        return None

    price_rows: list = []
    volume_rows: list = []
    for dataset in payload.get("datasets", []):
        metric = str(dataset.get("metric", "")).lower()
        values = dataset.get("values") or []
        if metric == "price":
            price_rows = values
        elif metric == "volume":
            volume_rows = values

    if not price_rows:
        return None

    dates: list[pd.Timestamp] = []
    closes: list[float] = []
    for row in price_rows:
        if not row or len(row) < 2:
            continue
        dates.append(pd.Timestamp(row[0]))
        closes.append(float(row[1]))

    vol_by_date: dict[pd.Timestamp, float] = {}
    for row in volume_rows:
        if not row or len(row) < 2:
            continue
        vol_by_date[pd.Timestamp(row[0])] = float(row[1])

    if not dates:
        return None

    frame = pd.DataFrame({"close": closes}, index=pd.DatetimeIndex(dates))
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame["volume"] = [vol_by_date.get(idx, 0.0) for idx in frame.index]
    frame["open"] = frame["close"].shift(1).fillna(frame["close"])
    frame["high"] = frame[["open", "close"]].max(axis=1)
    frame["low"] = frame[["open", "close"]].min(axis=1)
    return frame


def fetch_symbol_history(
    symbol: str,
    days: int,
    root: Path,
    config: dict,
) -> pd.DataFrame | None:
    sym = symbol.strip().upper()
    period = days_to_period(days)
    params = {"stock_name": sym, "period": period, "filter": "price"}

    payload = _request("/historical_data", config, params=params)
    frame = _parse_historical_payload(payload)
    if frame is None or frame.empty:
        # Docs sometimes show `symbol` instead of `stock_name`
        payload = _request("/historical_data", config, params={"symbol": sym, "period": period, "filter": "price"})
        frame = _parse_historical_payload(payload)

    if frame is None or frame.empty:
        logger.debug("No Indian API history for %s", sym)
        return None
    return frame.tail(days).copy()


def fetch_nifty_benchmark(days: int, root: Path, config: dict) -> pd.DataFrame | None:
    """Nifty benchmark via index proxy symbols (Indian API has no index OHLC endpoint on free tier)."""
    candidates = _api_cfg(config).get(
        "benchmark_symbols",
        ["NIFTYBEES", "NIFTY", "NIFTY50"],
    )
    for name in candidates:
        frame = fetch_symbol_history(str(name), days, root, config)
        if frame is not None and len(frame) >= 30:
            logger.info("Using %s as Nifty benchmark proxy (%s bars)", name, len(frame))
            return frame
    logger.warning("Could not load Nifty benchmark from Indian API")
    return None
