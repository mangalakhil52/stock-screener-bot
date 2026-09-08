"""Track recent picks to avoid repeat recommendations within a cooldown window."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from ranker import TradePick

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


def _today() -> date:
    return datetime.now(IST).date()


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read pick history at %s — starting fresh", path)
        return []


def save_history(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2)


def recent_symbols(records: list[dict], cooldown_days: int, as_of: date | None = None) -> set[str]:
    """Symbols picked within the last `cooldown_days` trading days (calendar days)."""
    if cooldown_days <= 0:
        return set()

    today = as_of or _today()
    cutoff = today - timedelta(days=cooldown_days)
    blocked: set[str] = set()

    for row in records:
        symbol = str(row.get("symbol", "")).upper()
        if not symbol:
            continue
        try:
            picked_on = date.fromisoformat(str(row["date"]))
        except (KeyError, ValueError):
            continue
        if picked_on >= cutoff:
            blocked.add(symbol)

    return blocked


def append_picks(path: Path, picks: list[TradePick]) -> None:
    if not picks:
        return

    records = load_history(path)
    today_str = _today().isoformat()
    for pick in picks:
        records.append(
            {
                "date": today_str,
                "symbol": pick.symbol,
                "setup": pick.setup,
                "entry": pick.entry,
                "stop_loss": pick.stop_loss,
                "target_low": pick.target_low,
                "target_high": pick.target_high,
                "score": pick.score,
            }
        )

    # Keep last 180 days to avoid unbounded growth.
    cutoff = _today() - timedelta(days=180)
    trimmed = []
    for row in records:
        try:
            if date.fromisoformat(str(row["date"])) >= cutoff:
                trimmed.append(row)
        except ValueError:
            continue

    save_history(path, trimmed)
    logger.info("Saved %s pick(s) to history (%s total rows)", len(picks), len(trimmed))
