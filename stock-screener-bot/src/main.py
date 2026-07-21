"""Daily swing screener — fetch Chartink scans, rank picks, send alerts."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv  # noqa: F401 — loads .env into os.environ

from chartink_client import ChartinkClient
from notifier import format_message, notify
from ranker import build_picks

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    setup_logging()
    load_dotenv(ROOT / ".env")
    logger = logging.getLogger("main")

    config = load_config()
    client = ChartinkClient()

    logger.info("Running Chartink scans...")
    candidates = client.run_all_scans(config.get("scans", []))
    total = len(candidates)
    logger.info("Total raw hits: %s", total)

    unique_count = len(set(c.symbol for c in candidates))

    if total == 0:
        logger.warning("No stocks matched any scan today")
        try:
            notify([], 0, config)
        except RuntimeError as exc:
            logger.warning("%s — set up .env for Telegram alerts", exc)
        return 0

    picks = build_picks(candidates, config)
    logger.info("Top picks: %s", [p.symbol for p in picks])

    console_msg = format_message(picks, unique_count)
    print(console_msg.encode("ascii", errors="replace").decode("ascii"))

    try:
        notify(picks, unique_count, config)
    except RuntimeError as exc:
        logger.warning("%s — copy .env.example to .env and add Telegram credentials", exc)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logging.exception("Screener failed")
        raise SystemExit(1)
