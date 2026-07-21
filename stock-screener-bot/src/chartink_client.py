"""Fetch stock screener results from Chartink."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

CHARTINK_SCREENER_URL = "https://chartink.com/screener/"
CHARTINK_PROCESS_URL = "https://chartink.com/screener/process"


@dataclass
class StockCandidate:
    symbol: str
    name: str
    close: float
    per_chg: float
    volume: float
    setup: str
    setup_weight: float


class ChartinkClient:
    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def _session_with_csrf(self) -> tuple[requests.Session, str]:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
        )
        response = session.get(CHARTINK_SCREENER_URL, timeout=self.timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "lxml")
        token_tag = soup.select_one('[name="csrf-token"]')
        if not token_tag or not token_tag.get("content"):
            raise RuntimeError("Could not fetch Chartink CSRF token")

        token = token_tag["content"]
        session.headers.update(
            {
                "X-CSRF-TOKEN": token,
                "Referer": CHARTINK_SCREENER_URL,
                "X-Requested-With": "XMLHttpRequest",
            }
        )
        return session, token

    def run_scan(
        self,
        scan_clause: str,
        setup_name: str,
        setup_weight: float = 1.0,
    ) -> list[StockCandidate]:
        session, _ = self._session_with_csrf()
        response = session.post(
            CHARTINK_PROCESS_URL,
            data={"scan_clause": scan_clause.strip()},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        records = payload.get("recordsTotal", 0)
        rows = payload.get("data") or []
        logger.info("Scan '%s' returned %s stocks", setup_name, records)

        candidates: list[StockCandidate] = []
        for row in rows:
            symbol = row.get("nsecode") or row.get("bsecode") or ""
            if not symbol:
                continue

            candidates.append(
                StockCandidate(
                    symbol=str(symbol).upper(),
                    name=str(row.get("name", symbol)),
                    close=_to_float(row.get("close")),
                    per_chg=_to_float(row.get("per_chg")),
                    volume=_to_float(row.get("volume")),
                    setup=setup_name,
                    setup_weight=setup_weight,
                )
            )
        return candidates

    def run_all_scans(self, scans: list[dict]) -> list[StockCandidate]:
        all_candidates: list[StockCandidate] = []
        for scan in scans:
            try:
                results = self.run_scan(
                    scan_clause=scan["scan_clause"],
                    setup_name=scan["name"],
                    setup_weight=float(scan.get("weight", 1.0)),
                )
                all_candidates.extend(results)
            except Exception:
                logger.exception("Failed scan: %s", scan.get("name"))
        return all_candidates


def _to_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
