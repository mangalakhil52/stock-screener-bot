"""Tests for ranker and trade history."""

import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chartink_client import StockCandidate
from ranker import aggregate_candidates, build_picks, passes_filters, resolve_risk
from trade_history import append_picks, load_history, recent_symbols


def _candidate(symbol: str, setup: str, weight: float, close: float = 3000, chg: float = 3.0) -> StockCandidate:
    return StockCandidate(
        symbol=symbol,
        name=symbol,
        close=close,
        per_chg=chg,
        volume=50000,
        setup=setup,
        setup_weight=weight,
    )


BASE_CONFIG = {
    "picks": {"max_daily_picks": 3, "max_candidates": 40},
    "filters": {
        "price_min": 100,
        "price_max": 10000,
        "min_turnover_inr": 50000000,
        "max_daily_change_pct": 8.0,
        "reject_overextended": True,
    },
    "risk": {
        "stop_loss_pct": 3.0,
        "target_min_pct": 5.0,
        "target_max_pct": 10.0,
        "partial_exit_pct": 50.0,
        "breakeven_after_pct": 4.0,
        "by_setup": {
            "Breakout Momentum": {
                "stop_loss_pct": 4.0,
                "target_min_pct": 6.0,
                "target_max_pct": 12.0,
            },
            "EMA Pullback": {
                "stop_loss_pct": 2.5,
                "target_min_pct": 5.0,
                "target_max_pct": 8.0,
            },
        },
    },
    "ranking": {
        "volume_ratio": 0.25,
        "momentum_sweet_spot": 0.25,
        "liquidity": 0.20,
        "setup_weight": 0.10,
        "confluence": 0.10,
        "price_stability": 0.10,
    },
    "advanced": {"enabled": False},
    "ml": {"enabled": False},
}


class RankerTests(unittest.TestCase):
    def test_confluence_boosts_multi_setup_stocks(self):
        single = [_candidate("AAA", "Breakout Momentum", 1.0)]
        multi = [
            _candidate("BBB", "Breakout Momentum", 1.0),
            _candidate("BBB", "EMA Pullback", 0.9),
        ]
        single_pick = build_picks(single, BASE_CONFIG)[0]
        multi_pick = build_picks(multi, BASE_CONFIG)[0]
        self.assertEqual(multi_pick.setup_count, 2)
        self.assertGreater(multi_pick.score, single_pick.score)

    def test_setup_specific_risk(self):
        agg = aggregate_candidates([_candidate("X", "EMA Pullback", 0.9, close=2000)])[0]
        risk = resolve_risk(agg, BASE_CONFIG)
        self.assertEqual(risk["stop_loss_pct"], 2.5)
        self.assertEqual(risk["target_max_pct"], 8.0)

    def test_rejects_overextended_and_low_liquidity(self):
        overextended = aggregate_candidates(
            [_candidate("OT", "Breakout Momentum", 1.0, chg=9.5)]
        )[0]
        low_liq = aggregate_candidates(
            [_candidate("LL", "Breakout Momentum", 1.0, close=500, chg=3.0)]
        )[0]
        low_liq.volume = 1000  # turnover << 5 Cr
        self.assertFalse(passes_filters(overextended, BASE_CONFIG))
        self.assertFalse(passes_filters(low_liq, BASE_CONFIG))

    def test_cooldown_blocks_recent_symbol(self):
        candidates = [
            _candidate("NETWEB", "Breakout Momentum", 1.0),
            _candidate("HAL", "Range Breakout", 0.85, close=4800),
        ]
        picks = build_picks(candidates, BASE_CONFIG, blocked_symbols={"NETWEB"})
        symbols = {p.symbol for p in picks}
        self.assertNotIn("NETWEB", symbols)
        self.assertIn("HAL", symbols)


class TradeHistoryTests(unittest.TestCase):
    def test_recent_symbols_respects_cooldown(self):
        records = [
            {"date": "2026-08-20", "symbol": "NETWEB"},
            {"date": "2026-08-01", "symbol": "OLD"},
        ]
        blocked = recent_symbols(records, cooldown_days=7, as_of=date(2026, 8, 24))
        self.assertIn("NETWEB", blocked)
        self.assertNotIn("OLD", blocked)

    def test_append_picks_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.json"
            picks = build_picks([_candidate("HAL", "Range Breakout", 0.85, close=4800)], BASE_CONFIG)
            append_picks(path, picks)
            saved = load_history(path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["symbol"], "HAL")


if __name__ == "__main__":
    unittest.main()
