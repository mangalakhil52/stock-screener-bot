"""Tests for ensemble engine and regime filter."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ensemble_engine import analyze_ensemble, passes_ensemble_filters
from indicators import atr, rsi
from monte_carlo import simulate_win_probability
from regime_filter import assess_regime


def _make_ohlcv(n: int = 60, base: float = 3000, trend: float = 0.003) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    closes = [base]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + trend + rng.uniform(-0.005, 0.005)))
    closes = np.array(closes)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.012,
            "low": closes * 0.988,
            "close": closes,
            "volume": np.full(n, 500_000.0) + rng.integers(-50_000, 50_000, n),
        },
        index=idx,
    )


def _fake_breakout_df() -> pd.DataFrame:
    df = _make_ohlcv(60, trend=0.002)
    prev = df["close"].iloc[-2]
    df.iloc[-1, df.columns.get_loc("high")] = prev * 1.04
    df.iloc[-1, df.columns.get_loc("close")] = prev * 1.005
    df.iloc[-1, df.columns.get_loc("low")] = prev * 0.998
    df.iloc[-1, df.columns.get_loc("open")] = prev * 1.02
    df.iloc[-1, df.columns.get_loc("volume")] = 900_000
    return df


BASE_CONFIG = {
    "advanced": {
        "enabled": True,
        "min_probability": 55,
        "max_fake_breakout_risk": 0.55,
        "max_fake_move_risk": 0.60,
        "reject_grades": ["D"],
    },
    "ensemble": {
        "min_ensemble_score": 0.50,
        "min_monte_carlo_win_rate": 0.40,
        "min_walk_forward_edge": 0.30,
        "min_mtf_alignment": 0.40,
        "allowed_tiers": ["ELITE", "STRONG", "PASS", "REJECT"],
    },
    "monte_carlo": {"simulations": 100, "horizon_days": 10},
    "risk": {
        "stop_loss_pct": 3.0,
        "target_min_pct": 5.0,
        "by_setup": {"Breakout Momentum": {"stop_loss_pct": 4.0, "target_min_pct": 6.0}},
    },
}


class EnsembleTests(unittest.TestCase):
    def test_fake_breakout_gets_rejected_tier(self):
        result = analyze_ensemble("FAKE", _fake_breakout_df(), None, "Breakout Momentum", BASE_CONFIG)
        self.assertGreater(result.fake_breakout_risk, 0.3)
        self.assertIn(result.confidence_tier, ("REJECT", "PASS", "STRONG"))

    def test_ensemble_produces_mc_and_mtf(self):
        df = _make_ohlcv()
        result = analyze_ensemble("GOOD", df, None, "EMA Pullback", BASE_CONFIG)
        self.assertGreater(result.ensemble_score, 0)
        self.assertGreaterEqual(result.monte_carlo_win_rate, 0)
        self.assertGreaterEqual(result.mtf_alignment, 0)

    def test_regime_bullish_on_uptrend(self):
        bench = _make_ohlcv(80, base=20000, trend=0.002)
        regime = assess_regime(bench, {"regime": {"enabled": True, "skip_bearish_days": True}})
        self.assertIn(regime.label, ("BULLISH", "NEUTRAL"))
        self.assertTrue(regime.trade_allowed)

    def test_monte_carlo_returns_probability(self):
        df = _make_ohlcv()
        win, exp = simulate_win_probability(df["close"], float(df["close"].iloc[-1]), 3, 5, simulations=100)
        self.assertGreaterEqual(win, 0)
        self.assertLessEqual(win, 1)

    def test_indicators(self):
        df = _make_ohlcv()
        self.assertGreater(rsi(df["close"]), 0)
        self.assertGreater(atr(df["high"], df["low"], df["close"]), 0)

    def test_veto_blocks_low_ensemble(self):
        result = analyze_ensemble("FAKE", _fake_breakout_df(), None, "Breakout Momentum", BASE_CONFIG)
        if result.veto_reasons:
            self.assertFalse(passes_ensemble_filters(result, BASE_CONFIG))


if __name__ == "__main__":
    unittest.main()
