"""Tests for advanced signal engine."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from advanced_analyzer import analyze_symbol, passes_advanced_filters

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
    "risk": {"stop_loss_pct": 3.0, "target_min_pct": 5.0, "by_setup": {}},
}


def _make_ohlcv(
    n: int = 60,
    base: float = 3000,
    trend: float = 0.003,
    vol_base: float = 500_000,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    closes = [base]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + trend + rng.uniform(-0.005, 0.005)))
    closes = np.array(closes)
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.012,
            "low": closes * 0.988,
            "close": closes,
            "volume": (np.full(n, vol_base) + rng.integers(-50_000, 50_000, n)).astype(float),
        },
        index=pd.date_range("2025-01-01", periods=n, freq="B"),
    )


def _fake_breakout_df() -> pd.DataFrame:
    df = _make_ohlcv(60, base=3000, trend=0.002)
    prev = df["close"].iloc[-2]
    df.iloc[-1, df.columns.get_loc("high")] = prev * 1.04
    df.iloc[-1, df.columns.get_loc("close")] = prev * 1.005
    df.iloc[-1, df.columns.get_loc("low")] = prev * 0.998
    df.iloc[-1, df.columns.get_loc("open")] = prev * 1.02
    df.iloc[-1, df.columns.get_loc("volume")] = 900_000
    return df


def _strong_breakout_df() -> pd.DataFrame:
    df = _make_ohlcv(60, base=3000, trend=0.002)
    prev = df["close"].iloc[-2]
    df.iloc[-1, df.columns.get_loc("open")] = prev * 1.01
    df.iloc[-1, df.columns.get_loc("high")] = prev * 1.035
    df.iloc[-1, df.columns.get_loc("low")] = prev * 1.008
    df.iloc[-1, df.columns.get_loc("close")] = prev * 1.032
    df.iloc[-1, df.columns.get_loc("volume")] = df["volume"].mean() * 2.2
    return df


def _distribution_df() -> pd.DataFrame:
    df = _make_ohlcv(60, base=3000, trend=0.0)
    last = df["close"].iloc[-2]
    df.iloc[-1, df.columns.get_loc("open")] = last
    df.iloc[-1, df.columns.get_loc("high")] = last * 1.004
    df.iloc[-1, df.columns.get_loc("low")] = last * 0.996
    df.iloc[-1, df.columns.get_loc("close")] = last * 1.001
    df.iloc[-1, df.columns.get_loc("volume")] = df["volume"].mean() * 3.0
    return df


class AdvancedAnalyzerTests(unittest.TestCase):
    def test_detects_fake_breakout(self):
        signals = analyze_symbol("FAKE", _fake_breakout_df(), setup="Breakout Momentum", config=BASE_CONFIG)
        self.assertGreater(signals.fake_breakout_risk, 0.3)
        self.assertTrue(signals.warnings)

    def test_strong_breakout_scores_higher(self):
        fake = analyze_symbol("FAKE", _fake_breakout_df(), setup="Breakout Momentum", config=BASE_CONFIG)
        strong = analyze_symbol("GOOD", _strong_breakout_df(), setup="Breakout Momentum", config=BASE_CONFIG)
        self.assertGreater(strong.probability, fake.probability)
        self.assertLess(strong.fake_breakout_risk, fake.fake_breakout_risk)

    def test_passes_advanced_filters(self):
        bad = analyze_symbol("FAKE", _fake_breakout_df(), setup="Breakout Momentum", config=BASE_CONFIG)
        if bad.veto_reasons:
            self.assertFalse(passes_advanced_filters(bad, BASE_CONFIG))

    def test_detects_distribution(self):
        signals = analyze_symbol("DIST", _distribution_df(), setup="Range Breakout", config=BASE_CONFIG)
        self.assertGreater(signals.fake_move_risk, 0.1)

    def test_grade_for_strong_setup(self):
        signals = analyze_symbol("GOOD", _strong_breakout_df(), setup="Breakout Momentum", config=BASE_CONFIG)
        self.assertIn(signals.grade, ("A", "B", "C"))
        self.assertGreater(signals.probability, 50)


if __name__ == "__main__":
    unittest.main()
