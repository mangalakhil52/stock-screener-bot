"""Tests for advanced signal engine."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from advanced_analyzer import analyze_symbol, passes_advanced_filters


def _make_ohlcv(
    n: int = 60,
    base: float = 3000,
    trend: float = 0.003,
    vol_base: float = 500_000,
) -> pd.DataFrame:
    """Synthetic uptrending OHLCV (deterministic)."""
    rng = np.random.default_rng(42)
    closes = [base]
    for i in range(n - 1):
        closes.append(closes[-1] * (1 + trend + rng.uniform(-0.005, 0.005)))
    closes = np.array(closes)
    highs = closes * 1.012
    lows = closes * 0.988
    opens = closes * 0.999
    volumes = np.full(n, vol_base) + rng.integers(-50_000, 50_000, n)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes.astype(float)}
    )


def _fake_breakout_df() -> pd.DataFrame:
    """Breakout day with long upper wick and weak close."""
    df = _make_ohlcv(60, base=3000, trend=0.002)
    # Last bar: spike high but close near low
    df.iloc[-1, df.columns.get_loc("high")] = df["close"].iloc[-2] * 1.04
    df.iloc[-1, df.columns.get_loc("close")] = df["close"].iloc[-2] * 1.005
    df.iloc[-1, df.columns.get_loc("low")] = df["close"].iloc[-2] * 0.998
    df.iloc[-1, df.columns.get_loc("open")] = df["close"].iloc[-2] * 1.02
    df.iloc[-1, df.columns.get_loc("volume")] = df["volume"].mean() * 1.8
    return df


def _strong_breakout_df() -> pd.DataFrame:
    """Clean breakout: close near high, volume surge."""
    df = _make_ohlcv(60, base=3000, trend=0.002)
    prev = df["close"].iloc[-2]
    df.iloc[-1, df.columns.get_loc("open")] = prev * 1.01
    df.iloc[-1, df.columns.get_loc("high")] = prev * 1.035
    df.iloc[-1, df.columns.get_loc("low")] = prev * 1.008
    df.iloc[-1, df.columns.get_loc("close")] = prev * 1.032
    df.iloc[-1, df.columns.get_loc("volume")] = df["volume"].mean() * 2.2
    return df


def _distribution_df() -> pd.DataFrame:
    """High volume, flat price — distribution."""
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
        signals = analyze_symbol("FAKE", _fake_breakout_df(), setup="Breakout Momentum")
        self.assertGreater(signals.fake_breakout_risk, 0.4)
        self.assertTrue(any("wick" in w.lower() or "weak close" in w.lower() for w in signals.warnings))
        self.assertLess(signals.probability, 60)

    def test_strong_breakout_scores_higher(self):
        fake = analyze_symbol("FAKE", _fake_breakout_df(), setup="Breakout Momentum")
        strong = analyze_symbol("GOOD", _strong_breakout_df(), setup="Breakout Momentum")
        self.assertGreater(strong.probability, fake.probability)
        self.assertLess(strong.fake_breakout_risk, fake.fake_breakout_risk)

    def test_passes_advanced_filters(self):
        good = analyze_symbol("GOOD", _strong_breakout_df(), setup="Breakout Momentum")
        bad = analyze_symbol("FAKE", _fake_breakout_df(), setup="Breakout Momentum")
        config = {
            "advanced": {
                "enabled": True,
                "min_probability": 55,
                "max_fake_breakout_risk": 0.55,
                "max_fake_move_risk": 0.60,
                "reject_grades": ["D"],
            }
        }
        if good.probability >= 55 and good.fake_breakout_risk <= 0.55:
            self.assertTrue(passes_advanced_filters(good, config))
        self.assertFalse(passes_advanced_filters(bad, config))

    def test_detects_distribution(self):
        signals = analyze_symbol("DIST", _distribution_df(), setup="Range Breakout")
        self.assertGreater(signals.fake_move_risk, 0.2)
        self.assertTrue(signals.warnings)

    def test_grade_for_strong_setup(self):
        signals = analyze_symbol("GOOD", _strong_breakout_df(), setup="Breakout Momentum")
        self.assertIn(signals.grade, ("A", "B", "C"))
        self.assertGreater(signals.probability, 50)


if __name__ == "__main__":
    unittest.main()
