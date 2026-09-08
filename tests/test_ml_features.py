"""Tests for ML feature extraction and labeling."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ml_features import (
    MARKET_FEATURE_NAMES,
    extract_features_at,
    features_to_array,
    is_setup_like_day,
    label_forward_outcome,
)


def _ohlcv(n: int = 80, trend: float = 0.002) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    closes = [3000.0]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + trend + rng.uniform(-0.004, 0.004)))
    closes = np.array(closes)
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.015,
            "low": closes * 0.985,
            "close": closes,
            "volume": np.full(n, 600_000.0),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="B"),
    )


class MLFeatureTests(unittest.TestCase):
    def test_extract_features_shape(self):
        df = _ohlcv()
        feats = extract_features_at(df, len(df) - 1)
        self.assertIsNotNone(feats)
        arr = features_to_array(feats)
        self.assertEqual(len(arr), len(MARKET_FEATURE_NAMES))

    def test_label_forward_outcome(self):
        df = _ohlcv()
        # Force a winning path
        idx = 40
        entry = float(df["close"].iloc[idx])
        for j in range(idx + 1, idx + 6):
            df.iloc[j, df.columns.get_loc("high")] = entry * 1.06
            df.iloc[j, df.columns.get_loc("low")] = entry * 0.99
        label = label_forward_outcome(df, idx, target_pct=5.0, stop_pct=3.0, horizon=5)
        self.assertEqual(label, 1)

    def test_setup_like_day(self):
        df = _ohlcv()
        self.assertIsInstance(is_setup_like_day(df, len(df) - 1), bool)


if __name__ == "__main__":
    unittest.main()
