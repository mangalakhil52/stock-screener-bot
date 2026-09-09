"""Tests for NSEPython client parsing (no live NSE calls)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import nse_client as nsc
from nse_client import (
    _normalize_equity_df,
    _normalize_index_df,
    get_nse_equity_universe,
)


class NseClientTests(unittest.TestCase):
    def test_normalize_equity_df(self):
        raw = pd.DataFrame(
            {
                "CH_TIMESTAMP": ["2024-01-01", "2024-01-02"],
                "CH_OPENING_PRICE": [100, 102],
                "CH_TRADE_HIGH_PRICE": [101, 104],
                "CH_TRADE_LOW_PRICE": [99, 101],
                "CH_CLOSING_PRICE": [100.5, 103],
                "CH_TOT_TRADED_QTY": [1000, 1200],
            }
        )
        frame = _normalize_equity_df(raw)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame), 2)
        self.assertAlmostEqual(float(frame["close"].iloc[-1]), 103.0)

    def test_normalize_index_df(self):
        raw = pd.DataFrame(
            {
                "HistoricalDate": ["01-Jan-2024", "02-Jan-2024"],
                "OPEN": [22000, 22100],
                "HIGH": [22150, 22200],
                "LOW": [21950, 22050],
                "CLOSE": [22100, 22180],
            }
        )
        frame = _normalize_index_df(raw)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame), 2)

    @patch("nse_client._load_nse_module")
    def test_get_nse_equity_universe(self, mock_load):
        nsc._UNIVERSE_CACHE = None
        mock_nse = mock_load.return_value
        mock_nse.nse_eq_symbols.return_value = ["reliance", "TCS", "reliance"]
        symbols = get_nse_equity_universe(Path("/tmp"), {})
        self.assertEqual(symbols, ["RELIANCE", "TCS"])


if __name__ == "__main__":
    unittest.main()
