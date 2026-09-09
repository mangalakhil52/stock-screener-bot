"""Tests for Indian API client parsing and universe (no live API key required)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import indian_api_client as iac
from indian_api_client import _parse_historical_payload, days_to_period, get_nse_equity_universe


class IndianApiClientTests(unittest.TestCase):
    def test_days_to_period(self):
        self.assertEqual(days_to_period(20), "1m")
        self.assertEqual(days_to_period(90), "6m")
        self.assertEqual(days_to_period(365), "1yr")

    def test_parse_historical_payload(self):
        payload = {
            "datasets": [
                {
                    "metric": "Price",
                    "values": [
                        ["2024-01-01", "100.0"],
                        ["2024-01-02", "102.0"],
                        ["2024-01-03", "101.0"],
                    ],
                },
                {
                    "metric": "Volume",
                    "values": [
                        ["2024-01-01", 1000, {}],
                        ["2024-01-02", 1200, {}],
                        ["2024-01-03", 900, {}],
                    ],
                },
            ]
        }
        frame = _parse_historical_payload(payload)
        self.assertIsNotNone(frame)
        self.assertEqual(len(frame), 3)
        self.assertIn("close", frame.columns)
        self.assertIn("volume", frame.columns)
        self.assertIn("open", frame.columns)
        self.assertAlmostEqual(float(frame["close"].iloc[-1]), 101.0)

    @patch("indian_api_client._load_stock_list")
    def test_get_nse_equity_universe(self, mock_load):
        iac._UNIVERSE_CACHE = None
        mock_load.return_value = [
            {"nse-code": "RELIANCE", "name": "Reliance"},
            {"nse-code": "TCS", "name": "TCS"},
            {"nse-code": "", "name": "BSE only"},
            {"nse-code": "RELIANCE", "name": "dup"},
        ]
        symbols = get_nse_equity_universe(Path("/tmp"), {})
        self.assertEqual(symbols, ["RELIANCE", "TCS"])


if __name__ == "__main__":
    unittest.main()
