"""Tests for Dhan instrument master and universe selection (no API credentials required)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dhan_client import select_training_symbols, symbol_to_security_id


class DhanUniverseTests(unittest.TestCase):
    def test_select_training_symbols_rotation(self):
        universe = [f"S{i}" for i in range(10)]
        picked = select_training_symbols(universe, 3)
        self.assertEqual(len(picked), 3)
        self.assertTrue(all(s in universe for s in picked))

    def test_select_training_symbols_no_cap(self):
        universe = ["A", "B", "C"]
        self.assertEqual(select_training_symbols(universe, 0), universe)
        self.assertEqual(select_training_symbols(universe, 10), universe)

    @patch("dhan_client.load_instrument_master")
    def test_symbol_to_security_id(self, mock_load):
        import pandas as pd

        mock_load.return_value = pd.DataFrame(
            {
                "SEM_EXM_EXCH_ID": ["NSE", "NSE"],
                "SEM_INSTRUMENT_NAME": ["EQUITY", "EQUITY"],
                "SEM_SEGMENT": ["E", "E"],
                "SEM_SERIES": ["EQ", "EQ"],
                "SEM_TRADING_SYMBOL": ["RELIANCE", "TCS"],
                "SEM_SMST_SECURITY_ID": [2885, 11536],
            }
        )
        root = Path("/tmp")
        config = {"dhan": {"universe_series": ["EQ"]}}
        self.assertEqual(symbol_to_security_id("RELIANCE", root, config), "2885")
        self.assertIsNone(symbol_to_security_id("UNKNOWN", root, config))


if __name__ == "__main__":
    unittest.main()
