"""Advanced multi-signal analysis — delegates to ensemble engine."""

from __future__ import annotations

import pandas as pd

from ensemble_engine import EnsembleResult, analyze_ensemble, passes_ensemble_filters


# Backward-compatible alias.
class AdvancedSignals(EnsembleResult):
    """Ensemble result exposed as AdvancedSignals for existing imports."""

    @property
    def is_high_risk(self) -> bool:
        return self.fake_breakout_risk >= 0.55 or self.fake_move_risk >= 0.60


def analyze_symbol(
    symbol: str,
    df: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    setup: str = "",
    config: dict | None = None,
) -> AdvancedSignals:
    cfg = config or {"advanced": {"enabled": True}}
    return analyze_ensemble(symbol, df, benchmark, setup, cfg)


def analyze_batch(
    symbols: list[str],
    history: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame | None,
    setups: dict[str, str],
    config: dict | None = None,
) -> dict[str, AdvancedSignals]:
    cfg = config or {"advanced": {"enabled": True}}
    results: dict[str, AdvancedSignals] = {}
    for symbol in symbols:
        df = history.get(symbol)
        setup = setups.get(symbol, "")
        results[symbol] = analyze_ensemble(symbol, df, benchmark, setup, cfg)
    return results


def passes_advanced_filters(signals: AdvancedSignals, config: dict) -> bool:
    return passes_ensemble_filters(signals, config)
