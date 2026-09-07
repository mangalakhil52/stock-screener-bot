"""Monte Carlo path simulation for target/stop hit probability."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_win_probability(
    close: pd.Series,
    entry: float,
    stop_pct: float,
    target_pct: float,
    horizon_days: int = 10,
    simulations: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap daily returns to estimate P(hit target before stop) within horizon.
    Returns (win_probability, expected_return_pct).
    """
    if len(close) < 30 or entry <= 0:
        return 0.5, 0.0

    returns = close.pct_change().dropna().values
    if len(returns) < 20:
        return 0.5, 0.0

    rng = np.random.default_rng(seed)
    stop_price = entry * (1 - stop_pct / 100)
    target_price = entry * (1 + target_pct / 100)

    wins = 0
    total_return = 0.0

    for _ in range(simulations):
        price = entry
        hit = None
        for _ in range(horizon_days):
            r = rng.choice(returns)
            price *= 1 + r
            if price <= stop_price:
                hit = "loss"
                break
            if price >= target_price:
                hit = "win"
                break
        if hit == "win":
            wins += 1
        total_return += (price - entry) / entry * 100

    win_prob = wins / simulations
    expected_return = total_return / simulations
    return round(win_prob, 3), round(expected_return, 2)
