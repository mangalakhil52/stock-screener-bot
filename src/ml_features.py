"""ML feature extraction shared by training and live scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import (
    adx_proxy,
    atr,
    bollinger_bandwidth,
    close_strength,
    macd_histogram,
    obv_slope,
    returns_pct,
    rsi,
    sma,
)

# Features the daily market model learns on.
MARKET_FEATURE_NAMES: list[str] = [
    "rsi_norm",
    "vol_ratio",
    "above_sma20",
    "above_sma50",
    "sma20_above_sma50",
    "ret_5d",
    "ret_10d",
    "bb_width",
    "atr_pct",
    "close_strength",
    "macd_bullish",
    "adx_strength",
    "obv_slope_norm",
    "breakout_proximity",
    "bench_rel_5d",
    "gap_pct",
    "range_pct",
]


def extract_features_at(
    df: pd.DataFrame,
    idx: int,
    benchmark: pd.DataFrame | None = None,
) -> dict[str, float] | None:
    """Extract normalized feature vector at bar index `idx` (for training or inference)."""
    if df is None or idx < 30 or idx >= len(df):
        return None

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    sub_close = close.iloc[: idx + 1]
    sub_high = high.iloc[: idx + 1]
    sub_low = low.iloc[: idx + 1]
    sub_open = open_.iloc[: idx + 1]
    sub_vol = volume.iloc[: idx + 1]

    last_close = float(sub_close.iloc[-1])
    last_high = float(sub_high.iloc[-1])
    last_low = float(sub_low.iloc[-1])
    last_open = float(sub_open.iloc[-1])
    last_vol = float(sub_vol.iloc[-1])

    if last_close <= 0:
        return None

    vol_20 = float(sma(sub_vol, 20).iloc[-1])
    sma20 = float(sma(sub_close, 20).iloc[-1])
    sma50 = float(sma(sub_close, 50).iloc[-1])
    high_20 = float(sub_high.iloc[-20:].max())

    rsi_val = rsi(sub_close)
    vol_ratio = last_vol / vol_20 if vol_20 > 0 else 1.0
    atr_val = atr(sub_high, sub_low, sub_close)
    atr_pct = atr_val / last_close * 100 if last_close else 0
    bb_w = bollinger_bandwidth(sub_close)
    c_str = close_strength(last_close, last_high, last_low)

    ret_5 = returns_pct(sub_close, 5) / 10.0  # scale down
    ret_10 = returns_pct(sub_close, 10) / 15.0

    bench_rel = 0.0
    if benchmark is not None and len(benchmark) > idx + 5:
        bench_close = benchmark["close"].iloc[: idx + 1]
        stock_5 = returns_pct(sub_close, 5)
        bench_5 = returns_pct(bench_close, 5)
        bench_rel = (stock_5 - bench_5) / 10.0

    gap_pct = 0.0
    if idx >= 1:
        prev = float(close.iloc[idx - 1])
        gap_pct = (last_open - prev) / prev * 100 / 5.0 if prev else 0.0

    range_pct = (last_high - last_low) / last_close * 100 / 5.0
    breakout_prox = (last_close - high_20) / high_20 * 10 if high_20 else 0.0

    return {
        "rsi_norm": rsi_val / 100.0,
        "vol_ratio": min(vol_ratio / 3.0, 1.0),
        "above_sma20": 1.0 if last_close > sma20 else 0.0,
        "above_sma50": 1.0 if last_close > sma50 else 0.0,
        "sma20_above_sma50": 1.0 if sma20 > sma50 else 0.0,
        "ret_5d": float(np.clip(ret_5, -1, 1)),
        "ret_10d": float(np.clip(ret_10, -1, 1)),
        "bb_width": min(bb_w / 0.15, 1.0),
        "atr_pct": min(atr_pct / 8.0, 1.0),
        "close_strength": c_str,
        "macd_bullish": 1.0 if macd_histogram(sub_close) > 0 else 0.0,
        "adx_strength": adx_proxy(sub_high, sub_low, sub_close),
        "obv_slope_norm": float(np.clip(obv_slope(sub_close, sub_vol, 10) * 5, -1, 1)),
        "breakout_proximity": float(np.clip(breakout_prox, -1, 1)),
        "bench_rel_5d": float(np.clip(bench_rel, -1, 1)),
        "gap_pct": float(np.clip(gap_pct, -1, 1)),
        "range_pct": float(np.clip(range_pct, 0, 1)),
    }


def features_to_array(features: dict[str, float]) -> np.ndarray:
    return np.array([features.get(k, 0.0) for k in MARKET_FEATURE_NAMES], dtype=float)


def is_setup_like_day(df: pd.DataFrame, idx: int) -> bool:
    """Only train on days resembling swing scan setups (reduces noise)."""
    if idx < 30:
        return False
    close = float(df["close"].iloc[idx])
    if close < 100 or close > 10000:
        return False
    vol = float(df["volume"].iloc[idx])
    vol_avg = float(df["volume"].iloc[idx - 20 : idx].mean())
    if vol_avg <= 0:
        return False
    high_20 = float(df["high"].iloc[idx - 20 : idx].max())
    sma50 = float(df["close"].iloc[idx - 50 : idx].mean())
    vol_spike = vol > vol_avg * 1.15
    near_breakout = close >= high_20 * 0.98
    uptrend = close > sma50
    return (vol_spike and near_breakout) or (uptrend and vol > vol_avg)


def label_forward_outcome(
    df: pd.DataFrame,
    idx: int,
    target_pct: float = 5.0,
    stop_pct: float = 3.0,
    horizon: int = 10,
) -> int | None:
    """1 = target hit before stop within horizon, 0 = stop first or neither."""
    if idx + horizon >= len(df):
        return None
    entry = float(df["close"].iloc[idx])
    if entry <= 0:
        return None
    stop = entry * (1 - stop_pct / 100)
    target = entry * (1 + target_pct / 100)

    for j in range(idx + 1, idx + 1 + horizon):
        if float(df["low"].iloc[j]) <= stop:
            return 0
        if float(df["high"].iloc[j]) >= target:
            return 1
    return 0
