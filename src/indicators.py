"""Technical indicator library for advanced analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=1).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    last_loss = float(loss.iloc[-1])
    if last_loss == 0:
        return 100.0
    rs = float(gain.iloc[-1]) / last_loss
    return float(100 - (100 / (1 + rs)))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    if len(close) < 2:
        return 0.0
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    val = tr.rolling(period, min_periods=1).mean().iloc[-1]
    return float(val) if not np.isnan(val) else 0.0


def macd_histogram(close: pd.Series) -> float:
    if len(close) < 26:
        return 0.0
    line = ema(close, 12) - ema(close, 26)
    signal = ema(line, 9)
    hist = line - signal
    return float(hist.iloc[-1])


def bollinger_bandwidth(close: pd.Series, period: int = 20) -> float:
    if len(close) < period:
        return 0.0
    mid = sma(close, period).iloc[-1]
    std = float(close.tail(period).std())
    if mid == 0:
        return 0.0
    return (4 * std) / mid


def adx_proxy(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Simplified trend-strength proxy (0–1)."""
    if len(close) < period + 1:
        return 0.5
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr_series = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr_series
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr_series
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = float(dx.rolling(period).mean().iloc[-1])
    if np.isnan(adx_val):
        return 0.5
    return min(adx_val / 50.0, 1.0)


def obv_slope(close: pd.Series, volume: pd.Series, lookback: int = 10) -> float:
    if len(close) < lookback + 1:
        return 0.0
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    recent = obv.tail(lookback)
    x = np.arange(len(recent))
    if len(x) < 2:
        return 0.0
    slope = np.polyfit(x, recent.values, 1)[0]
    return float(slope / max(abs(obv.iloc[-1]), 1))


def close_strength(close: float, high: float, low: float) -> float:
    span = high - low
    if span <= 0:
        return 0.5
    return (close - low) / span


def upper_wick_ratio(open_: float, high: float, close: float, low: float) -> float:
    span = high - low
    if span <= 0:
        return 0.0
    return (high - max(open_, close)) / span


def returns_pct(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return 0.0
    start = float(series.iloc[-days - 1])
    end = float(series.iloc[-1])
    if start == 0:
        return 0.0
    return (end - start) / start * 100


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly."""
    if df is None or df.empty:
        return df
    weekly = df.resample("W").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    return weekly
