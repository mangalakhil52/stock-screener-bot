"""Advanced multi-signal analysis: fake breakouts, distribution, probability."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AdvancedSignals:
    symbol: str
    probability: float              # 0–100 composite move probability
    fake_breakout_risk: float         # 0–1 higher = more likely trap
    fake_move_risk: float             # 0–1 distribution / churn risk
    trend_strength: float             # 0–1
    volume_quality: float             # 0–1 institutional participation proxy
    relative_strength: float          # 0–1 vs Nifty
    grade: str                        # A / B / C / D
    warnings: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)

    @property
    def is_high_risk(self) -> bool:
        return self.fake_breakout_risk >= 0.55 or self.fake_move_risk >= 0.60

    @property
    def summary(self) -> str:
        parts = [f"P={self.probability:.0f}%", f"grade={self.grade}"]
        if self.warnings:
            parts.append(f"⚠ {self.warnings[0]}")
        elif self.confirmations:
            parts.append(f"✓ {self.confirmations[0]}")
        return " | ".join(parts)


def _rsi(series: pd.Series, period: int = 14) -> float:
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
    return float(100 - (100 / (1 + rs)))


def _sma(series: pd.Series, period: int) -> float:
    if len(series) < period:
        return float(series.iloc[-1])
    return float(series.tail(period).mean())


def _close_strength(close: float, high: float, low: float) -> float:
    """Where close sits in day's range: 1 = closed at high, 0 = at low."""
    span = high - low
    if span <= 0:
        return 0.5
    return (close - low) / span


def _upper_wick_ratio(open_: float, high: float, close: float, low: float) -> float:
    span = high - low
    if span <= 0:
        return 0.0
    body_top = max(open_, close)
    return (high - body_top) / span


def _returns(series: pd.Series, days: int) -> float:
    if len(series) <= days:
        return 0.0
    start = float(series.iloc[-days - 1])
    end = float(series.iloc[-1])
    if start == 0:
        return 0.0
    return (end - start) / start * 100


def analyze_symbol(
    symbol: str,
    df: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
    setup: str = "",
) -> AdvancedSignals:
    """Run composite signal engine on one symbol's OHLCV history."""
    warnings: list[str] = []
    confirmations: list[str] = []

    if df is None or len(df) < 25:
        return AdvancedSignals(
            symbol=symbol,
            probability=0.0,
            fake_breakout_risk=1.0,
            fake_move_risk=1.0,
            trend_strength=0.0,
            volume_quality=0.0,
            relative_strength=0.0,
            grade="D",
            warnings=["Insufficient price history"],
        )

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    last_close = float(close.iloc[-1])
    last_high = float(high.iloc[-1])
    last_low = float(low.iloc[-1])
    last_open = float(open_.iloc[-1])
    last_vol = float(volume.iloc[-1])

    vol_20 = _sma(volume, 20)
    vol_3 = float(volume.tail(3).mean())
    vol_10 = float(volume.tail(10).mean())
    rsi = _rsi(close)
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    high_20 = float(high.tail(20).max())
    high_5 = float(high.tail(5).max())

    high_20_prior = float(high.iloc[-21:-1].max()) if len(high) > 21 else float(high.iloc[:-1].max())
    high_5_prior = float(high.iloc[-6:-1].max()) if len(high) > 6 else high_20_prior

    close_str = _close_strength(last_close, last_high, last_low)
    wick = _upper_wick_ratio(last_open, last_high, last_close, last_low)
    daily_range_pct = (last_high - last_low) / last_close * 100 if last_close else 0
    vol_ratio = last_vol / vol_20 if vol_20 > 0 else 1.0

    # Broke prior resistance intraday but may have failed to hold.
    broke_resistance = last_high > high_20_prior * 1.001
    failed_hold = last_close < high_5_prior or close_str < 0.50

    # --- Fake breakout detection ---
    fake_breakout_flags: list[float] = []

    is_breakout_setup = setup in {"Breakout Momentum", "Range Breakout"}

    if is_breakout_setup and broke_resistance:
        if close_str < 0.42:
            fake_breakout_flags.append(0.35)
            warnings.append("Fake breakout: weak close in lower half of range")
        if wick > 0.45:
            fake_breakout_flags.append(0.30)
            warnings.append("Long upper wick — buyers rejected at highs")
        if vol_3 < vol_10 * 0.85 and vol_ratio < 1.2:
            fake_breakout_flags.append(0.25)
            warnings.append("Volume fading — breakout lacks participation")
        if rsi > 72:
            fake_breakout_flags.append(0.20)
            warnings.append(f"RSI {rsi:.0f} overbought — exhaustion risk")
        if failed_hold and last_close < high_5_prior:
            fake_breakout_flags.append(0.30)
            warnings.append("Intraday breakout failed — closed below 5-day high")

    fake_breakout_risk = min(sum(fake_breakout_flags), 1.0)

    # --- Fake move / distribution detection ---
    fake_move_flags: list[float] = []

    # High volume but tiny net move = churning / distribution
    if vol_ratio > 1.4 and abs(float(close.pct_change().iloc[-1]) * 100) < 0.8:
        fake_move_flags.append(0.35)
        warnings.append("High volume, flat price — possible distribution")

    # Gap up trap: opened >1.5% above prior close but closed weak
    if len(close) >= 2:
        prev_close = float(close.iloc[-2])
        gap_pct = (last_open - prev_close) / prev_close * 100 if prev_close else 0
        if gap_pct > 1.5 and close_str < 0.35:
            fake_move_flags.append(0.30)
            warnings.append("Gap-up fade — opened strong, closed weak")

    # Narrow range on huge volume (absorption)
    if vol_ratio > 2.0 and daily_range_pct < 1.2:
        fake_move_flags.append(0.25)
        warnings.append("Tight range on heavy volume — battle zone")

    # Three declining closes on rising volume
    if len(close) >= 4:
        last_3_closes = close.tail(3).values
        last_3_vols = volume.tail(3).values
        if (
            last_3_closes[0] > last_3_closes[1] > last_3_closes[2]
            and last_3_vols[1] > last_3_vols[0]
            and last_3_vols[2] > last_3_vols[1]
        ):
            fake_move_flags.append(0.30)
            warnings.append("Distribution pattern: falling price + rising volume")

    fake_move_risk = min(sum(fake_move_flags), 1.0)

    # --- Positive confirmations ---
    positive_scores: list[tuple[float, str]] = []

    if is_breakout_setup and broke_resistance and close_str > 0.72 and vol_ratio >= 1.5:
        positive_scores.append((0.20, "Breakout accepted — strong close on volume"))
    elif close_str > 0.72:
        positive_scores.append((0.18, "Strong close near day high — acceptance"))
    if vol_ratio >= 1.5:
        positive_scores.append((0.15, f"Volume {vol_ratio:.1f}× 20-day avg — conviction"))
    if 55 <= rsi <= 68:
        positive_scores.append((0.12, f"RSI {rsi:.0f} in momentum sweet spot"))
    if last_close > sma20 > sma50:
        positive_scores.append((0.15, "Bullish SMA stack (20 > 50)"))
    if sma20 > _sma(close.iloc[:-5], 20) if len(close) > 25 else sma20:
        positive_scores.append((0.08, "Rising 20-day trend"))

    # Relative strength vs Nifty
    rs_score = 0.5
    if benchmark is not None and len(benchmark) >= 10:
        stock_ret_5 = _returns(close, 5)
        bench_ret_5 = _returns(benchmark["close"], 5)
        stock_ret_10 = _returns(close, 10)
        bench_ret_10 = _returns(benchmark["close"], 10)
        if stock_ret_5 > bench_ret_5 and stock_ret_10 > bench_ret_10:
            rs_score = 0.9
            positive_scores.append((0.15, "Outperforming Nifty on 5d & 10d"))
        elif stock_ret_5 > bench_ret_5:
            rs_score = 0.7
            positive_scores.append((0.08, "Beating Nifty over 5 days"))
        elif stock_ret_5 < bench_ret_5 - 2:
            rs_score = 0.25
            warnings.append("Lagging Nifty — weak relative strength")
    relative_strength = rs_score

    # Trend strength composite
    trend_parts = [
        1.0 if last_close > sma50 else 0.0,
        1.0 if sma20 > sma50 else 0.3,
        min(max((last_close - sma20) / sma20 * 10, 0), 1.0) if sma20 else 0.5,
    ]
    trend_strength = float(np.mean(trend_parts))

    # Volume quality
    vol_quality_parts = [
        min(vol_ratio / 2.5, 1.0),
        1.0 if vol_3 >= vol_10 else 0.4,
        close_str,
    ]
    volume_quality = float(np.mean(vol_quality_parts))

    confirmations = [msg for _, msg in positive_scores]

    # --- Composite probability (0–100) ---
    base_prob = 50.0
    base_prob += sum(pts for pts, _ in positive_scores) * 100
    base_prob += trend_strength * 12
    base_prob += volume_quality * 10
    base_prob += relative_strength * 8
    base_prob -= fake_breakout_risk * 35
    base_prob -= fake_move_risk * 25

    if setup == "EMA Pullback" and last_close > sma20 and rsi < 65:
        base_prob += 5
        confirmations.append("Pullback holding above 20 EMA")

    probability = float(np.clip(base_prob, 0, 99))

    grade = _grade(probability, fake_breakout_risk, fake_move_risk)

    return AdvancedSignals(
        symbol=symbol,
        probability=round(probability, 1),
        fake_breakout_risk=round(fake_breakout_risk, 3),
        fake_move_risk=round(fake_move_risk, 3),
        trend_strength=round(trend_strength, 3),
        volume_quality=round(volume_quality, 3),
        relative_strength=round(relative_strength, 3),
        grade=grade,
        warnings=warnings[:4],
        confirmations=confirmations[:4],
    )


def _grade(probability: float, fake_bo: float, fake_mv: float) -> str:
    if fake_bo >= 0.55 or fake_mv >= 0.60:
        return "D"
    if probability >= 72 and fake_bo < 0.25:
        return "A"
    if probability >= 62 and fake_bo < 0.40:
        return "B"
    if probability >= 52:
        return "C"
    return "D"


def analyze_batch(
    symbols: list[str],
    history: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame | None,
    setups: dict[str, str],
) -> dict[str, AdvancedSignals]:
    results: dict[str, AdvancedSignals] = {}
    for symbol in symbols:
        df = history.get(symbol)
        setup = setups.get(symbol, "")
        results[symbol] = analyze_symbol(symbol, df, benchmark, setup)
    return results


def passes_advanced_filters(signals: AdvancedSignals, config: dict) -> bool:
    adv = config.get("advanced", {})
    if not adv.get("enabled", True):
        return True

    min_prob = float(adv.get("min_probability", 55))
    max_fake_bo = float(adv.get("max_fake_breakout_risk", 0.55))
    max_fake_mv = float(adv.get("max_fake_move_risk", 0.60))
    reject_grades = set(adv.get("reject_grades", ["D"]))

    if signals.probability < min_prob:
        return False
    if signals.fake_breakout_risk > max_fake_bo:
        return False
    if signals.fake_move_risk > max_fake_mv:
        return False
    if signals.grade in reject_grades:
        return False
    return True
