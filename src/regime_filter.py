"""Market regime detection — gate trades on unfavorable conditions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from indicators import returns_pct, rsi, sma


@dataclass
class MarketRegime:
    label: str              # BULLISH / NEUTRAL / BEARISH
    score: float            # 0–1
    trade_allowed: bool
    min_probability_boost: float  # raise bar in weak regimes
    reasons: list[str]

    @property
    def summary(self) -> str:
        return f"{self.label} ({self.score:.0%}) — {', '.join(self.reasons[:2])}"


def assess_regime(benchmark: pd.DataFrame | None, config: dict) -> MarketRegime:
    regime_cfg = config.get("regime", {})
    if not regime_cfg.get("enabled", True) or benchmark is None or len(benchmark) < 30:
        return MarketRegime(
            label="NEUTRAL",
            score=0.5,
            trade_allowed=True,
            min_probability_boost=0.0,
            reasons=["Regime data unavailable"],
        )

    close = benchmark["close"]
    last = float(close.iloc[-1])
    sma20 = float(sma(close, 20).iloc[-1])
    sma50 = float(sma(close, 50).iloc[-1])
    nifty_rsi = rsi(close)
    ret_5 = returns_pct(close, 5)
    ret_20 = returns_pct(close, 20)

    score_parts: list[float] = []
    reasons: list[str] = []

    if last > sma20 > sma50:
        score_parts.append(1.0)
        reasons.append("Nifty above 20/50 SMA")
    elif last > sma50:
        score_parts.append(0.65)
        reasons.append("Nifty above 50 SMA")
    else:
        score_parts.append(0.2)
        reasons.append("Nifty below 50 SMA")

    if ret_20 > 2:
        score_parts.append(0.9)
    elif ret_20 > 0:
        score_parts.append(0.6)
    else:
        score_parts.append(0.25)
        reasons.append(f"Nifty 20d return {ret_20:+.1f}%")

    if 45 <= nifty_rsi <= 65:
        score_parts.append(0.8)
    elif nifty_rsi > 70:
        score_parts.append(0.4)
        reasons.append(f"Nifty RSI {nifty_rsi:.0f} overbought")
    elif nifty_rsi < 35:
        score_parts.append(0.35)
        reasons.append(f"Nifty RSI {nifty_rsi:.0f} oversold — fragile bounce")
    else:
        score_parts.append(0.55)

    if ret_5 < -2:
        score_parts.append(0.2)
        reasons.append(f"Nifty falling {ret_5:.1f}% in 5d")

    score = sum(score_parts) / len(score_parts)

    if score >= 0.72:
        label = "BULLISH"
        boost = 0.0
    elif score >= 0.48:
        label = "NEUTRAL"
        boost = float(regime_cfg.get("neutral_probability_boost", 5.0))
    else:
        label = "BEARISH"
        boost = float(regime_cfg.get("bearish_probability_boost", 12.0))

    skip_bearish = regime_cfg.get("skip_bearish_days", True)
    trade_allowed = not (skip_bearish and label == "BEARISH")

    return MarketRegime(
        label=label,
        score=round(score, 3),
        trade_allowed=trade_allowed,
        min_probability_boost=boost,
        reasons=reasons[:4],
    )
