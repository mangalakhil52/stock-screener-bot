"""Market regime detection — adjust strictness, don't block bearish markets entirely."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from indicators import returns_pct, rsi, sma


@dataclass
class MarketRegime:
    label: str              # BULLISH / NEUTRAL / BEARISH
    score: float            # 0–1
    mode: str               # normal / selective / blocked
    trade_allowed: bool
    min_probability_boost: float
    min_relative_strength: float    # 0–1, stock must clear this in bearish selective mode
    min_ml_score: float             # extra ML floor in bearish selective mode
    allowed_tiers: list[str] = field(default_factory=list)
    max_picks: int | None = None    # cap picks in weak regimes
    reasons: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        mode_note = ""
        if self.mode == "selective":
            mode_note = " | selective stock-picking (relative strength required)"
        elif self.mode == "blocked":
            mode_note = " | no new picks"
        parts = [f"{self.label} ({self.score:.0%})"]
        if self.reasons:
            parts.append(", ".join(self.reasons[:2]))
        return "".join(parts) + mode_note


def assess_regime(benchmark: pd.DataFrame | None, config: dict) -> MarketRegime:
    regime_cfg = config.get("regime", {})
    if not regime_cfg.get("enabled", True) or benchmark is None or len(benchmark) < 30:
        return MarketRegime(
            label="NEUTRAL",
            score=0.5,
            mode="normal",
            trade_allowed=True,
            min_probability_boost=0.0,
            min_relative_strength=0.0,
            min_ml_score=0.0,
            allowed_tiers=[],
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
        reasons.append(f"Nifty RSI {nifty_rsi:.0f} oversold")
    else:
        score_parts.append(0.55)

    if ret_5 < -2:
        score_parts.append(0.2)
        reasons.append(f"Nifty falling {ret_5:.1f}% in 5d")

    score = sum(score_parts) / len(score_parts)

    if score >= 0.72:
        label = "BULLISH"
        boost = 0.0
        mode = "normal"
        min_rs = 0.0
        min_ml = 0.0
        allowed_tiers: list[str] = []
        max_picks = None
    elif score >= 0.48:
        label = "NEUTRAL"
        boost = float(regime_cfg.get("neutral_probability_boost", 5.0))
        mode = "normal"
        min_rs = 0.0
        min_ml = 0.0
        allowed_tiers = []
        max_picks = None
    else:
        label = "BEARISH"
        boost = float(regime_cfg.get("bearish_probability_boost", 8.0))
        # selective = still trade stocks beating the index; strict = block all
        bearish_mode = regime_cfg.get("bearish_mode", "selective")
        if bearish_mode == "strict" or regime_cfg.get("skip_bearish_days", False):
            mode = "blocked"
            min_rs = 1.0  # unreachable — blocks via trade_allowed
            min_ml = 1.0
            allowed_tiers = []
            max_picks = 0
        else:
            mode = "selective"
            min_rs = float(regime_cfg.get("bearish_min_relative_strength", 0.65))
            min_ml = float(regime_cfg.get("bearish_min_ml_score", 0.50))
            allowed_tiers = list(
                regime_cfg.get("bearish_allowed_tiers", ["ELITE", "STRONG"])
            )
            max_picks = int(regime_cfg.get("bearish_max_picks", 2))
            reasons.append("Only index-beating setups allowed")

    trade_allowed = mode != "blocked"

    return MarketRegime(
        label=label,
        score=round(score, 3),
        mode=mode,
        trade_allowed=trade_allowed,
        min_probability_boost=boost,
        min_relative_strength=min_rs,
        min_ml_score=min_ml,
        allowed_tiers=allowed_tiers,
        max_picks=max_picks,
        reasons=reasons[:4],
    )
