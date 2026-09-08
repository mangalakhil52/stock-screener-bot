"""Ensemble scoring engine: MTF, walk-forward, Wyckoff, Monte Carlo, ML features."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    to_weekly,
    upper_wick_ratio,
)
from monte_carlo import simulate_win_probability

try:
    from ml_scorer import predict_probability as ml_predict
except ImportError:
    ml_predict = None


@dataclass
class FeatureVector:
    values: dict[str, float]

    def as_array(self, keys: list[str]) -> np.ndarray:
        return np.array([self.values.get(k, 0.0) for k in keys], dtype=float)


# Hand-tuned ensemble weights (mimics trained model without sklearn dependency).
ENSEMBLE_FEATURES = [
    "rule_probability",
    "mtf_alignment",
    "walk_forward_edge",
    "mc_win_rate",
    "wyckoff_score",
    "trend_strength",
    "volume_quality",
    "relative_strength",
    "adx_strength",
    "macd_bullish",
    "support_distance",
]
ENSEMBLE_WEIGHTS = np.array(
    [0.18, 0.12, 0.14, 0.14, 0.08, 0.10, 0.08, 0.08, 0.04, 0.02, 0.02]
)


@dataclass
class EnsembleResult:
    rule_probability: float
    ensemble_score: float           # 0–1 unified score
    probability: float              # final 0–100%
    mtf_alignment: float
    walk_forward_edge: float
    monte_carlo_win_rate: float
    wyckoff_score: float
    atr_pct: float
    support_level: float
    resistance_level: float
    confidence_tier: str            # ELITE / STRONG / PASS / REJECT
    fake_breakout_risk: float
    fake_move_risk: float
    trend_strength: float
    volume_quality: float
    relative_strength: float
    ml_score: float                 # daily-trained market model P(win)
    grade: str
    warnings: list[str] = field(default_factory=list)
    confirmations: list[str] = field(default_factory=list)
    veto_reasons: list[str] = field(default_factory=list)

    @property
    def is_high_risk(self) -> bool:
        return self.fake_breakout_risk >= 0.55 or self.fake_move_risk >= 0.60

    @property
    def summary(self) -> str:
        return (
            f"{self.confidence_tier} P={self.probability:.0f}% "
            f"ens={self.ensemble_score:.0%} MC={self.monte_carlo_win_rate:.0%}"
        )


def _walk_forward_edge(
    df: pd.DataFrame,
    stop_pct: float = 3.0,
    target_pct: float = 5.0,
    horizon: int = 10,
    lookback: int = 60,
) -> float:
    """Historical win rate of similar breakout setups on this stock."""
    close = df["close"]
    high = df["high"]
    volume = df["volume"]
    if len(close) < lookback + horizon + 5:
        return 0.5

    wins = 0
    total = 0
    vol_avg = volume.rolling(20).mean()

    start_idx = max(20, len(close) - lookback - horizon)
    for i in range(start_idx, len(close) - horizon):
        if pd.isna(vol_avg.iloc[i]) or vol_avg.iloc[i] == 0:
            continue
        high_20 = float(high.iloc[i - 20 : i].max())
        if close.iloc[i] <= high_20 * 1.001:
            continue
        if volume.iloc[i] < vol_avg.iloc[i] * 1.2:
            continue

        entry = float(close.iloc[i])
        stop = entry * (1 - stop_pct / 100)
        target = entry * (1 + target_pct / 100)
        total += 1

        for j in range(1, horizon + 1):
            low_j = float(df["low"].iloc[i + j])
            high_j = float(high.iloc[i + j])
            if low_j <= stop:
                break
            if high_j >= target:
                wins += 1
                break

    return wins / total if total > 0 else 0.5


def _wyckoff_accumulation_score(df: pd.DataFrame) -> float:
    """Detect accumulation: higher lows, rising OBV, compressing range."""
    if len(df) < 30:
        return 0.5
    close = df["close"]
    low = df["low"]
    volume = df["volume"]

    low_10 = float(low.tail(10).min())
    low_prev_10 = float(low.iloc[-20:-10].min())
    higher_lows = 1.0 if low_10 > low_prev_10 else 0.3

    bb_width = bollinger_bandwidth(close)
    squeeze = 1.0 if bb_width < 0.08 else 0.5 if bb_width < 0.12 else 0.2

    obv = obv_slope(close, volume, 15)
    obv_pos = min(max(obv * 5 + 0.5, 0), 1)

    return float(np.mean([higher_lows, squeeze, obv_pos]))


def _mtf_alignment(df: pd.DataFrame) -> float:
    weekly = to_weekly(df)
    if weekly is None or len(weekly) < 10:
        return 0.5
    daily_close = float(df["close"].iloc[-1])
    daily_sma20 = float(sma(df["close"], 20).iloc[-1])
    weekly_close = float(weekly["close"].iloc[-1])
    weekly_sma = float(sma(weekly["close"], 10).iloc[-1])

    score = 0.0
    if daily_close > daily_sma20:
        score += 0.5
    if weekly_close > weekly_sma:
        score += 0.5
    return score


def _support_resistance(df: pd.DataFrame) -> tuple[float, float]:
    if len(df) < 20:
        c = float(df["close"].iloc[-1])
        return c * 0.97, c * 1.03
    support = float(df["low"].tail(20).min())
    resistance = float(df["high"].tail(20).max())
    return support, resistance


def _detect_traps(df: pd.DataFrame, setup: str) -> tuple[float, float, list[str], list[str]]:
    """Fake breakout and distribution detection."""
    warnings: list[str] = []
    confirmations: list[str] = []

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

    vol_20 = float(sma(volume, 20).iloc[-1])
    vol_3 = float(volume.tail(3).mean())
    vol_10 = float(volume.tail(10).mean())
    rsi_val = rsi(close)

    high_20_prior = float(high.iloc[-21:-1].max()) if len(high) > 21 else float(high.iloc[:-1].max())
    high_5_prior = float(high.iloc[-6:-1].max()) if len(high) > 6 else high_20_prior

    c_str = close_strength(last_close, last_high, last_low)
    wick = upper_wick_ratio(last_open, last_high, last_close, last_low)
    vol_ratio = last_vol / vol_20 if vol_20 > 0 else 1.0
    broke_resistance = last_high > high_20_prior * 1.001

    fake_bo_flags: list[float] = []
    is_breakout = setup in {"Breakout Momentum", "Range Breakout"}

    if is_breakout and broke_resistance:
        if c_str < 0.42:
            fake_bo_flags.append(0.35)
            warnings.append("Fake breakout: weak close")
        if wick > 0.45:
            fake_bo_flags.append(0.30)
            warnings.append("Long upper wick — rejection at highs")
        if vol_3 < vol_10 * 0.85 and vol_ratio < 1.2:
            fake_bo_flags.append(0.25)
            warnings.append("Volume fading on breakout")
        if rsi_val > 72:
            fake_bo_flags.append(0.20)
            warnings.append(f"RSI {rsi_val:.0f} overbought")
        if last_close < high_5_prior:
            fake_bo_flags.append(0.30)
            warnings.append("Failed to hold above 5-day high")

    fake_mv_flags: list[float] = []
    if vol_ratio > 1.4 and abs(float(close.pct_change().iloc[-1]) * 100) < 0.8:
        fake_mv_flags.append(0.35)
        warnings.append("High volume, flat price — distribution")
    if len(close) >= 2:
        prev = float(close.iloc[-2])
        gap = (last_open - prev) / prev * 100 if prev else 0
        if gap > 1.5 and c_str < 0.35:
            fake_mv_flags.append(0.30)
            warnings.append("Gap-up fade trap")

    if is_breakout and broke_resistance and c_str > 0.72 and vol_ratio >= 1.5:
        confirmations.append("Breakout accepted on volume")

    return min(sum(fake_bo_flags), 1.0), min(sum(fake_mv_flags), 1.0), warnings, confirmations


def _rule_probability(
    df: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    setup: str,
    fake_bo: float,
    fake_mv: float,
    confirmations: list[str],
    warnings: list[str],
) -> tuple[float, float, float, float]:
    close = df["close"]
    volume = df["volume"]
    last_close = float(close.iloc[-1])
    vol_20 = float(sma(volume, 20).iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / vol_20 if vol_20 else 1.0
    rsi_val = rsi(close)
    sma20 = float(sma(close, 20).iloc[-1])
    sma50 = float(sma(close, 50).iloc[-1])
    c_str = close_strength(
        last_close, float(df["high"].iloc[-1]), float(df["low"].iloc[-1])
    )

    positive = 0.0
    if c_str > 0.72:
        positive += 0.18
        confirmations.append("Strong close near high")
    if vol_ratio >= 1.5:
        positive += 0.15
    if 55 <= rsi_val <= 68:
        positive += 0.12
    if last_close > sma20 > sma50:
        positive += 0.15
        confirmations.append("Bullish SMA stack")

    rs_score = 0.5
    if benchmark is not None and len(benchmark) >= 10:
        s5 = returns_pct(close, 5)
        b5 = returns_pct(benchmark["close"], 5)
        s10 = returns_pct(close, 10)
        b10 = returns_pct(benchmark["close"], 10)
        if s5 > b5 and s10 > b10:
            rs_score = 0.9
            confirmations.append("Outperforming Nifty 5d & 10d")
        elif s5 > b5:
            rs_score = 0.7
        elif s5 < b5 - 2:
            rs_score = 0.25
            warnings.append("Lagging Nifty")

    trend = float(np.mean([
        1.0 if last_close > sma50 else 0.0,
        1.0 if sma20 > sma50 else 0.3,
        min(max((last_close - sma20) / sma20 * 10, 0), 1.0) if sma20 else 0.5,
    ]))
    vol_q = float(np.mean([
        min(vol_ratio / 2.5, 1.0),
        c_str,
    ]))

    prob = 50.0 + positive * 100 + trend * 12 + vol_q * 10 + rs_score * 8
    prob -= fake_bo * 35 + fake_mv * 25
    if setup == "EMA Pullback" and last_close > sma20 and rsi_val < 65:
        prob += 5

    return float(np.clip(prob, 0, 99)), trend, vol_q, rs_score


def _ml_ensemble_score(features: FeatureVector) -> float:
    arr = features.as_array(ENSEMBLE_FEATURES)
    arr = np.clip(arr, 0, 1)
    score = float(np.dot(arr, ENSEMBLE_WEIGHTS))
    return round(min(max(score, 0), 1), 4)


def _confidence_tier(
    ensemble: float,
    mc: float,
    probability: float,
    fake_bo: float,
    fake_mv: float,
    warnings: list[str],
    config: dict,
) -> str:
    ens_cfg = config.get("ensemble", {})
    elite_ens = float(ens_cfg.get("elite_ensemble_score", 0.78))
    strong_ens = float(ens_cfg.get("strong_ensemble_score", 0.68))
    elite_mc = float(ens_cfg.get("elite_monte_carlo", 0.58))
    strong_mc = float(ens_cfg.get("strong_monte_carlo", 0.52))

    if fake_bo >= 0.55 or fake_mv >= 0.60:
        return "REJECT"
    if ensemble >= elite_ens and mc >= elite_mc and probability >= 70 and len(warnings) == 0:
        return "ELITE"
    if ensemble >= strong_ens and mc >= strong_mc and probability >= 62:
        return "STRONG"
    if probability >= float(config.get("advanced", {}).get("min_probability", 58)):
        return "PASS"
    return "REJECT"


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


def analyze_ensemble(
    symbol: str,
    df: pd.DataFrame,
    benchmark: pd.DataFrame | None,
    setup: str,
    config: dict,
) -> EnsembleResult:
    veto: list[str] = []
    if df is None or len(df) < 30:
        return EnsembleResult(
            rule_probability=0,
            ensemble_score=0,
            probability=0,
            mtf_alignment=0,
            walk_forward_edge=0,
            monte_carlo_win_rate=0,
            wyckoff_score=0,
            atr_pct=0,
            support_level=0,
            resistance_level=0,
            confidence_tier="REJECT",
            fake_breakout_risk=1,
            fake_move_risk=1,
            trend_strength=0,
            volume_quality=0,
            relative_strength=0,
            ml_score=0,
            grade="D",
            warnings=["Insufficient history"],
            veto_reasons=["Insufficient history"],
        )

    risk_cfg = config.get("risk", {})
    setup_risk = risk_cfg.get("by_setup", {}).get(setup, {})
    stop_pct = float(setup_risk.get("stop_loss_pct", risk_cfg.get("stop_loss_pct", 3.0)))
    target_pct = float(setup_risk.get("target_min_pct", risk_cfg.get("target_min_pct", 5.0)))

    fake_bo, fake_mv, warnings, confirmations = _detect_traps(df, setup)
    rule_prob, trend, vol_q, rs = _rule_probability(
        df, benchmark, setup, fake_bo, fake_mv, confirmations, warnings
    )

    mtf = _mtf_alignment(df)
    wf_edge = _walk_forward_edge(df, stop_pct, target_pct)
    wyckoff = _wyckoff_accumulation_score(df)
    support, resistance = _support_resistance(df)

    last_close = float(df["close"].iloc[-1])
    atr_val = atr(df["high"], df["low"], df["close"])
    atr_pct = (atr_val / last_close * 100) if last_close else stop_pct

    mc_cfg = config.get("monte_carlo", {})
    mc_win, _ = simulate_win_probability(
        df["close"],
        last_close,
        stop_pct,
        target_pct,
        horizon_days=int(mc_cfg.get("horizon_days", 10)),
        simulations=int(mc_cfg.get("simulations", 500)),
    )

    adx = adx_proxy(df["high"], df["low"], df["close"])
    macd_bull = 1.0 if macd_histogram(df["close"]) > 0 else 0.0
    sup_dist = min(max((last_close - support) / last_close * 20, 0), 1) if last_close else 0.5

    features = FeatureVector(
        values={
            "rule_probability": rule_prob / 100,
            "mtf_alignment": mtf,
            "walk_forward_edge": wf_edge,
            "mc_win_rate": mc_win,
            "wyckoff_score": wyckoff,
            "trend_strength": trend,
            "volume_quality": vol_q,
            "relative_strength": rs,
            "adx_strength": adx,
            "macd_bullish": macd_bull,
            "support_distance": sup_dist,
        }
    )
    ensemble = _ml_ensemble_score(features)

    # Daily-trained market ML model (whole NSE universe).
    ml_score = 0.5
    ml_cfg = config.get("ml", {})
    if ml_cfg.get("enabled", True) and ml_predict is not None:
        from pathlib import Path

        root = Path(config.get("_root", "."))
        ml_pred = ml_predict(df, benchmark, root, config)
        if ml_pred is not None:
            ml_score = ml_pred
            if ml_score >= 0.65:
                confirmations.append(f"ML market model: {ml_score:.0%} win probability")
            elif ml_score < 0.45:
                warnings.append(f"ML market model: only {ml_score:.0%} win probability")

    # Blend rule + ensemble + ML into final probability.
    ml_weight = float(ml_cfg.get("weight_in_probability", 0.25))
    ens_weight = 1.0 - ml_weight - 0.30
    probability = (
        0.30 * rule_prob
        + ens_weight * (ensemble * 100)
        + ml_weight * (ml_score * 100)
    )
    probability = float(np.clip(probability, 0, 99))

    tier = _confidence_tier(ensemble, mc_win, probability, fake_bo, fake_mv, warnings, config)

    ens_cfg = config.get("ensemble", {})
    if mtf < float(ens_cfg.get("min_mtf_alignment", 0.5)):
        veto.append("Multi-timeframe misalignment")
    if mc_win < float(ens_cfg.get("min_monte_carlo_win_rate", 0.50)):
        veto.append(f"Monte Carlo win rate {mc_win:.0%} too low")
    if wf_edge < float(ens_cfg.get("min_walk_forward_edge", 0.40)):
        veto.append(f"Historical edge {wf_edge:.0%} on this stock")
    if ensemble < float(ens_cfg.get("min_ensemble_score", 0.62)):
        veto.append(f"Ensemble score {ensemble:.0%} below threshold")
    min_ml = float(ml_cfg.get("min_probability", 0.48))
    if ml_cfg.get("enabled", True) and ml_score < min_ml:
        veto.append(f"ML market score {ml_score:.0%} below {min_ml:.0%}")

    allowed_tiers = set(ens_cfg.get("allowed_tiers", ["ELITE", "STRONG", "PASS"]))
    if tier not in allowed_tiers:
        veto.append(f"Confidence tier {tier} not allowed")

    return EnsembleResult(
        rule_probability=round(rule_prob, 1),
        ensemble_score=ensemble,
        probability=round(probability, 1),
        mtf_alignment=round(mtf, 3),
        walk_forward_edge=round(wf_edge, 3),
        monte_carlo_win_rate=mc_win,
        wyckoff_score=round(wyckoff, 3),
        atr_pct=round(atr_pct, 2),
        support_level=round(support, 2),
        resistance_level=round(resistance, 2),
        confidence_tier=tier,
        fake_breakout_risk=round(fake_bo, 3),
        fake_move_risk=round(fake_mv, 3),
        trend_strength=round(trend, 3),
        volume_quality=round(vol_q, 3),
        relative_strength=round(rs, 3),
        ml_score=round(ml_score, 4),
        grade=_grade(probability, fake_bo, fake_mv),
        warnings=warnings[:5],
        confirmations=confirmations[:5],
        veto_reasons=veto,
    )


def passes_ensemble_filters(result: EnsembleResult, config: dict) -> bool:
    adv = config.get("advanced", {})
    if not adv.get("enabled", True):
        return True

    if result.veto_reasons:
        return False
    if result.fake_breakout_risk > float(adv.get("max_fake_breakout_risk", 0.55)):
        return False
    if result.fake_move_risk > float(adv.get("max_fake_move_risk", 0.60)):
        return False
    if result.grade in set(adv.get("reject_grades", ["D"])):
        return False
    return True
