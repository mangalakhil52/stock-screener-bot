"""Rank screener candidates and produce buy recommendations."""

from __future__ import annotations

from dataclasses import dataclass

from chartink_client import StockCandidate


@dataclass
class TradePick:
    symbol: str
    name: str
    setup: str
    price: float
    change_pct: float
    volume: float
    score: float
    entry: float
    stop_loss: float
    target_low: float
    target_high: float
    rationale: str


def dedupe_candidates(candidates: list[StockCandidate]) -> list[StockCandidate]:
    """Keep the highest-weight setup when a stock appears in multiple scans."""
    best: dict[str, StockCandidate] = {}
    for item in candidates:
        existing = best.get(item.symbol)
        if existing is None or item.setup_weight > existing.setup_weight:
            best[item.symbol] = item
    return list(best.values())


def score_candidate(candidate: StockCandidate, weights: dict) -> float:
    turnover = candidate.close * candidate.volume

    # Proxy volume ratio: scans already filter >1.5x; reward higher turnover.
    volume_score = min(turnover / 50_000_000, 1.0)

    # Sweet spot: strong but not overextended on breakout day.
    abs_chg = abs(candidate.per_chg)
    if 1.5 <= abs_chg <= 6.0:
        momentum_score = 1.0
    elif abs_chg < 1.5:
        momentum_score = 0.5
    elif abs_chg <= 8.0:
        momentum_score = 0.4
    else:
        momentum_score = 0.1

    liquidity_score = min(turnover / 100_000_000, 1.0)
    setup_score = candidate.setup_weight
    stability_score = 1.0 if abs_chg <= 8.0 else 0.2

    total = (
        volume_score * weights.get("volume_ratio", 0.30)
        + momentum_score * weights.get("momentum_sweet_spot", 0.25)
        + liquidity_score * weights.get("liquidity", 0.20)
        + setup_score * weights.get("setup_weight", 0.15)
        + stability_score * weights.get("price_stability", 0.10)
    )
    return round(total, 4)


def build_picks(
    candidates: list[StockCandidate],
    config: dict,
) -> list[TradePick]:
    picks_cfg = config.get("picks", {})
    risk_cfg = config.get("risk", {})
    ranking_weights = config.get("ranking", {})

    max_picks = int(picks_cfg.get("max_daily_picks", 3))
    max_candidates = int(picks_cfg.get("max_candidates", 40))
    stop_pct = float(risk_cfg.get("stop_loss_pct", 3.0))
    target_min = float(risk_cfg.get("target_min_pct", 5.0))
    target_max = float(risk_cfg.get("target_max_pct", 10.0))

    unique = dedupe_candidates(candidates)
    scored: list[tuple[StockCandidate, float]] = [
        (c, score_candidate(c, ranking_weights)) for c in unique
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:max_candidates][:max_picks]

    results: list[TradePick] = []
    for candidate, score in top:
        entry = candidate.close
        stop = round(entry * (1 - stop_pct / 100), 2)
        target_low = round(entry * (1 + target_min / 100), 2)
        target_high = round(entry * (1 + target_max / 100), 2)

        rationale = _build_rationale(candidate)
        results.append(
            TradePick(
                symbol=candidate.symbol,
                name=candidate.name,
                setup=candidate.setup,
                price=entry,
                change_pct=candidate.per_chg,
                volume=candidate.volume,
                score=score,
                entry=entry,
                stop_loss=stop,
                target_low=target_low,
                target_high=target_high,
                rationale=rationale,
            )
        )
    return results


def _build_rationale(candidate: StockCandidate) -> str:
    if candidate.setup == "Breakout Momentum":
        return "Weekly high breakout with volume and 50/200 SMA trend support."
    if candidate.setup == "EMA Pullback":
        return "Pullback to 20 EMA in uptrend; bounce with volume confirmation."
    if candidate.setup == "Range Breakout":
        return "5-day range breakout with rising volume — squeeze expansion play."
    return "Matches configured swing setup criteria."
