"""Rank screener candidates and produce buy recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field

from chartink_client import StockCandidate


@dataclass
class AggregatedCandidate:
    symbol: str
    name: str
    close: float
    per_chg: float
    volume: float
    primary_setup: str
    setup_weight: float
    setups: list[str] = field(default_factory=list)
    setup_count: int = 1


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
    partial_exit: float
    breakeven_stop: float
    setup_count: int
    confluence: str
    rationale: str
    exit_plan: str


def aggregate_candidates(candidates: list[StockCandidate]) -> list[AggregatedCandidate]:
    """Merge duplicate symbols; track all matching setups (confluence)."""
    grouped: dict[str, list[StockCandidate]] = {}
    for item in candidates:
        grouped.setdefault(item.symbol, []).append(item)

    aggregated: list[AggregatedCandidate] = []
    for symbol, items in grouped.items():
        best = max(items, key=lambda x: x.setup_weight)
        setups = sorted({i.setup for i in items})
        aggregated.append(
            AggregatedCandidate(
                symbol=symbol,
                name=best.name,
                close=best.close,
                per_chg=best.per_chg,
                volume=best.volume,
                primary_setup=best.setup,
                setup_weight=best.setup_weight,
                setups=setups,
                setup_count=len(setups),
            )
        )
    return aggregated


def passes_filters(candidate: AggregatedCandidate, config: dict) -> bool:
    filters = config.get("filters", {})
    price_min = float(filters.get("price_min", 0))
    price_max = float(filters.get("price_max", float("inf")))
    min_turnover = float(filters.get("min_turnover_inr", 0))
    max_daily_change = float(filters.get("max_daily_change_pct", 100))

    if candidate.close < price_min or candidate.close > price_max:
        return False

    turnover = candidate.close * candidate.volume
    if turnover < min_turnover:
        return False

    if filters.get("reject_overextended", True) and abs(candidate.per_chg) > max_daily_change:
        return False

    return True


def score_candidate(candidate: AggregatedCandidate, weights: dict) -> float:
    turnover = candidate.close * candidate.volume

    volume_score = min(turnover / 50_000_000, 1.0)

    abs_chg = abs(candidate.per_chg)
    if 1.5 <= abs_chg <= 6.0:
        momentum_score = 1.0
    elif abs_chg < 1.5:
        momentum_score = 0.5
    elif abs_chg <= 8.0:
        momentum_score = 0.35
    else:
        momentum_score = 0.05

    liquidity_score = min(turnover / 100_000_000, 1.0)
    setup_score = candidate.setup_weight

    # Bonus when multiple scans agree on the same stock.
    confluence_score = min(candidate.setup_count / 3.0, 1.0)

    stability_score = 1.0 if abs_chg <= 8.0 else 0.1

    total = (
        volume_score * weights.get("volume_ratio", 0.25)
        + momentum_score * weights.get("momentum_sweet_spot", 0.25)
        + liquidity_score * weights.get("liquidity", 0.20)
        + setup_score * weights.get("setup_weight", 0.10)
        + confluence_score * weights.get("confluence", 0.10)
        + stability_score * weights.get("price_stability", 0.10)
    )
    return round(total, 4)


def resolve_risk(candidate: AggregatedCandidate, config: dict) -> dict[str, float]:
    risk_cfg = config.get("risk", {})
    by_setup = risk_cfg.get("by_setup", {})
    setup_risk = by_setup.get(candidate.primary_setup, {})

    return {
        "stop_loss_pct": float(setup_risk.get("stop_loss_pct", risk_cfg.get("stop_loss_pct", 3.0))),
        "target_min_pct": float(setup_risk.get("target_min_pct", risk_cfg.get("target_min_pct", 5.0))),
        "target_max_pct": float(setup_risk.get("target_max_pct", risk_cfg.get("target_max_pct", 10.0))),
        "partial_exit_pct": float(setup_risk.get("partial_exit_pct", risk_cfg.get("partial_exit_pct", 50.0))),
        "breakeven_after_pct": float(
            setup_risk.get("breakeven_after_pct", risk_cfg.get("breakeven_after_pct", 4.0))
        ),
    }


def build_picks(
    candidates: list[StockCandidate],
    config: dict,
    blocked_symbols: set[str] | None = None,
) -> list[TradePick]:
    picks_cfg = config.get("picks", {})
    ranking_weights = config.get("ranking", {})
    blocked = blocked_symbols or set()

    max_picks = int(picks_cfg.get("max_daily_picks", 3))
    max_candidates = int(picks_cfg.get("max_candidates", 40))

    aggregated = aggregate_candidates(candidates)
    eligible = [
        c for c in aggregated if passes_filters(c, config) and c.symbol not in blocked
    ]

    scored: list[tuple[AggregatedCandidate, float]] = [
        (c, score_candidate(c, ranking_weights)) for c in eligible
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:max_candidates][:max_picks]

    results: list[TradePick] = []
    for candidate, score in top:
        risk = resolve_risk(candidate, config)
        entry = candidate.close
        stop_pct = risk["stop_loss_pct"]
        target_min = risk["target_min_pct"]
        target_max = risk["target_max_pct"]
        partial_pct = risk["partial_exit_pct"]
        breakeven_trigger_pct = risk["breakeven_after_pct"]

        stop = round(entry * (1 - stop_pct / 100), 2)
        target_low = round(entry * (1 + target_min / 100), 2)
        target_high = round(entry * (1 + target_max / 100), 2)
        partial_exit = round(entry * (1 + target_min / 100), 2)
        breakeven_stop = round(entry * (1 + breakeven_trigger_pct / 100), 2)

        confluence = (
            f"{candidate.setup_count} setup(s): {', '.join(candidate.setups)}"
            if candidate.setup_count > 1
            else candidate.primary_setup
        )
        rationale = _build_rationale(candidate)
        exit_plan = (
            f"Book {partial_pct:.0f}% at ₹{partial_exit:,.2f} (+{target_min:.0f}%). "
            f"Move stop to entry once price crosses ₹{breakeven_stop:,.2f}. "
            f"Trail remainder toward ₹{target_high:,.2f}."
        )

        results.append(
            TradePick(
                symbol=candidate.symbol,
                name=candidate.name,
                setup=candidate.primary_setup,
                price=entry,
                change_pct=candidate.per_chg,
                volume=candidate.volume,
                score=score,
                entry=entry,
                stop_loss=stop,
                target_low=target_low,
                target_high=target_high,
                partial_exit=partial_exit,
                breakeven_stop=breakeven_stop,
                setup_count=candidate.setup_count,
                confluence=confluence,
                rationale=rationale,
                exit_plan=exit_plan,
            )
        )
    return results


def _build_rationale(candidate: AggregatedCandidate) -> str:
    base = {
        "Breakout Momentum": "Weekly high breakout with volume and 50/200 SMA trend support.",
        "EMA Pullback": "Pullback to 20 EMA in uptrend; bounce with volume confirmation.",
        "Range Breakout": "5-day range breakout with rising volume — squeeze expansion play.",
    }.get(candidate.primary_setup, "Matches configured swing setup criteria.")

    if candidate.setup_count > 1:
        return f"{base} Confluence: flagged by {candidate.setup_count} independent scans."
    return base
