"""Rank screener candidates and produce buy recommendations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from advanced_analyzer import AdvancedSignals, analyze_batch, passes_advanced_filters
from chartink_client import StockCandidate
from diversification import diversify_picks
from market_data import fetch_history
from regime_filter import assess_regime

logger = logging.getLogger(__name__)


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
    probability: float
    grade: str
    confidence_tier: str
    ensemble_score: float
    monte_carlo_win_rate: float
    mtf_alignment: float
    walk_forward_edge: float
    fake_breakout_risk: float
    fake_move_risk: float
    support_level: float
    resistance_level: float
    warnings: list[str]
    confirmations: list[str]
    advanced_summary: str


def aggregate_candidates(candidates: list[StockCandidate]) -> list[AggregatedCandidate]:
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


def resolve_risk(
    candidate: AggregatedCandidate,
    config: dict,
    signals: AdvancedSignals | None = None,
) -> dict[str, float]:
    risk_cfg = config.get("risk", {})
    by_setup = risk_cfg.get("by_setup", {})
    setup_risk = by_setup.get(candidate.primary_setup, {})

    stop_pct = float(setup_risk.get("stop_loss_pct", risk_cfg.get("stop_loss_pct", 3.0)))
    target_min = float(setup_risk.get("target_min_pct", risk_cfg.get("target_min_pct", 5.0)))
    target_max = float(setup_risk.get("target_max_pct", risk_cfg.get("target_max_pct", 10.0)))

    if risk_cfg.get("use_atr_stops", True) and signals and signals.atr_pct > 0:
        atr_mult = float(risk_cfg.get("atr_stop_multiplier", 1.5))
        atr_stop = signals.atr_pct * atr_mult
        min_stop = float(risk_cfg.get("min_stop_pct", 2.0))
        max_stop = float(risk_cfg.get("max_stop_pct", 6.0))
        stop_pct = float(min(max(atr_stop, min_stop), max_stop))

    return {
        "stop_loss_pct": stop_pct,
        "target_min_pct": target_min,
        "target_max_pct": target_max,
        "partial_exit_pct": float(setup_risk.get("partial_exit_pct", risk_cfg.get("partial_exit_pct", 50.0))),
        "breakeven_after_pct": float(
            setup_risk.get("breakeven_after_pct", risk_cfg.get("breakeven_after_pct", 4.0))
        ),
    }


def _combined_score(basic: float, signals: AdvancedSignals, config: dict) -> float:
    adv = config.get("advanced", {})
    w_basic = float(adv.get("weights", {}).get("basic_score", 0.25))
    w_ens = float(adv.get("weights", {}).get("ensemble", 0.75))
    penalty = (signals.fake_breakout_risk * 0.12) + (signals.fake_move_risk * 0.08)
    tier_bonus = {"ELITE": 0.08, "STRONG": 0.04, "PASS": 0.0, "REJECT": -0.2}.get(
        signals.confidence_tier, 0.0
    )
    return round(
        basic * w_basic + signals.ensemble_score * w_ens + tier_bonus - penalty,
        4,
    )


def _run_advanced_analysis(
    candidates: list[AggregatedCandidate],
    config: dict,
) -> tuple[dict[str, AdvancedSignals], dict[str, "pd.DataFrame"], object]:
    import pandas as pd  # noqa: F401 — type context for history dict

    adv = config.get("advanced", {})
    if not adv.get("enabled", True):
        return {}, {}, None

    symbols = [c.symbol for c in candidates]
    days = int(adv.get("history_days", 90))
    logger.info("Running ensemble analysis on %s symbols...", len(symbols))

    history, benchmark = fetch_history(symbols, days=days)
    setups = {c.symbol: c.primary_setup for c in candidates}
    signals = analyze_batch(symbols, history, benchmark, setups, config)

    regime = assess_regime(benchmark, config)
    config["_market_regime"] = regime
    config["_regime_probability_boost"] = regime.min_probability_boost

    loaded = sum(1 for s in symbols if s in history)
    logger.info(
        "Loaded OHLCV for %s/%s | Market regime: %s",
        loaded,
        len(symbols),
        regime.summary,
    )
    return signals, history, regime


def build_picks(
    candidates: list[StockCandidate],
    config: dict,
    blocked_symbols: set[str] | None = None,
) -> list[TradePick]:
    picks_cfg = config.get("picks", {})
    ranking_weights = config.get("ranking", {})
    adv_cfg = config.get("advanced", {})
    div_cfg = config.get("diversification", {})
    blocked = blocked_symbols or set()

    max_picks = int(picks_cfg.get("max_daily_picks", 3))
    deep_top = int(adv_cfg.get("deep_analyze_top", 20))

    aggregated = aggregate_candidates(candidates)
    eligible = [c for c in aggregated if passes_filters(c, config) and c.symbol not in blocked]

    basic_scored: list[tuple[AggregatedCandidate, float]] = [
        (c, score_candidate(c, ranking_weights)) for c in eligible
    ]
    basic_scored.sort(key=lambda x: x[1], reverse=True)

    to_analyze = [c for c, _ in basic_scored[:deep_top]]
    advanced_signals, history, regime = _run_advanced_analysis(to_analyze, config)

    if regime and not regime.trade_allowed:
        logger.warning("Market regime BEARISH — no picks today (%s)", regime.summary)
        return []

    use_advanced = adv_cfg.get("enabled", True) and bool(advanced_signals)
    if adv_cfg.get("enabled", True) and not advanced_signals:
        logger.warning("Advanced analysis unavailable — falling back to basic scoring")

    final_scored: list[tuple[AggregatedCandidate, float, AdvancedSignals | None]] = []
    for candidate, basic in basic_scored:
        signals = advanced_signals.get(candidate.symbol) if use_advanced else None
        if use_advanced:
            if signals is None:
                continue
            min_prob = float(adv_cfg.get("min_probability", 58))
            boost = float(config.get("_regime_probability_boost", 0))
            if signals.probability < min_prob + boost:
                logger.info("Rejected %s: P=%.0f%% < %.0f%% (regime-adjusted)", candidate.symbol, signals.probability, min_prob + boost)
                continue
            if not passes_advanced_filters(signals, config):
                logger.info(
                    "Rejected %s: tier=%s vetoes=%s",
                    candidate.symbol,
                    signals.confidence_tier,
                    signals.veto_reasons,
                )
                continue
            combined = _combined_score(basic, signals, config)
        else:
            combined = basic
        final_scored.append((candidate, combined, signals))

    final_scored.sort(key=lambda x: x[1], reverse=True)

    if use_advanced and div_cfg.get("enabled", True):
        diversified = diversify_picks(
            final_scored,
            history,
            max_per_sector=int(div_cfg.get("max_per_sector", 1)),
            max_correlation=float(div_cfg.get("max_correlation", 0.75)),
        )
    else:
        diversified = final_scored

    top = diversified[:max_picks]

    results: list[TradePick] = []
    for candidate, score, signals in top:
        risk = resolve_risk(candidate, config, signals)
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
        rationale = _build_rationale(candidate, signals)
        exit_plan = (
            f"Book {partial_pct:.0f}% at ₹{partial_exit:,.2f} (+{target_min:.0f}%). "
            f"Move stop to entry at ₹{breakeven_stop:,.2f}. "
            f"Trail toward ₹{target_high:,.2f}. Support ₹{signals.support_level:,.0f}."
            if signals
            else f"Book {partial_pct:.0f}% at T1; trail to T2."
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
                probability=signals.probability if signals else 0.0,
                grade=signals.grade if signals else "?",
                confidence_tier=signals.confidence_tier if signals else "?",
                ensemble_score=signals.ensemble_score if signals else 0.0,
                monte_carlo_win_rate=signals.monte_carlo_win_rate if signals else 0.0,
                mtf_alignment=signals.mtf_alignment if signals else 0.0,
                walk_forward_edge=signals.walk_forward_edge if signals else 0.0,
                fake_breakout_risk=signals.fake_breakout_risk if signals else 0.0,
                fake_move_risk=signals.fake_move_risk if signals else 0.0,
                support_level=signals.support_level if signals else 0.0,
                resistance_level=signals.resistance_level if signals else 0.0,
                warnings=signals.warnings if signals else [],
                confirmations=signals.confirmations if signals else [],
                advanced_summary=signals.summary if signals else "Basic mode",
            )
        )
    return results


def _build_rationale(candidate: AggregatedCandidate, signals: AdvancedSignals | None) -> str:
    base = {
        "Breakout Momentum": "Weekly high breakout with volume and 50/200 SMA trend support.",
        "EMA Pullback": "Pullback to 20 EMA in uptrend; bounce with volume confirmation.",
        "Range Breakout": "5-day range breakout with rising volume — squeeze expansion play.",
    }.get(candidate.primary_setup, "Matches configured swing setup criteria.")
    parts = [base]
    if candidate.setup_count > 1:
        parts.append(f"Confluence: {candidate.setup_count} scans agree.")
    if signals:
        parts.append(
            f"Ensemble {signals.ensemble_score:.0%} | MC {signals.monte_carlo_win_rate:.0%} "
            f"| hist edge {signals.walk_forward_edge:.0%} | MTF {signals.mtf_alignment:.0%}."
        )
        if signals.confirmations:
            parts.append(signals.confirmations[0])
    return " ".join(parts)
