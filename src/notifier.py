"""Send alerts via Telegram, SMS, or WhatsApp."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from ranker import TradePick

logger = logging.getLogger(__name__)


def format_message(
    picks: list[TradePick],
    total_scanned: int,
    regime_summary: str | None = None,
    ml_summary: str | None = None,
) -> str:
    now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %I:%M %p IST")

    if not picks:
        regime_line = f"\nMarket: {regime_summary}\n" if regime_summary else ""
        return (
            f"📊 *Daily Swing Screener* — {now}\n"
            f"{regime_line}\n"
            f"No elite setups today from {total_scanned} candidates.\n"
            "🛡️ System chose *no trade* over a risky trade. Stay in cash."
        )

    lines = [
        f"📊 *Elite Swing Picks* — {now}",
        f"Pool: {total_scanned} stocks → {len(picks)} high-confidence pick(s)\n",
    ]
    if regime_summary:
        lines.append(f"📈 Market regime: {regime_summary}\n")
    if ml_summary:
        lines.append(f"🤖 ML model: {ml_summary}\n")
    lines.append("⚠️ Not financial advice. Verify on chart before buying.\n")

    for idx, pick in enumerate(picks, start=1):
        stop_pct = round((1 - pick.stop_loss / pick.entry) * 100, 1) if pick.entry else 0
        target_min_pct = round((pick.target_low / pick.entry - 1) * 100, 1) if pick.entry else 0
        target_max_pct = round((pick.target_high / pick.entry - 1) * 100, 1) if pick.entry else 0
        pick_lines = [
            f"*#{idx} {pick.symbol}* [{pick.confidence_tier}] grade {pick.grade} ({pick.setup})",
            f"   🎯 Probability: *{pick.probability:.0f}%* | Ensemble: {pick.ensemble_score:.0%}",
            f"   📊 MC win: {pick.monte_carlo_win_rate:.0%} | Hist edge: {pick.walk_forward_edge:.0%} | MTF: {pick.mtf_alignment:.0%}",
            f"   🤖 ML market: {pick.ml_score:.0%} win probability (daily-trained)",
            f"   Price: ₹{pick.price:,.2f} ({pick.change_pct:+.2f}%)",
            f"   Entry: ₹{pick.entry:,.2f} | Support: ₹{pick.support_level:,.0f} | Resist: ₹{pick.resistance_level:,.0f}",
            f"   Stop: ₹{pick.stop_loss:,.2f} (-{stop_pct:.1f}%)",
            f"   Target: ₹{pick.target_low:,.2f} – ₹{pick.target_high:,.2f} (+{target_min_pct:.0f}–{target_max_pct:.0f}%)",
            f"   Trap risk: fake BO {pick.fake_breakout_risk:.0%} | fake move {pick.fake_move_risk:.0%}",
        ]
        if pick.setup_count > 1:
            pick_lines.append(f"   Confluence: {pick.confluence}")
        if pick.confirmations:
            pick_lines.append(f"   ✓ {pick.confirmations[0]}")
        if pick.warnings:
            pick_lines.append(f"   ⚠ {pick.warnings[0]}")
        pick_lines.extend([
            f"   _{pick.rationale}_",
            f"   Exit: {pick.exit_plan}",
            "",
        ])
        lines.extend(pick_lines)

    lines.append("📌 7-layer filter passed. Book 50% at T1; trail rest with ATR stop.")
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        logger.error("Telegram credentials missing in .env")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    response = requests.post(
        url,
        json={
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        },
        timeout=30,
    )
    if not response.ok:
        logger.error("Telegram error: %s", response.text)
        return False
    return True


def send_twilio_sms(message: str) -> bool:
    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.getenv("TWILIO_FROM_NUMBER", "").strip()
    to_number = os.getenv("TWILIO_TO_NUMBER", "").strip()
    use_whatsapp = os.getenv("TWILIO_WHATSAPP", "").lower() in {"1", "true", "yes"}

    if not all([account_sid, auth_token, from_number, to_number]):
        logger.error("Twilio credentials missing in .env")
        return False

    if use_whatsapp:
        from_number = f"whatsapp:{from_number}"
        to_number = f"whatsapp:{to_number}"

    url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
    response = requests.post(
        url,
        auth=(account_sid, auth_token),
        data={"From": from_number, "To": to_number, "Body": message[:1500]},
        timeout=30,
    )
    if not response.ok:
        logger.error("Twilio error: %s", response.text)
        return False
    return True


def notify(picks: list[TradePick], total_scanned: int, config: dict) -> None:
    notify_cfg = config.get("notifications", {})
    regime = config.get("_market_regime")
    regime_summary = regime.summary if regime else None
    ml_meta = config.get("_ml_meta")
    ml_summary = None
    if ml_meta:
        ml_summary = (
            f"{ml_meta.get('samples', 0):,} samples | "
            f"AUC {ml_meta.get('test_auc', 0):.2f} | "
            f"trained {ml_meta.get('trained_at', '')[:10]}"
        )

    msg = format_message(picks, total_scanned, regime_summary, ml_summary)
    plain_message = msg.replace("*", "").replace("_", "")

    sent_any = False
    if notify_cfg.get("telegram", True):
        if send_telegram(msg):
            sent_any = True
            logger.info("Telegram alert sent")

    if notify_cfg.get("sms") or notify_cfg.get("whatsapp"):
        if send_twilio_sms(plain_message):
            sent_any = True
            logger.info("Twilio alert sent")

    if not sent_any:
        raise RuntimeError("No notification channel succeeded. Check .env and config.")
