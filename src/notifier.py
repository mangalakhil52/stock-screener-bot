"""Send alerts via Telegram, SMS, or WhatsApp."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from ranker import TradePick

logger = logging.getLogger(__name__)


def format_message(picks: list[TradePick], total_scanned: int) -> str:
    now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y, %I:%M %p IST")

    if not picks:
        return (
            f"📊 *Daily Swing Screener* — {now}\n\n"
            f"No A+ setups today from {total_scanned} candidates.\n"
            "Stay in cash — discipline beats forcing trades."
        )

    lines = [
        f"📊 *Daily Swing Picks* — {now}",
        f"Pool: {total_scanned} stocks → Top {len(picks)} recommendations\n",
        "⚠️ Not financial advice. Always verify on chart before buying.\n",
    ]

    for idx, pick in enumerate(picks, start=1):
        lines.extend(
            [
                f"*#{idx} {pick.symbol}* ({pick.setup})",
                f"   Price: ₹{pick.price:,.2f} ({pick.change_pct:+.2f}%)",
                f"   Entry: ₹{pick.entry:,.2f}",
                f"   Stop: ₹{pick.stop_loss:,.2f} (-3%)",
                f"   Target: ₹{pick.target_low:,.2f} – ₹{pick.target_high:,.2f} (+5–10%)",
                f"   Score: {pick.score:.2f} | Vol: {pick.volume:,.0f}",
                f"   _{pick.rationale}_",
                "",
            ]
        )

    lines.append("📌 Hold horizon: 5–10 trading days. Exit if stop hits.")
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
    # Plain text for SMS/WhatsApp (no markdown)
    plain_message = format_message(picks, total_scanned).replace("*", "").replace("_", "")

    sent_any = False
    if notify_cfg.get("telegram", True):
        if send_telegram(format_message(picks, total_scanned)):
            sent_any = True
            logger.info("Telegram alert sent")

    if notify_cfg.get("sms") or notify_cfg.get("whatsapp"):
        if send_twilio_sms(plain_message):
            sent_any = True
            logger.info("Twilio alert sent")

    if not sent_any:
        raise RuntimeError("No notification channel succeeded. Check .env and config.")
