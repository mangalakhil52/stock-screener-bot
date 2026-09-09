# Daily Swing Stock Screener Bot

Automated NSE swing screener using **Chartink** scans, with daily **Telegram** alerts for top buy candidates (5–10% target in 5–10 days).

## Why Chartink (not Screener.in)?

| Tool | Best for | Automation |
|------|----------|------------|
| **Chartink** | Price action, breakouts, volume, EMA/SMA | Easy via `/screener/process` API |
| **Screener.in** | Fundamentals (P/E, ROCE, debt) | Harder; no official API |

**Recommendation:** Chartink for daily technical scans + optional manual Screener.in check on final picks.

---

## Quick Start

### 1. Install dependencies

```powershell
cd "C:\Users\Akhil M bin pandey\Documents\Cursor\stock-screener-bot"
python -m pip install -r requirements.txt
```

> **Note:** If `pip install` fails with "Unable to create process", use `python -m pip` instead — your `pip` shortcut may point to an old Python install.

### 2. Set up Dhan API (required for market data)

The bot uses **Dhan** for OHLCV history and the full NSE equity universe (no hardcoded symbol list).

1. Log in at [web.dhan.co](https://web.dhan.co) → **Profile** → **DhanHQ Trading APIs**
2. Note your **Client ID**
3. Click **Generate Access Token** (or use a static token if you enabled it)
4. Copy `.env.example` to `.env` in the project root and add:

```
DHAN_CLIENT_ID=your_client_id
DHAN_ACCESS_TOKEN=your_access_token
```

**GitHub Actions:** add the same two values as repository secrets (`Settings → Secrets and variables → Actions`):

- `DHAN_CLIENT_ID`
- `DHAN_ACCESS_TOKEN`

> Access tokens expire (~24h for generated tokens). For daily GitHub runs, use a **static access token** from the Dhan API page, or refresh the secret when training fails.

### 3. Set up Telegram (recommended — free)

1. Open Telegram → search **@BotFather** → `/newbot` → copy the **bot token**
2. Start a chat with your new bot (send any message)
3. Get your chat ID:
   - Visit `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Find `"chat":{"id":123456789}` in the JSON
4. Add to `.env`:

```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

For GitHub Actions, also set secrets `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`.

### 4. Run manually

```powershell
python src\main.py
```

You should receive a Telegram message with top 2–3 picks.

### 5. Schedule daily (Windows Task Scheduler)

1. Open **Task Scheduler** → Create Basic Task
2. Trigger: Daily at **3:45 PM** (after market close) or **9:00 AM** (before open)
3. Action: Start a program → `run_daily.bat`
4. Start in: `C:\Users\Akhil M bin pandey\Documents\Cursor\stock-screener-bot`

Logs are saved to `logs/daily_run.log`.

---

## What the bot does daily

```
Chartink Scans (3 setups)
        ↓
   20–40 stocks
        ↓
   Rank & score
        ↓
   Top 3 picks
        ↓
 Telegram alert (entry, stop, target)
```

### The 3 setups (in `config.yaml`)

1. **Breakout Momentum** — Weekly high breakout + 1.5× volume + above 50/200 SMA
2. **EMA Pullback** — Bounce off 20 EMA in uptrend
3. **Range Breakout** — 5-day range break with volume expansion

### Ranking logic

Stocks are scored on:
- Volume / liquidity (rupee turnover, min ₹5 Cr/day)
- Momentum sweet spot (1.5–6% daily move preferred)
- Setup type weight
- **Confluence** — bonus when 2+ scans agree on the same stock
- Penalty for overextended moves (>8% in one day — rejected)

### Trade management (new)

- **Setup-specific stops/targets** — wider for breakouts, tighter for EMA pullbacks
- **7-day cooldown** — won't re-recommend a symbol picked in the last week
- **Exit plan in alerts** — book 50% at T1, move stop to entry after +4%

### 7-layer defense system

No system wins 100% — this bot **prefers no trade over a bad trade** using seven filter layers:

1. Chartink scans → 2. Nifty regime gate → 3. Fake breakout traps → 4. Multi-timeframe alignment → 5. Walk-forward historical edge → 6. Monte Carlo (500 paths) → 7. Ensemble scoring + diversification

| Module | Purpose |
|--------|---------|
| `regime_filter.py` | Bearish **selective** mode — index-beating stocks only |
| `ensemble_engine.py` | Wyckoff, S/R, confidence tiers (ELITE/STRONG/PASS) |
| `monte_carlo.py` | P(hit target before stop) via bootstrap simulation |
| `diversification.py` | Max 1 pick/sector, correlation < 0.75 |

Picks must pass: min **62%** probability, ensemble ≥ 62%, MC win ≥ 50%, grade A or B only.

### Price filter (₹100 – ₹10,000)

Every Chartink scan includes `latest close > 100 and latest close < 10000`. This is intentional:

| Bound | Reason |
|-------|--------|
| **> ₹100** | Excludes penny stocks with thin liquidity and manipulation risk |
| **< ₹10,000** | Focuses on tradable mid/large caps; very few NSE names trade above this |

Most of your historical picks (₹2,000–₹9,000) sit in the sweet spot where daily turnover is high enough for clean 5–10% swings. Adjust `filters.price_min` / `price_max` in `config.yaml` (and matching scan clauses) if you want a different band.

---

## Optional: SMS / WhatsApp via Twilio (paid)

Add to `.env`:

```
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+14155238886
TWILIO_TO_NUMBER=+91XXXXXXXXXX
TWILIO_WHATSAPP=true
```

Enable in `config.yaml`:

```yaml
notifications:
  telegram: true
  sms: false
  whatsapp: true
```

---

## Customize scans

Edit `config.yaml` → `scans` section. Test any query visually at [chartink.com/screener](https://chartink.com/screener/) before adding it.

See `scans/chartink_queries.txt` for copy-paste queries.

---

## Sample Telegram alert

```
📊 Daily Swing Picks — 20 Jul 2026, 03:45 PM IST
Pool: 28 stocks → Top 3 recommendations

#1 RELIANCE (Breakout Momentum)
   Price: ₹2,850.00 (+3.20%)
   Entry: ₹2,850.00
   Stop: ₹2,764.50 (-3%)
   Target: ₹2,992.50 – ₹3,135.00 (+5–10%)
   Score: 0.82 | Vol: 12,500,000
   Weekly high breakout with volume and 50/200 SMA trend support.
```

---

## Important disclaimers

- This is a **decision-support tool**, not financial advice
- Always verify picks on a daily chart before buying
- Use stop losses — no system wins every trade
- Chartink free data may be **15–30 min delayed**; premium account improves timeliness
- Paper trade for 2–4 weeks before risking real capital

---

## Project structure

```
stock-screener-bot/
├── config.yaml          # Scans, ranking, notification settings
├── .env                 # Secrets (Telegram token, etc.)
├── requirements.txt
├── run_daily.bat        # Windows scheduler entry point
├── src/
│   ├── main.py          # Orchestrator
│   ├── chartink_client.py
│   ├── ranker.py
│   ├── advanced_analyzer.py  # Ensemble wrapper
│   ├── ensemble_engine.py    # 7-layer scoring engine
│   ├── indicators.py         # Technical indicators
│   ├── regime_filter.py      # Nifty market regime gate
│   ├── monte_carlo.py        # Path simulation
│   ├── diversification.py    # Sector & correlation filter
│   ├── dhan_client.py        # Dhan API + dynamic NSE universe
│   ├── market_data.py        # Dhan OHLCV fetch
│   ├── trade_history.py      # Cooldown / pick logging
│   └── notifier.py
└── scans/
    └── chartink_queries.txt
```
