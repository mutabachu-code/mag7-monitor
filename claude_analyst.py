import anthropic
import json
import re
import time
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradeSignal:
    ticker: str
    action: str
    confidence: str
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    reasoning: str
    sentiment_summary: str
    raw_signal: str


_client = anthropic.Anthropic()

# ── CACHE ─────────────────────────────────────────────────────────────────────
# Claude is only called when signal changes OR cache is older than 5 minutes.
# This prevents repeated API calls on every 60s dashboard refresh.
_signal_cache: dict = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _is_cache_valid(ticker: str, current_raw_signal: str) -> bool:
    if ticker not in _signal_cache:
        return False
    entry = _signal_cache[ticker]
    age = time.time() - entry["timestamp"]
    return age < CACHE_TTL_SECONDS and entry["raw_signal"] == current_raw_signal


def _store_cache(ticker: str, signal: TradeSignal, raw_signal: str):
    _signal_cache[ticker] = {
        "signal":     signal,
        "raw_signal": raw_signal,
        "timestamp":  time.time(),
    }


def analyse(
    ticker: str,
    current_price: float,
    raw_signal: str,
    rsi: float,
    vol_ratio: float,
    macd_bullish: bool,
    trend_status: str,
    delta_val: float,
    sma200: float,
    account_balance: float = 100.0,
    lot_size: float = 0.02,
    implied_volatility: str = "unavailable",
) -> Optional[TradeSignal]:

    # ── CACHE CHECK ───────────────────────────────────────────────────────────
    if _is_cache_valid(ticker, raw_signal):
        print(f"[claude_analyst] Cache hit for {ticker} — skipping API call")
        return _signal_cache[ticker]["signal"]

    # ── API CALL ──────────────────────────────────────────────────────────────
    prompt = f"""You are a disciplined quantitative trading analyst for Magnificent 7
tech stocks traded as CFDs on a small $100 account via Exness MT5.

=== TECHNICAL DATA: {ticker} ===
Current Price      : ${current_price:.2f}
SMA 200 (1H)       : ${sma200:.2f}
Trend vs SMA200    : {trend_status}
RSI (5m, 14-period): {rsi:.1f}
Volume Surge Ratio : {vol_ratio:.2f}x  (>1.2 = elevated volume)
MACD (1H)          : {"Bullish — MACD line above signal line" if macd_bullish else "Bearish — MACD line below signal line"}
Implied Volatility : {implied_volatility}
Options Delta      : {delta_val:.2f}
Dashboard Signal   : {raw_signal}

=== ACCOUNT & RISK PARAMETERS ===
Account Balance    : ${account_balance:.2f}
Lot Size (fixed)   : {lot_size} lots
Stop Loss Rule     : 1.0% from entry (BUY: below entry, SELL: above entry)
Take Profit Rule   : Minimum 2:1 reward-to-risk ratio

=== YOUR TASK ===
STEP 1 — Check technicals are valid for a trade.
STEP 2 — Search for latest news, sentiment, analyst views on {ticker} today.
STEP 3 — Factor in Implied Volatility:
- HIGH IV Rank (>60): options are expensive — prefer selling strategies or tighter SL
- LOW IV Rank (<35): options are cheap — favours directional trades with wider TP targets
- EXTREME IV (>80): avoid new entries unless signal is exceptionally strong
STEP 4 — Combine technicals + news + IV into a final BUY / SELL / HOLD decision.
Downgrade to HOLD if adverse news, earnings risk, extreme IV, or macro contradicts the signal.

IMPORTANT: Plain text only in all string fields — no <cite> tags, no HTML, no markup.

=== RESPONSE FORMAT ===
Respond ONLY with a valid JSON object — no markdown, no preamble:
{{
  "action": "BUY" or "SELL" or "HOLD",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "entry_price": {current_price:.2f},
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "lot_size": {lot_size},
  "reasoning": "Plain text. 2-3 sentences on technicals and key factor driving decision.",
  "sentiment_summary": "Plain text. 1-2 sentences on current news and fundamental backdrop."
}}"""

    try:
        response = _client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )

        # Extract text blocks only — skip tool_use, tool_result, error blocks
        text_parts = []
        for block in response.content:
            if hasattr(block, "type") and block.type == "text":
                text_parts.append(block.text)
            elif isinstance(block, str):
                text_parts.append(block)

        raw = " ".join(text_parts).strip()

        if not raw:
            print(f"[claude_analyst] No text content returned for {ticker}")
            return None

        # Strip accidental markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        # Strip citation tags
        raw = re.sub(r'<cite[^>]*>', '', raw)
        raw = re.sub(r'</cite>', '', raw)

        # Extract first { ... } JSON block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            print(f"[claude_analyst] No JSON found for {ticker}: {raw[:200]}")
            return None

        data = json.loads(match.group())

        # Calculate SL/TP if Claude left them as 0
        entry  = float(data.get("entry_price", current_price))
        sl     = float(data.get("stop_loss", 0))
        tp     = float(data.get("take_profit", 0))
        action = data.get("action", "HOLD")

        if action == "BUY":
            if sl == 0 or sl >= entry:
                sl = round(entry * 0.99, 2)
            if tp == 0 or tp <= entry:
                tp = round(entry + 2 * (entry - sl), 2)
        elif action == "SELL":
            if sl == 0 or sl <= entry:
                sl = round(entry * 1.01, 2)
            if tp == 0 or tp >= entry:
                tp = round(entry - 2 * (sl - entry), 2)

        result = TradeSignal(
            ticker=ticker,
            action=action,
            confidence=data.get("confidence", "LOW"),
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            lot_size=float(data.get("lot_size", lot_size)),
            reasoning=data.get("reasoning", ""),
            sentiment_summary=data.get("sentiment_summary", ""),
            raw_signal=raw_signal,
        )

        # Store in cache before returning
        _store_cache(ticker, result, raw_signal)
        return result

    except Exception as e:
        print(f"[claude_analyst] Error for {ticker}: {e}")
        return None
