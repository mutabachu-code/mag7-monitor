import anthropic
import json
import re
import time
import streamlit as st
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

# ── CACHE — stored in Streamlit session_state so it survives 60s refreshes ───
# Module-level dicts get wiped on Streamlit Cloud process restarts.
# session_state persists for the lifetime of the browser session.
CACHE_TTL = 600  # 10 minutes — increased from 5 to further reduce calls


def _is_cache_valid(ticker: str, current_raw_signal: str) -> bool:
    cache = st.session_state.get("analyst_cache", {})
    if ticker not in cache:
        return False
    entry = cache[ticker]
    age   = time.time() - entry["timestamp"]
    return age < CACHE_TTL and entry["raw_signal"] == current_raw_signal


def _store_cache(ticker: str, signal: "TradeSignal", raw_signal: str):
    if "analyst_cache" not in st.session_state:
        st.session_state["analyst_cache"] = {}
    st.session_state["analyst_cache"][ticker] = {
        "signal":    signal,
        "raw_signal": raw_signal,
        "timestamp": time.time(),
    }


def _get_cache(ticker: str) -> Optional["TradeSignal"]:
    cache = st.session_state.get("analyst_cache", {})
    return cache.get(ticker, {}).get("signal")


def _strip_citations(text: str) -> str:
    text = re.sub(r'<cite[^>]*>', '', text)
    text = re.sub(r'</cite>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


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
        print(f"[claude_analyst] Cache hit {ticker} — no API call")
        return _get_cache(ticker)

    prompt = f"""You are a disciplined quantitative trading analyst for Magnificent 7
tech stocks traded as CFDs on a small $100 account via Exness MT5.

=== TECHNICAL DATA: {ticker} ===
Current Price      : ${current_price:.2f}
SMA 200 (1H)       : ${sma200:.2f}
Trend vs SMA200    : {trend_status}
RSI (5m, 14-period): {rsi:.1f}
Volume Surge Ratio : {vol_ratio:.2f}x
MACD (1H)          : {"Bullish — MACD above signal" if macd_bullish else "Bearish — MACD below signal"}
Options Delta      : {delta_val:.2f}
Implied Volatility : {implied_volatility}
Dashboard Signal   : {raw_signal}

=== ACCOUNT & RISK PARAMETERS ===
Account Balance    : ${account_balance:.2f}
Lot Size           : {lot_size} lots
Stop Loss Rule     : 1.0% from entry
Take Profit Rule   : Minimum 2:1 reward-to-risk

=== YOUR TASK ===
STEP 1 — Confirm technicals are valid for a trade.
STEP 2 — Use your knowledge of {ticker} recent performance, analyst views, and macro context.
STEP 3 — Combine both into a final BUY / SELL / HOLD decision.
Consider IV: HIGH IV Rank = tighter SL. LOW IV Rank = wider TP targets.

IMPORTANT: Plain text only — no <cite> tags, no HTML, no markup.
Return ONLY valid JSON:
{{
  "action": "BUY" or "SELL" or "HOLD",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "entry_price": {current_price:.2f},
  "stop_loss": <exact price float>,
  "take_profit": <exact price float>,
  "lot_size": {lot_size},
  "reasoning": "Plain text. 2-3 sentences.",
  "sentiment_summary": "Plain text. 1-2 sentences on fundamental backdrop."
}}"""

    try:
        # Web search disabled by default — only enable during peak sessions
        # to keep token costs low. Claude uses its training knowledge instead.
        response = _client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
            # NOTE: web_search removed — saves ~3000-8000 tokens per call
            # Re-enable only if you want live news: tools=[{"type": "web_search_20250305", "name": "web_search"}]
        )

        text_parts = [
            b.text for b in response.content
            if hasattr(b, "type") and b.type == "text"
        ]
        raw = " ".join(text_parts).strip()
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        raw = re.sub(r'<cite[^>]*>|</cite>', '', raw)

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            print(f"[claude_analyst] No JSON for {ticker}")
            return None

        data   = json.loads(match.group())
        entry  = float(data.get("entry_price", current_price))
        sl     = float(data.get("stop_loss", 0))
        tp     = float(data.get("take_profit", 0))
        action = data.get("action", "HOLD")

        if action == "BUY":
            if sl == 0 or sl >= entry: sl = round(entry * 0.99, 2)
            if tp == 0 or tp <= entry: tp = round(entry + 2 * (entry - sl), 2)
        elif action == "SELL":
            if sl == 0 or sl <= entry: sl = round(entry * 1.01, 2)
            if tp == 0 or tp >= entry: tp = round(entry - 2 * (sl - entry), 2)

        result = TradeSignal(
            ticker=ticker,
            action=action,
            confidence=data.get("confidence", "LOW"),
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            lot_size=float(data.get("lot_size", lot_size)),
            reasoning=_strip_citations(data.get("reasoning", "")),
            sentiment_summary=_strip_citations(data.get("sentiment_summary", "")),
            raw_signal=raw_signal,
        )

        _store_cache(ticker, result, raw_signal)
        return result

    except Exception as e:
        print(f"[claude_analyst] Error for {ticker}: {e}")
        return None
