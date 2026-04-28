import anthropic
import json
import re
import time
import streamlit as st
from dataclasses import dataclass
from typing import Optional
from forex_volume_profile import VolumeProfile

@dataclass
class FXSignal:
    pair: str
    action: str
    confidence: str
    entry: float
    stop_loss: float
    take_profit: float
    stop_pips: float
    target_pips: float
    lot_size: float
    cb_sentiment: str
    reasoning: str
    news_summary: str
    signal_type: str


_client = anthropic.Anthropic()

# ── CACHE in session_state — survives Streamlit 60s refreshes ────────────────
CACHE_TTL = 600  # 10 minutes


def _cache_valid(pair: str, signal: str) -> bool:
    cache = st.session_state.get("fx_analyst_cache", {})
    if pair not in cache:
        return False
    e = cache[pair]
    return (time.time() - e["ts"]) < CACHE_TTL and e["signal"] == signal


def _store_cache(pair: str, result: "FXSignal", signal: str):
    if "fx_analyst_cache" not in st.session_state:
        st.session_state["fx_analyst_cache"] = {}
    st.session_state["fx_analyst_cache"][pair] = {
        "result": result, "signal": signal, "ts": time.time()
    }


def analyse_pair(vp: VolumeProfile, lot_size: float = 0.02,
                 account_balance: float = 100.0) -> Optional[FXSignal]:

    if _cache_valid(vp.pair, vp.signal):
        print(f"[forex_analyst] Cache hit {vp.pair} — no API call")
        return st.session_state["fx_analyst_cache"][vp.pair]["result"]

    base_ccy  = vp.pair[:3]
    quote_ccy = vp.pair[3:]
    pip       = 0.01 if 'JPY' in vp.pair else 0.0001

    prompt = f"""You are an expert forex analyst for major currency pairs traded as CFDs on Exness MT5.

=== PAIR: {vp.pair} ({base_ccy}/{quote_ccy}) ===
Current Price     : {vp.current_price:.5f}
Day Change        : {vp.day_pct:+.2f}%
RSI (1H)          : {vp.rsi_1h:.1f}

=== VOLUME PROFILE (1H) ===
POC (Fair Value)  : {vp.poc:.5f}
Value Area High   : {vp.vah:.5f}
Value Area Low    : {vp.val:.5f}
Price Location    : {vp.price_location}
Volume Intensity  : {vp.volume_intensity:.2f} (< 0.7 = exhaustion)
Dashboard Signal  : {vp.signal_icon} {vp.signal}

=== TREND FILTERS ===
Daily SMA200      : {vp.sma200_1d:.5f} → {"BULLISH" if vp.trend_bullish else "BEARISH"}
4H SMA50          : {vp.sma50_4h:.5f}  → {"BULLISH" if vp.trend_4h_bullish else "BEARISH"}
ATR               : {vp.atr:.5f} ({vp.atr_pct*100:.3f}%)

=== ACCOUNT ===
Balance: ${account_balance:.2f} | Lot: {lot_size} | Spread: {vp.spread_pips} pips

=== TASK ===
Using your knowledge of current {base_ccy} and {quote_ccy} central bank stance and recent macro:
1. Is {base_ccy} central bank currently HAWKISH, DOVISH, or NEUTRAL?
2. Validate the Volume Profile signal
3. Apply Central Bank Drift Rule: if extreme CB sentiment, disable counter-trend signals
4. Calculate SL at nearest VA edge, TP at next HVN. Min 1.5:1 R:R.
5. Return BUY / SELL / HOLD

Plain text only in strings — no HTML tags.
Return ONLY valid JSON:
{{
  "action": "BUY" or "SELL" or "HOLD",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "entry": {vp.current_price:.5f},
  "stop_loss": <exact price>,
  "take_profit": <exact price>,
  "stop_pips": <float>,
  "target_pips": <float>,
  "lot_size": {lot_size},
  "cb_sentiment": "HAWKISH" or "DOVISH" or "NEUTRAL",
  "signal_type": "MEAN_REVERSION" or "BREAKOUT" or "HOLD",
  "reasoning": "2-3 sentences plain text.",
  "news_summary": "1-2 sentences on CB stance and key risk."
}}"""

    try:
        # No web search — uses Claude's training knowledge of CB stances
        # Re-enable web search only on major event days (NFP, FOMC, CPI)
        response = _client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )

        text_parts = [
            b.text for b in response.content
            if hasattr(b, "type") and b.type == "text"
        ]
        raw = " ".join(text_parts).strip()
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        raw = re.sub(r"<cite[^>]*>|</cite>|<[^>]+>", "", raw)

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            print(f"[forex_analyst] No JSON for {vp.pair}")
            return None

        d = json.loads(match.group())

        result = FXSignal(
            pair=vp.pair,
            action=d.get("action", "HOLD"),
            confidence=d.get("confidence", "LOW"),
            entry=float(d.get("entry", vp.current_price)),
            stop_loss=float(d.get("stop_loss", 0)),
            take_profit=float(d.get("take_profit", 0)),
            stop_pips=float(d.get("stop_pips", 0)),
            target_pips=float(d.get("target_pips", 0)),
            lot_size=float(d.get("lot_size", lot_size)),
            cb_sentiment=d.get("cb_sentiment", "NEUTRAL"),
            reasoning=d.get("reasoning", ""),
            news_summary=d.get("news_summary", ""),
            signal_type=d.get("signal_type", "HOLD"),
        )

        _store_cache(vp.pair, result, vp.signal)
        return result

    except Exception as e:
        print(f"[forex_analyst] Error {vp.pair}: {e}")
        return None
