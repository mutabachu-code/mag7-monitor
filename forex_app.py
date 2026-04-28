import google.generativeai as genai
import json
import time
from dataclasses import dataclass
from typing import Optional
from forex_volume_profile import VolumeProfile

# --- KEEP YOUR EXISTING DATACLASS ---
@dataclass
class FXSignal:
    pair: str
    action: str            # BUY | SELL | HOLD
    confidence: str        # HIGH | MEDIUM | LOW
    entry: float
    stop_loss: float
    take_profit: float
    stop_pips: float
    target_pips: float
    lot_size: float
    cb_sentiment: str      # HAWKISH | DOVISH | NEUTRAL
    reasoning: str
    news_summary: str
    signal_type: str       # MEAN_REVERSION | BREAKOUT | HOLD

# --- NEW GEMINI CONFIGURATION ---
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GEMINI_API_KEY)

# We move your expert logic into the System Instruction for efficiency
SYSTEM_LOGIC = """
You are an expert forex analyst specializing in Volume Profile and institutional order flow.
Analyze the provided setup and return a structured trade decision in JSON.

CENTRAL BANK DRIFT RULE:
- If Base CCY sentiment is EXTREME HAWKISH: Disable MEAN REVERSION SELL, only take BREAKOUT BUY.
- If Base CCY sentiment is EXTREME DOVISH: Disable MEAN REVERSION BUY, only take BREAKOUT SELL.

TRADING RULES:
- MEAN REVERSION: Volume exhaustion at VA edges + trend alignment.
- BREAKOUT: Price acceptance outside VA + low ATR entry.
- RISK/REWARD: Minimum 1.5:1. Account for spread.

Return ONLY a JSON object. No conversational text.
"""

# Initialize model once (saves latency)
_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash", 
    system_instruction=SYSTEM_LOGIC
)

_cache: dict = {}
CACHE_TTL = 300 

def _cache_valid(pair: str, signal: str) -> bool:
    if pair not in _cache: return False
    e = _cache[pair]
    return (time.time() - e["ts"]) < CACHE_TTL and e["signal"] == signal

def analyse_pair(vp: VolumeProfile, lot_size: float = 0.02,
                 account_balance: float = 100.0) -> Optional[FXSignal]:

    if _cache_valid(vp.pair, vp.signal):
        return _cache[vp.pair]["result"]

    # Session check and Neutral check handled by app.py, but we keep the safety here
    if "NEUTRAL" in vp.signal or "INSIDE" in vp.signal or "Watch" in vp.signal:
        return None

    base_ccy  = vp.pair[:3]
    quote_ccy = vp.pair[3:]

    # Prompt focuses strictly on the dynamic data for the current date: April 28, 2026
    prompt = f"""
ANALYZE PAIR: {vp.pair}
Price: {vp.current_price:.5f} | RSI: {vp.rsi_1h:.1f}
Volume Profile: POC: {vp.poc:.5f}, VAH: {vp.vah:.5f}, VAL: {vp.val:.5f}
Location: {vp.price_location} | Intensity: {vp.volume_intensity:.2f}
Trend: 1D: {"BULLISH" if vp.trend_bullish else "BEARISH"} | 4H: {"BULLISH" if vp.trend_4h_bullish else "BEARISH"}
ATR: {vp.atr:.5f} | Spread: {vp.spread_pips} pips
Account: ${account_balance:.2f} | Lot: {lot_size}

TASK: Check latest Central Bank stance (Fed/ECB/BOJ) and risk sentiment for April 28, 2026.
Return JSON ONLY.
"""

    try:
        # Use Gemini's native JSON constrained output
        response = _model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1
            }
        )

        d = json.loads(response.text)

        # Mapping the JSON directly to your FXSignal dataclass
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

        _cache[vp.pair] = {"result": result, "signal": vp.signal, "ts": time.time()}
        return result

    except Exception as e:
        print(f"[forex_analyst] Gemini Error {vp.pair}: {e}")
        return None
