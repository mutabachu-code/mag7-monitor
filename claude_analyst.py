"""
claude_analyst.py
-----------------
Sends Mag 7 indicator data to Claude and returns a structured trade signal.
Drop this file into the same directory as app.py.
"""

import anthropic
import json
import re
import dataclasses
from dataclasses import dataclass
from typing import Optional


@dataclass
class TradeSignal:
    ticker: str
    action: str          # "BUY" | "SELL" | "HOLD"
    confidence: str      # "HIGH" | "MEDIUM" | "LOW"
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float   # % of account to risk (e.g. 1.0 = 1%)
    reasoning: str
    raw_signal: str      # the original signal string from app.py


# One shared Anthropic client (reads ANTHROPIC_API_KEY from environment)
_client = anthropic.Anthropic()


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
) -> Optional[TradeSignal]:
    """
    Call Claude with the full indicator set for one ticker.
    Returns a TradeSignal dataclass, or None on failure.
    """

    prompt = f"""You are a disciplined quantitative trading analyst specialising in 
Magnificent 7 tech stocks traded as CFDs. Analyse the following real-time 
technical data and return a structured trade decision.

=== MARKET DATA: {ticker} ===
Current Price      : ${current_price:.2f}
SMA 200 (1H)       : ${sma200:.2f}
Trend (vs SMA200)  : {trend_status}
RSI (5m, 14)       : {rsi:.1f}
Volume Surge Ratio : {vol_ratio:.2f}x  (>1.2 = elevated)
MACD (1H)          : {"Bullish - MACD line above signal" if macd_bullish else "Bearish - MACD line below signal"}
Options Delta      : {delta_val:.2f}
Dashboard Signal   : {raw_signal}

=== YOUR TASK ===
1. Assess whether this is a genuine trade opportunity or noise.
2. ONLY recommend BUY or SELL if ALL of the following are true:
   - Strong trend confirmation (price clearly above/below SMA200)
   - RSI confirms (oversold <40 for BUY, overbought >60 for SELL)
   - MACD aligns with direction
   - Volume ratio >= 1.1 (participation confirms the move)
3. For HOLD, briefly state what condition is missing.
4. Use tight risk management: stop-loss within 1.5% of entry, 
   take-profit at 2:1 reward-to-risk minimum.
5. Max position size recommendation: 2% of account per trade.

=== RESPONSE FORMAT ===
Respond ONLY with a valid JSON object, no markdown, no preamble:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "entry_price": <float>,
  "stop_loss": <float>,
  "take_profit": <float>,
  "position_size_pct": <float between 0.5 and 2.0>,
  "reasoning": "<one concise sentence explaining the decision>"
}}"""

    try:
        response = _client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()

        # Strip any accidental markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        data = json.loads(raw)

        return TradeSignal(
            ticker=ticker,
            action=data["action"],
            confidence=data["confidence"],
            entry_price=float(data["entry_price"]),
            stop_loss=float(data["stop_loss"]),
            take_profit=float(data["take_profit"]),
            position_size_pct=float(data["position_size_pct"]),
            reasoning=data["reasoning"],
            raw_signal=raw_signal,
        )

    except Exception as e:
        # Non-fatal: return None, app.py will show an error card
        print(f"[claude_analyst] Error for {ticker}: {e}")
        return None
