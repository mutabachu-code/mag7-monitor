import anthropic
import json
import re
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
) -> Optional[TradeSignal]:

    prompt = f"""You are a disciplined quantitative trading analyst for Magnificent 7
tech stocks traded as CFDs on a small $100 account via Exness MT5.

=== TECHNICAL DATA: {ticker} ===
Current Price      : ${current_price:.2f}
SMA 200 (1H)       : ${sma200:.2f}
Trend vs SMA200    : {trend_status}
RSI (5m, 14-period): {rsi:.1f}
Volume Surge Ratio : {vol_ratio:.2f}x  (>1.2 = elevated volume)
MACD (1H)          : {"Bullish — MACD line above signal line" if macd_bullish else "Bearish — MACD line below signal line"}
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
STEP 3 — Combine both into a final BUY / SELL / HOLD decision.
Downgrade to HOLD if adverse news, earnings risk, or macro contradicts the signal.

=== RESPONSE FORMAT ===
Respond ONLY with a valid JSON object — no markdown, no preamble:
{{
  "action": "BUY",
  "confidence": "HIGH",
  "entry_price": {current_price:.2f},
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "lot_size": {lot_size},
  "reasoning": "2-3 sentences on technicals and key factor driving decision.",
  "sentiment_summary": "1-2 sentences on current news and fundamental backdrop."
}}"""

    try:
        response = _client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )

        # Safely extract only text blocks — skip tool_use, tool_result, etc.
        text_parts = []
        for block in response.content:
            # block can be TextBlock, ToolUseBlock, ToolResultBlock — only want text
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

        # Extract the first { ... } JSON block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            print(f"[claude_analyst] No JSON found for {ticker}: {raw[:200]}")
            return None

        data = json.loads(match.group())

        # Calculate SL/TP if Claude left them as 0
        entry = float(data.get("entry_price", current_price))
        sl    = float(data.get("stop_loss", 0))
        tp    = float(data.get("take_profit", 0))
        action = data.get("action", "HOLD")

        if action == "BUY":
            if sl == 0:
                sl = round(entry * 0.99, 2)       # 1% below entry
            if tp == 0:
                tp = round(entry + 2 * (entry - sl), 2)   # 2:1 R:R
        elif action == "SELL":
            if sl == 0:
                sl = round(entry * 1.01, 2)       # 1% above entry
            if tp == 0:
                tp = round(entry - 2 * (sl - entry), 2)   # 2:1 R:R

        return TradeSignal(
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

    except Exception as e:
        print(f"[claude_analyst] Error for {ticker}: {e}")
        return None
