import anthropic
import json
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class TradeSignal:
    ticker: str
    action: str           # "BUY" | "SELL" | "HOLD"
    confidence: str       # "HIGH" | "MEDIUM" | "LOW"
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float       # fixed 0.02 for now
    reasoning: str
    sentiment_summary: str
    raw_signal: str


_client = anthropic.Anthropic(ANTHROPIC_API_KEY = "sk-ant-...")


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
    """
    Call Claude with full indicator set + web search for news/sentiment/fundamentals.
    Returns a TradeSignal, or None on failure.
    """

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
Max Risk per Trade : 2% of account (${account_balance * 0.02:.2f})
Stop Loss Rule     : Place SL at the nearest significant level — the greater of:
                     (a) 1.0% below entry for BUY / above entry for SELL, OR
                     (b) below/above the most recent swing high/low based on price context
Take Profit Rule   : Minimum 2:1 reward-to-risk ratio from entry

=== YOUR FULL ANALYSIS TASK ===

STEP 1 — TECHNICALS
Confirm whether the technical setup is valid:
- Price must be clearly above/below SMA200 (trend confirmation)
- RSI must be oversold (<40) for BUY or overbought (>60) for SELL
- MACD must align with the trade direction
- Volume ratio >= 1.1 (participation)
Only proceed to steps 2-3 if technicals are valid. If not, return HOLD immediately.

STEP 2 — SENTIMENT, NEWS & FUNDAMENTALS
Search your knowledge for the most recent information on {ticker}:
- Any significant recent news (earnings, product launches, regulatory issues, guidance)
- Current market sentiment (analyst ratings, institutional positioning)
- Relevant macro factors affecting this stock today (Fed policy, sector rotation, 
  broader Nasdaq trend, any breaking news)
- Whether the fundamental picture supports or contradicts the technical signal

STEP 3 — FINAL DECISION
Combine technicals + fundamentals + sentiment into a single decision.
A technically valid signal should be DOWNGRADED to HOLD if:
- There is adverse news pending or just released (earnings risk, regulatory scrutiny)
- Sentiment is strongly against the direction
- Macro environment contradicts the trade
Calculate exact SL and TP prices based on the rules above.

=== RESPONSE FORMAT ===
Respond ONLY with valid JSON — no markdown, no preamble, no explanation outside the JSON:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "entry_price": <current price as float>,
  "stop_loss": <absolute price level as float>,
  "take_profit": <absolute price level as float>,
  "lot_size": {lot_size},
  "reasoning": "<2-3 sentences: technical setup + key news/sentiment factor that drove the decision>",
  "sentiment_summary": "<1-2 sentences on the current fundamental/news backdrop for {ticker}>"
}}"""

    try:
        response = _client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )

        # Collect all text blocks (web search may produce multiple content blocks)
        raw = "".join(
            block.text for block in response.content
            if hasattr(block, "text")
        ).strip()

        # Strip accidental markdown fences
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        # Extract JSON — find first { ... } block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in response: {raw[:200]}")

        data = json.loads(match.group())

        return TradeSignal(
            ticker=ticker,
            action=data["action"],
            confidence=data["confidence"],
            entry_price=float(data["entry_price"]),
            stop_loss=float(data["stop_loss"]),
            take_profit=float(data["take_profit"]),
            lot_size=float(data.get("lot_size", lot_size)),
            reasoning=data["reasoning"],
            sentiment_summary=data.get("sentiment_summary", ""),
            raw_signal=raw_signal,
        )

    except Exception as e:
        print(f"[claude_analyst] Error for {ticker}: {e}")
        return None
