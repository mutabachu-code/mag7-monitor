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


def _strip_citations(text: str) -> str:
    text = re.sub(r'<cite[^>]*>', '', text)
    text = re.sub(r'</cite>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()


def _build_prompt(ticker, current_price, raw_signal, rsi, vol_ratio,
                  macd_bullish, trend_status, delta_val, sma200,
                  account_balance, lot_size, include_search_instruction):

    search_instruction = """
STEP 2 — Search for latest news, sentiment, analyst views on {ticker} today.
STEP 3 — Combine technicals + news into final decision.
Downgrade to HOLD if adverse news, earnings risk, or macro contradicts signal.""".format(ticker=ticker) if include_search_instruction else """
STEP 2 — Use your existing knowledge of {ticker}'s recent performance and outlook.
STEP 3 — Combine technicals + knowledge into final decision.""".format(ticker=ticker)

    return f"""You are a disciplined quantitative trading analyst for Magnificent 7
tech stocks traded as CFDs on a small $100 account via Exness MT5.

=== TECHNICAL DATA: {ticker} ===
Current Price      : ${current_price:.2f}
SMA 200 (1H)       : ${sma200:.2f}
Trend vs SMA200    : {trend_status}
RSI (5m, 14-period): {rsi:.1f}
Volume Surge Ratio : {vol_ratio:.2f}x
MACD (1H)          : {"Bullish — MACD above signal" if macd_bullish else "Bearish — MACD below signal"}
Options Delta      : {delta_val:.2f}
Dashboard Signal   : {raw_signal}

=== ACCOUNT & RISK PARAMETERS ===
Account Balance    : ${account_balance:.2f}
Lot Size           : {lot_size} lots
Stop Loss Rule     : 1.0% from entry
Take Profit Rule   : Minimum 2:1 reward-to-risk

=== YOUR TASK ===
STEP 1 — Confirm technicals are valid for a trade.
{search_instruction}

IMPORTANT:
- Return ONLY valid JSON — no markdown, no preamble, no tags of any kind
- All text must be plain text — no <cite>, no HTML, no markup
- entry_price must equal current price: {current_price:.2f}
- stop_loss and take_profit must be exact dollar price levels

=== RESPONSE FORMAT ===
{{
  "action": "BUY" or "SELL" or "HOLD",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "entry_price": {current_price:.2f},
  "stop_loss": <exact price as float>,
  "take_profit": <exact price as float>,
  "lot_size": {lot_size},
  "reasoning": "Plain text. 2-3 sentences on technicals and decision rationale.",
  "sentiment_summary": "Plain text. 1-2 sentences on fundamental or news backdrop."
}}"""


def _parse_and_build_signal(ticker, raw_response, current_price,
                             lot_size, raw_signal) -> Optional[TradeSignal]:
    """Parse Claude's JSON response into a TradeSignal."""

    # Strip markdown fences
    raw = re.sub(r"```(?:json)?", "", raw_response).strip()

    # Extract JSON block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        print(f"[claude_analyst] No JSON found for {ticker}: {raw[:200]}")
        return None

    data = json.loads(match.group())

    entry  = float(data.get("entry_price", current_price))
    sl     = float(data.get("stop_loss", 0))
    tp     = float(data.get("take_profit", 0))
    action = data.get("action", "HOLD")

    # Recalculate SL/TP if missing or invalid direction
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

    return TradeSignal(
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

    # ── ATTEMPT 1: with web search ────────────────────────────────────────────
    try:
        response = _client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": _build_prompt(
                    ticker, current_price, raw_signal, rsi, vol_ratio,
                    macd_bullish, trend_status, delta_val, sma200,
                    account_balance, lot_size,
                    include_search_instruction=True
                )
            }],
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
        )

        # Extract text blocks only — ignore tool_use / tool_result / error blocks
        text_parts = []
        has_error_block = False
        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "error":
                    has_error_block = True

        raw = " ".join(text_parts).strip()

        if raw and not has_error_block:
            return _parse_and_build_signal(
                ticker, raw, current_price, lot_size, raw_signal
            )

        # Web search returned an error block — fall through to attempt 2
        print(f"[claude_analyst] Web search error block for {ticker}, falling back to no-search")

    except Exception as e:
        print(f"[claude_analyst] Web search attempt failed for {ticker}: {e}")

    # ── ATTEMPT 2: fallback without web search ────────────────────────────────
    try:
        response = _client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=600,
            messages=[{
                "role": "user",
                "content": _build_prompt(
                    ticker, current_price, raw_signal, rsi, vol_ratio,
                    macd_bullish, trend_status, delta_val, sma200,
                    account_balance, lot_size,
                    include_search_instruction=False
                )
            }],
            # No web search tool — plain analysis only
        )

        text_parts = [
            block.text for block in response.content
            if hasattr(block, "type") and block.type == "text"
        ]
        raw = " ".join(text_parts).strip()

        if raw:
            return _parse_and_build_signal(
                ticker, raw, current_price, lot_size, raw_signal
            )

    except Exception as e:
        print(f"[claude_analyst] Fallback also failed for {ticker}: {e}")

    return None
