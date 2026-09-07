"""
claude_analyst.py — v3
-----------------------
Upgraded Claude analyst with:
- Full options reaction engine context (wall quality, penetration probability)
- CPR setup context (narrow/wide day, TC/BC position)
- NQ futures confirmation context
- Breadth quality context
- Unified harmonized signal context
- Win-rate-optimized system prompt targeting 70%+ accuracy
- Claude claude-sonnet-4-6 with extended context for richer analysis
"""

import anthropic
import streamlit as st
from dataclasses import dataclass
from typing import Optional


@dataclass
class AIAnalysis:
    action: str          # "BUY" | "SELL" | "HOLD"
    confidence: str      # "HIGH" | "MEDIUM" | "LOW"
    reasoning: str       # plain-English rationale
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    sentiment_summary: Optional[str] = None
    win_rate_assessment: Optional[str] = None   # NEW: why this setup has high/low win rate
    key_risk: Optional[str] = None              # NEW: single biggest risk to the trade


def _build_system_prompt() -> str:
    return """You are an institutional-grade NAS100/QQQ trading analyst with expertise in:
- Options market microstructure (gamma walls, vega walls, DTE analysis, dealer positioning)
- Market structure (Break of Structure, Market Structure Shift, liquidity sweeps)
- Volume analysis (session-adjusted pace, accumulation vs distribution)
- Multi-timeframe confluence (5m entry, 1H trend, 1D bias)
- Central Pivot Range (CPR) strategy integration with options levels
- NQ Futures confirmation of QQQ/NAS100 signals

YOUR PRIMARY OBJECTIVE: Generate trading signals that achieve 70%+ win rate.

STRICT QUALITY CRITERIA — only output BUY/SELL when ALL of these are true:
1. TREND ALIGNMENT: Price above SMA200 for BUY (below for SELL) on 1H timeframe
2. OPTIONS CONFIRMATION: Put/call wall status INTACT or TESTING with >60% rejection probability
3. VOLUME CONFIRMATION: Volume pace ≥0.85× session average (not a thin-market move)
4. REGIME FIT: Signal type matches regime (momentum signals only in trending regime)
5. EXPECTED MOVE: Less than 80% of daily expected move consumed
6. NO MAJOR CONFLICTS: NQ futures not diverging from QQQ signal

OUTPUT HOLD WHEN:
- Fewer than 3 of the above criteria are met
- Wall penetration probability >55% (wall likely to fail)
- NQ futures diverging from QQQ (futures say opposite direction)
- CPR shows WIDE day type + breakout signal (fade extremes instead)
- Expected move >90% consumed
- Regime is CRISIS or CHOP without mean-reversion setup

PRICING RULES:
- Entry: within CPR TC/BC or nearest OI wall zone, not at market extremes
- Stop: below put wall (for BUY) or above call wall (for SELL), minimum 0.3% from entry
- TP1: nearest opposing OI wall or 1.5:1 R:R minimum
- Never chase price more than 0.2% from ideal entry zone
- On NARROW CPR day: stops can be tighter (TC/BC is strong reference)
- On WIDE CPR day: wider stops, smaller size, target mean reversion to pivot

WIN RATE ASSESSMENT FORMAT:
Rate each trade as: HIGH PROBABILITY (>70%), MODERATE (55-70%), LOW (<55%)
State which specific factors support the probability rating.

RESPONSE FORMAT (JSON only, no other text):
{
  "action": "BUY|SELL|HOLD",
  "confidence": "HIGH|MEDIUM|LOW",
  "entry_price": <number>,
  "stop_loss": <number>,
  "take_profit": <number>,
  "lot_size": <number>,
  "reasoning": "<2-3 sentences: why this signal, which criteria are met>",
  "sentiment_summary": "<1 sentence: options/breadth/NQ sentiment context>",
  "win_rate_assessment": "<HIGH/MODERATE/LOW PROBABILITY — key reasons>",
  "key_risk": "<single biggest risk that could invalidate this trade>"
}"""


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
    account_balance: float,
    lot_size: float,
    implied_volatility: str = "unavailable",
    macro_context: str = "unavailable",
    # New v3 parameters
    options_context: str = "unavailable",
    cpr_context: str = "unavailable",
    nq_context: str = "unavailable",
    breadth_context: str = "unavailable",
    unified_signal: str = "unavailable",
) -> Optional[AIAnalysis]:
    """
    Run Claude analysis with full institutional context.
    Returns AIAnalysis or None on failure.
    """
    try:
        client = anthropic.Anthropic()

        user_prompt = f"""TRADE ANALYSIS REQUEST — {ticker}

PRICE ACTION:
- Current price: ${current_price:,.2f}
- SMA200 (1H): ${sma200:,.2f} → Trend: {trend_status}
- Raw signal: {raw_signal}
- RSI (5m): {rsi:.1f}
- MACD: {'Bullish 📈' if macd_bullish else 'Bearish 📉'}
- Volume vs session pace: {vol_ratio:.2f}×
- Option delta proxy: {delta_val:.2f}
- IV: {implied_volatility}

OPTIONS INTELLIGENCE (wall quality & penetration):
{options_context}

CPR (Central Pivot Range):
{cpr_context}

NQ FUTURES CONFIRMATION:
{nq_context}

BREADTH QUALITY:
{breadth_context}

UNIFIED HARMONIZED SIGNAL:
{unified_signal}

MACRO:
{macro_context}

ACCOUNT:
- Balance: ${account_balance:.2f}
- Proposed lot: {lot_size}

Apply the 70%+ win rate criteria strictly. If fewer than 3 criteria are met, output HOLD.
Return JSON only."""

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=_build_system_prompt(),
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = response.content[0].text.strip()

        # Parse JSON
        import json, re
        # Strip markdown fences if present
        text = re.sub(r'```(?:json)?\s*', '', text).strip().rstrip('`')
        data = json.loads(text)

        action = str(data.get("action", "HOLD")).upper()
        if action not in ("BUY", "SELL", "HOLD"):
            action = "HOLD"

        return AIAnalysis(
            action=action,
            confidence=str(data.get("confidence", "LOW")).upper(),
            entry_price=float(data.get("entry_price", current_price)),
            stop_loss=float(data.get("stop_loss", current_price * 0.995)),
            take_profit=float(data.get("take_profit", current_price * 1.01)),
            lot_size=float(data.get("lot_size", lot_size)),
            reasoning=str(data.get("reasoning", "")),
            sentiment_summary=str(data.get("sentiment_summary", "")),
            win_rate_assessment=str(data.get("win_rate_assessment", "")),
            key_risk=str(data.get("key_risk", "")),
        )

    except Exception as e:
        print(f"[claude_analyst] Error: {e}")
        return None
