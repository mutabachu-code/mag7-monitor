"""
regime_detector.py
------------------
Pragmatic market regime detector — no HMM library needed.
Reads from data_fetcher / forex_data_fetcher session_state cache.
Zero extra yfinance calls.

Detects 3 regimes using existing indicators:

  State 0 — QUIET TREND
    VIX < 18  |  ATR% low  |  Trend aligned  |  Breadth healthy
    → Enable trend-following signals (MOMENTUM BUY/SELL, BREAKOUT)
    → Wider TP targets, normal lot sizing

  State 1 — SIDEWAYS / CHOP
    VIX 18-25  |  ATR% medium  |  Mixed trend  |  Narrow breadth
    → Enable mean reversion only (DIP BUY, MEAN REVERSION)
    → Tighter TP, reduce lot size 50%

  State 2 — HIGH VOLATILITY / CRISIS
    VIX > 25  |  ATR% high  |  Yield trap  |  Risk-Off cross-asset
    → HOLD only — no new entries
    → Capital protection mode

Cross-asset Risk-Off detector (from screenshots):
    Gold + JPY + TNX all rising together = Crisis regime override
"""

import pandas as pd
import numpy as np
import streamlit as st
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class RegimeState:
    # Core regime
    state: int              # 0 | 1 | 2
    label: str              # "QUIET TREND" | "SIDEWAYS/CHOP" | "HIGH VOLATILITY"
    color: str              # hex color
    icon: str               # emoji
    confidence: float       # 0.0 - 1.0 (how clearly in this regime)

    # Component scores (0-100 each)
    volatility_score: int   # VIX + ATR based
    trend_score: int        # directional consistency
    breadth_score: int      # market participation
    risk_off_score: int     # cross-asset crisis signal

    # Regime-specific guidance
    allowed_signals: list   # which signal types are valid in this regime
    lot_multiplier: float   # 1.0 = normal, 0.5 = half size, 0.0 = no trade
    strategy_note: str      # human-readable instruction for Claude

    # Risk-Off details
    risk_off_active: bool
    risk_off_reason: str


CACHE_TTL = 65


def _cache_valid(key: str) -> bool:
    return (time.time() - st.session_state.get(f"{key}_ts", 0)) < CACHE_TTL


def _compute_volatility_score(vix: Optional[float], atr_pct: Optional[float]) -> tuple:
    """Returns (score 0-100, vix_state)"""
    score = 0
    vix_val = vix or 18.0
    atr_val = atr_pct or 0.01

    # VIX component (60 pts)
    if vix_val > 30:   score += 60
    elif vix_val > 25: score += 45
    elif vix_val > 20: score += 25
    elif vix_val > 15: score += 10
    else:              score += 0

    # ATR% component (40 pts)
    if atr_val > 0.025:   score += 40
    elif atr_val > 0.015: score += 25
    elif atr_val > 0.008: score += 10
    else:                  score += 0

    return min(score, 100), vix_val


def _compute_trend_score(trend_bullish: bool, macd_bullish: bool,
                          rsi: float, price_vs_sma: float) -> int:
    """Returns score 0-100. High = strong trend (good for State 0)."""
    score = 0
    # Directional alignment
    if trend_bullish and macd_bullish:   score += 40
    elif trend_bullish or macd_bullish:  score += 20

    # RSI not in chop zone (40-60 = chop)
    if rsi < 35 or rsi > 65:            score += 30
    elif rsi < 40 or rsi > 60:          score += 15

    # Distance from SMA (far = trending, near = ranging)
    sma_dist = abs(price_vs_sma)
    if sma_dist > 2.0:   score += 30
    elif sma_dist > 0.5: score += 15
    else:                score += 0

    return min(score, 100)


def _compute_breadth_score(breadth_ratio: Optional[float],
                            risk_score: Optional[int]) -> int:
    """Returns score 0-100. High = broad healthy participation."""
    score = 50   # neutral default

    if breadth_ratio is not None:
        # QQQ vs QQQE gap: small gap = broad market (healthy)
        if abs(breadth_ratio) < 1.0:   score = 80
        elif abs(breadth_ratio) < 2.5: score = 60
        elif abs(breadth_ratio) < 4.0: score = 35
        else:                           score = 15

    # Macro risk score penalty
    if risk_score is not None:
        if risk_score >= 70:   score = max(score - 40, 0)
        elif risk_score >= 40: score = max(score - 20, 0)

    return score


def _check_risk_off(gold_df, jpy_df, tnx_df) -> tuple:
    """
    Cross-asset Risk-Off: Gold + JPY + TNX bonds all rising = Crisis.
    Returns (is_risk_off: bool, reason: str, score: int)
    """
    signals = []
    score   = 0

    # Gold rising (GC=F or GLD)
    if gold_df is not None and len(gold_df) >= 5:
        gold_ret = float(gold_df['Close'].pct_change(5).iloc[-1]) * 100
        if gold_ret > 1.0:
            signals.append(f"Gold +{gold_ret:.1f}%")
            score += 35

    # JPY strengthening (USDJPY falling = JPY up)
    if jpy_df is not None and len(jpy_df) >= 5:
        jpy_ret = float(jpy_df['Close'].pct_change(5).iloc[-1]) * 100
        if jpy_ret < -0.8:   # USDJPY falling = JPY strengthening
            signals.append(f"JPY +{abs(jpy_ret):.1f}%")
            score += 35

    # TNX bonds (yields falling = bonds rising = safety buying)
    if tnx_df is not None and len(tnx_df) >= 5:
        tnx_ret = float(tnx_df['Close'].pct_change(5).iloc[-1]) * 100
        if tnx_ret < -3.0:   # yields falling sharply
            signals.append(f"Bond yields -{abs(tnx_ret):.1f}%")
            score += 30

    is_risk_off = score >= 60   # need at least 2 of 3 signals
    reason = " | ".join(signals) if signals else "No risk-off signals"
    return is_risk_off, reason, score


def detect_regime_stocks(
    vix: Optional[float] = None,
    atr_pct: Optional[float] = None,
    trend_bullish: bool = True,
    macd_bullish: bool = True,
    rsi: float = 50.0,
    price_vs_sma_pct: float = 1.0,
    breadth_ratio: Optional[float] = None,
    macro_risk_score: Optional[int] = None,
    gold_df=None,
    tnx_df=None,
) -> RegimeState:
    """
    Detect regime for Mag 7 / stock dashboard.
    Uses VIX, ATR, trend alignment, market breadth, and cross-asset signals.
    """
    cache_key = "regime_stocks"
    if _cache_valid(cache_key) and cache_key in st.session_state:
        return st.session_state[cache_key]

    vol_score, vix_val       = _compute_volatility_score(vix, atr_pct)
    trend_score              = _compute_trend_score(trend_bullish, macd_bullish,
                                                    rsi, price_vs_sma_pct)
    breadth_score            = _compute_breadth_score(breadth_ratio, macro_risk_score)
    risk_off, ro_reason, ro_score = _check_risk_off(gold_df, None, tnx_df)

    # ── REGIME CLASSIFICATION ─────────────────────────────────────────────────
    # State 2: High volatility or risk-off crisis
    if vol_score >= 55 or risk_off or (macro_risk_score or 0) >= 70:
        state   = 2
        label   = "HIGH VOLATILITY / CRISIS"
        color   = "#8b0000"
        icon    = "🔴"
        allowed = []   # no new trades
        lot_mul = 0.0
        note    = ("CRISIS regime: Capital protection mode. "
                   "No new entries. Wait for VIX < 20 and risk-off to resolve.")
        conf    = min(vol_score / 100 + (0.3 if risk_off else 0), 1.0)

    # State 1: Sideways / Chop
    elif vol_score >= 25 or trend_score < 40 or breadth_score < 40:
        state   = 1
        label   = "SIDEWAYS / CHOP"
        color   = "#e6a817"
        icon    = "🟡"
        allowed = ["DIP BUY", "CAUTION BUY", "MEAN REVERSION"]
        lot_mul = 0.5   # half position size in chop
        note    = ("CHOP regime: Mean reversion only. "
                   "Avoid momentum/breakout signals — likely to be fakeouts. "
                   "Use 50% normal lot size.")
        conf    = 0.5 + (vol_score / 200)

    # State 0: Quiet Trend
    else:
        state   = 0
        label   = "QUIET TREND"
        color   = "#2d9e2d"
        icon    = "🟢"
        allowed = ["MOMENTUM BUY", "STRONG BUY", "MOMENTUM SELL",
                   "STRONG SELL", "DIP BUY", "CAUTION BUY"]
        lot_mul = 1.0
        note    = ("TRENDING regime: All signal types valid. "
                   "Favour momentum and breakout signals. "
                   "Normal lot sizing applies.")
        conf    = min((100 - vol_score) / 100 + (breadth_score / 200), 1.0)

    result = RegimeState(
        state=state, label=label, color=color, icon=icon, confidence=conf,
        volatility_score=vol_score, trend_score=trend_score,
        breadth_score=breadth_score, risk_off_score=ro_score,
        allowed_signals=allowed, lot_multiplier=lot_mul, strategy_note=note,
        risk_off_active=risk_off, risk_off_reason=ro_reason,
    )

    st.session_state[cache_key]           = result
    st.session_state[f"{cache_key}_ts"]   = time.time()
    return result


def detect_regime_forex(
    vix: Optional[float] = None,
    atr_pct: Optional[float] = None,
    pair: str = "",
    trend_bullish: bool = True,
    rsi: float = 50.0,
    vol_intensity: float = 1.0,
    macro_risk_score: Optional[int] = None,
    gold_df=None,
    jpy_df=None,
    tnx_df=None,
) -> RegimeState:
    """
    Detect regime for Forex dashboard.
    Adds JPY safe-haven check to cross-asset risk-off.
    """
    cache_key = f"regime_fx_{pair}"
    if _cache_valid(cache_key) and cache_key in st.session_state:
        return st.session_state[cache_key]

    vol_score, vix_val       = _compute_volatility_score(vix, atr_pct)
    trend_score              = _compute_trend_score(trend_bullish, True, rsi, 1.0)
    breadth_score            = _compute_breadth_score(None, macro_risk_score)
    risk_off, ro_reason, ro_score = _check_risk_off(gold_df, jpy_df, tnx_df)

    # Volume intensity: low = exhaustion = chop
    if vol_intensity < 0.6:
        vol_score = min(vol_score + 20, 100)

    # Is this a safe-haven pair? (CHF, JPY benefit in crisis)
    is_safe_haven = any(sh in pair for sh in ['JPY', 'CHF'])
    is_risk_pair  = any(rp in pair for rp in ['AUD', 'NZD', 'CAD'])

    # ── REGIME CLASSIFICATION ─────────────────────────────────────────────────
    if vol_score >= 55 or (macro_risk_score or 0) >= 70:
        state = 2
        label = "HIGH VOLATILITY / CRISIS"
        color = "#8b0000"
        icon  = "🔴"
        if is_safe_haven and risk_off:
            allowed = ["BREAKOUT BUY"]   # CHF/JPY can still trade in crisis
            lot_mul = 0.5
            note    = (f"CRISIS: {pair} is a safe-haven. "
                       "Risk-off flows favour this pair. "
                       "Only BREAKOUT BUY with 50% lot size.")
        else:
            allowed = []
            lot_mul = 0.0
            note    = (f"CRISIS regime: No new entries on {pair}. "
                       "Risk-off environment — avoid risk currencies (AUD/NZD). "
                       "Wait for VIX < 20.")
        conf = min(vol_score / 100, 1.0)

    elif vol_score >= 25 or trend_score < 35:
        state   = 1
        label   = "SIDEWAYS / CHOP"
        color   = "#e6a817"
        icon    = "🟡"
        allowed = ["MEAN REVERSION BUY", "MEAN REVERSION SELL", "MEAN REVERSION",
                   "POC LEVEL"]
        lot_mul = 0.5
        note    = ("CHOP regime: Mean reversion to POC/VA edges only. "
                   "Avoid breakout signals — likely fakeouts in low-liquidity chop.")
        conf    = 0.5

    else:
        state   = 0
        label   = "QUIET TREND"
        color   = "#2d9e2d"
        icon    = "🟢"
        allowed = ["BREAKOUT BUY", "BREAKOUT SELL",
                   "MEAN REVERSION BUY", "MEAN REVERSION SELL",
                   "MEAN REVERSION", "POC LEVEL"]
        lot_mul = 1.0
        note    = ("TRENDING regime: All Volume Profile signals valid. "
                   "Breakout signals have highest probability in this regime.")
        conf    = min((100 - vol_score) / 100, 1.0)

    result = RegimeState(
        state=state, label=label, color=color, icon=icon, confidence=conf,
        volatility_score=vol_score, trend_score=trend_score,
        breadth_score=breadth_score, risk_off_score=ro_score,
        allowed_signals=allowed, lot_multiplier=lot_mul, strategy_note=note,
        risk_off_active=risk_off, risk_off_reason=ro_reason,
    )

    st.session_state[cache_key]          = result
    st.session_state[f"{cache_key}_ts"]  = time.time()
    return result


def render_regime_badge(regime: RegimeState, show_details: bool = False):
    """Render a compact regime badge for use in ticker cards."""
    st.markdown(
        f"<div style='padding:6px 10px;border-radius:6px;"
        f"background:{regime.color}22;border:1px solid {regime.color};"
        f"margin:4px 0;display:inline-block'>"
        f"<span style='color:{regime.color};font-weight:bold'>"
        f"{regime.icon} {regime.label}</span> "
        f"<span style='color:#aaa;font-size:0.85em'>"
        f"({regime.confidence*100:.0f}% confidence)</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    if show_details:
        c1, c2, c3 = st.columns(3)
        c1.caption(f"Volatility: {regime.volatility_score}/100")
        c2.caption(f"Trend: {regime.trend_score}/100")
        c3.caption(f"Breadth: {regime.breadth_score}/100")
        if regime.risk_off_active:
            st.error(f"⚠️ Risk-Off: {regime.risk_off_reason}")
        if regime.lot_multiplier < 1.0:
            mult_txt = "HALT" if regime.lot_multiplier == 0 else f"{regime.lot_multiplier}x"
            st.caption(f"Lot multiplier: {mult_txt} | {regime.strategy_note[:80]}...")


def render_regime_panel(regime: RegimeState, title: str = "Market Regime"):
    """Render full regime panel for dashboard header."""
    st.markdown(
        f"<div style='padding:12px;border-radius:8px;"
        f"background:{regime.color}22;border:2px solid {regime.color};margin-bottom:8px'>"
        f"<span style='font-size:1.2em;font-weight:bold;color:{regime.color}'>"
        f"{regime.icon} Regime: {regime.label}</span>"
        f"<span style='color:#aaa;margin-left:16px;font-size:0.9em'>"
        f"Confidence: {regime.confidence*100:.0f}%</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volatility Score", f"{regime.volatility_score}/100",
              delta="High risk" if regime.volatility_score > 50 else "Normal",
              delta_color="inverse")
    c2.metric("Trend Score",      f"{regime.trend_score}/100",
              delta="Strong" if regime.trend_score > 60 else "Weak")
    c3.metric("Breadth Score",    f"{regime.breadth_score}/100",
              delta="Broad" if regime.breadth_score > 60 else "Narrow")
    c4.metric("Risk-Off Score",   f"{regime.risk_off_score}/100",
              delta="⚠️ Active" if regime.risk_off_active else "Clear",
              delta_color="inverse" if regime.risk_off_active else "normal")

    if regime.risk_off_active:
        st.error(f"🚨 Cross-Asset Risk-Off: {regime.risk_off_reason}")

    st.info(f"📋 Strategy: {regime.strategy_note}")

    if regime.state == 2:
        st.error("🔴 CRISIS MODE: No new trade entries. Protect capital.")
    elif regime.state == 1:
        st.warning(f"🟡 CHOP MODE: Lot size reduced to {regime.lot_multiplier}x. Mean reversion only.")
    else:
        st.success("🟢 TREND MODE: All signals valid. Normal lot sizing.")
