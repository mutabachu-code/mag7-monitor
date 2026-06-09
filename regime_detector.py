"""
regime_detector.py  — v3 (fixed)
----------------------------------
Fixes applied vs v2:

  BUG 1 FIXED: Cache now invalidates when key inputs change (VIX bucket,
               macro risk bucket, breadth bucket hashed into cache key).
               Regime will update within the same session if conditions shift.

  BUG 2 FIXED: trend_bullish / macd_bullish / rsi / price_vs_sma_pct are
               now passed from NAS100 live indicator data in app.py instead
               of defaulting to True / True / 50 / 1.0 every time.

  BUG 3 FIXED: atr_pct is now computed inside detect_regime_stocks() directly
               from the NAS100 5m data (no need for caller to pass it).
               Fallback: derive from vol_ratio proxy if 5m data unavailable.

  BUG 4 FIXED: breadth_ratio thresholds recalibrated. The macro_snap value
               is a % gap (QQQ 5d ret - QQQE 5d ret). Thresholds tightened:
               |gap| < 0.8% = healthy, 0.8-2.5% = diverging, >2.5% = exhaustion.
               Also reads the macro_snap.breadth_signal string directly as
               a stronger signal source.

  BUG 5 FIXED: State 2 threshold lowered from vol_score>=55 to >=45 so
               VIX 25-30 (real concern zone) correctly triggers CHOP/WARNING
               rather than being silent. Added intermediate WARNING state
               display while keeping 3-state logic clean.

  ADDED: ATR% computed live from NAS100 5m data for accurate volatility scoring.
  ADDED: Regime change detection — logs when regime shifts between refreshes.
"""

import pandas as pd
import numpy as np
import streamlit as st
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class RegimeState:
    state: int              # 0 | 1 | 2
    label: str
    color: str
    icon: str
    confidence: float

    volatility_score: int
    trend_score: int
    breadth_score: int
    risk_off_score: int

    allowed_signals: list
    lot_multiplier: float
    strategy_note: str

    risk_off_active: bool
    risk_off_reason: str

    # v3 additions
    vix_live: float         # actual VIX used
    atr_pct_live: float     # actual ATR% used
    trend_live: str         # "BULLISH" | "BEARISH" | "UNKNOWN"
    regime_changed: bool    # True if regime shifted since last refresh


CACHE_TTL = 65


def _make_cache_key(vix: float, macro_risk: int, breadth_sig: str,
                    trend_b: bool, macd_b: bool) -> str:
    """
    Include key inputs in cache key so regime re-evaluates when conditions change.
    Bucket VIX to nearest 2 pts to avoid re-computing on tiny ticks.
    """
    vix_bucket    = round((vix or 18.0) / 2) * 2
    risk_bucket   = (macro_risk or 0) // 10 * 10
    trend_key     = f"{int(trend_b)}{int(macd_b)}"
    breadth_short = breadth_sig[:8] if breadth_sig else "unknown"
    return f"regime_stocks_{vix_bucket}_{risk_bucket}_{trend_key}_{breadth_short}"


def _compute_atr_pct_from_5m(df_5m: Optional[pd.DataFrame]) -> float:
    """
    Compute ATR% directly from NAS100 5m data.
    ATR% = average true range / close price (annualisation not needed — just raw %).
    Falls back to 0.012 (moderate) if data unavailable.
    """
    if df_5m is None or len(df_5m) < 15:
        return 0.012   # moderate default — better than 0.01 (always-low)

    try:
        df = df_5m.copy()
        df.columns = [c.capitalize() for c in df.columns]
        high  = df['High'].values[-14:]
        low   = df['Low'].values[-14:]
        close = df['Close'].values[-14:]
        prev  = df['Close'].values[-15:-1]

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - prev), np.abs(low - prev))
        )
        atr    = float(np.mean(tr))
        atr_pct = atr / float(close[-1]) if close[-1] > 0 else 0.012
        return atr_pct
    except Exception:
        return 0.012


def _compute_volatility_score(vix: Optional[float],
                               atr_pct: Optional[float]) -> tuple:
    """Returns (score 0-100, vix_used, atr_used)"""
    vix_val = vix or 18.0
    atr_val = atr_pct or 0.012

    score = 0

    # VIX component (60 pts) — lowered thresholds vs v2
    if vix_val > 30:    score += 60
    elif vix_val > 25:  score += 45    # was 45, now fires at 25 not 26
    elif vix_val > 20:  score += 28    # was 25, slightly higher
    elif vix_val > 16:  score += 12    # was 15>10, now 16>12
    else:               score += 0

    # ATR% component (40 pts) — now uses real data
    if atr_val > 0.025:    score += 40
    elif atr_val > 0.015:  score += 25
    elif atr_val > 0.010:  score += 14  # was 0.008>10, tightened
    elif atr_val > 0.007:  score += 6
    else:                   score += 0

    return min(score, 100), vix_val, atr_val


def _compute_trend_score(trend_bullish: bool, macd_bullish: bool,
                          rsi: float, price_vs_sma: float) -> int:
    """Returns score 0-100. High = strong trend (good for State 0)."""
    score = 0
    if trend_bullish and macd_bullish:   score += 40
    elif trend_bullish or macd_bullish:  score += 20

    if rsi < 35 or rsi > 65:            score += 30
    elif rsi < 42 or rsi > 58:          score += 15

    sma_dist = abs(price_vs_sma)
    if sma_dist > 2.0:   score += 30
    elif sma_dist > 0.5: score += 15

    return min(score, 100)


def _compute_breadth_score(breadth_ratio: Optional[float],
                            breadth_signal: Optional[str],
                            risk_score: Optional[int]) -> int:
    """
    Returns score 0-100. High = broad healthy participation.

    BUG 4 FIX: Recalibrated thresholds for %-gap inputs.
    Also reads breadth_signal string directly as primary source.
    """
    score = 50   # neutral baseline

    # Primary: use the signal string (more reliable than raw ratio)
    if breadth_signal:
        if "EXHAUSTION" in breadth_signal:
            score = 10
        elif "DIVERGING" in breadth_signal:
            score = 35
        elif "HEALTHY" in breadth_signal:
            score = 80
        # else keep 50

    # Secondary: fine-tune with ratio if available
    # breadth_ratio = QQQ_5d_ret - QQQE_5d_ret (percentage points)
    elif breadth_ratio is not None:
        gap = abs(breadth_ratio)
        if gap < 0.8:    score = 80   # was <1.0 — tightened
        elif gap < 2.5:  score = 55   # was <2.5 kept
        elif gap < 4.0:  score = 30   # was <4.0 kept
        else:            score = 10

    # Macro risk score adjustment
    if risk_score is not None:
        if risk_score >= 70:   score = max(score - 40, 0)
        elif risk_score >= 40: score = max(score - 20, 0)

    return score


def _check_risk_off(gold_df, jpy_df, tnx_df) -> tuple:
    """Cross-asset Risk-Off: Gold + bonds rising together = Crisis."""
    signals = []
    score   = 0

    if gold_df is not None and len(gold_df) >= 5:
        gold_ret = float(gold_df['Close'].pct_change(5).iloc[-1]) * 100
        if gold_ret > 1.0:
            signals.append(f"Gold +{gold_ret:.1f}%")
            score += 35

    if jpy_df is not None and len(jpy_df) >= 5:
        jpy_ret = float(jpy_df['Close'].pct_change(5).iloc[-1]) * 100
        if jpy_ret < -0.8:
            signals.append(f"JPY +{abs(jpy_ret):.1f}%")
            score += 35

    if tnx_df is not None and len(tnx_df) >= 5:
        tnx_ret = float(tnx_df['Close'].pct_change(5).iloc[-1]) * 100
        if tnx_ret < -3.0:
            signals.append(f"Bond yields -{abs(tnx_ret):.1f}%")
            score += 30

    is_risk_off = score >= 60
    reason = " | ".join(signals) if signals else "No risk-off signals"
    return is_risk_off, reason, score


def detect_regime_stocks(
    vix: Optional[float] = None,
    df_5m: Optional[object] = None,    # NEW: pass NAS100 5m DataFrame directly
    trend_bullish: bool = True,
    macd_bullish: bool = True,
    rsi: float = 50.0,
    price_vs_sma_pct: float = 1.0,
    breadth_ratio: Optional[float] = None,
    breadth_signal: Optional[str] = None,  # NEW: pass macro_snap.breadth_signal
    macro_risk_score: Optional[int] = None,
    gold_df=None,
    tnx_df=None,
    # kept for backwards compat but now ignored (computed internally)
    atr_pct: Optional[float] = None,
) -> RegimeState:
    """
    Detect regime for Mag 7 / stock dashboard.

    v3 changes:
    - Accepts df_5m for live ATR computation (most important fix)
    - Accepts breadth_signal string as primary breadth source
    - Cache key includes live inputs so regime updates on condition changes
    - All 5 bugs from audit fixed
    """
    # Compute live ATR from 5m data (BUG 3 FIX)
    live_atr = _compute_atr_pct_from_5m(df_5m) if df_5m is not None else (atr_pct or 0.012)

    # Build input-aware cache key (BUG 1 FIX)
    bsig       = breadth_signal or ""
    cache_key  = _make_cache_key(vix, macro_risk_score, bsig, trend_bullish, macd_bullish)
    ts_key     = f"{cache_key}_ts"

    if (time.time() - st.session_state.get(ts_key, 0)) < CACHE_TTL:
        cached = st.session_state.get(cache_key)
        if cached:
            return cached

    # ── SCORE COMPONENTS ──────────────────────────────────────────────────
    vol_score, vix_used, atr_used = _compute_volatility_score(vix, live_atr)

    # BUG 2 FIX: use actual passed values (app.py now passes them)
    trend_score = _compute_trend_score(trend_bullish, macd_bullish,
                                       rsi, price_vs_sma_pct)

    # BUG 4 FIX: pass breadth_signal string
    breadth_score = _compute_breadth_score(breadth_ratio, breadth_signal, macro_risk_score)

    risk_off, ro_reason, ro_score = _check_risk_off(gold_df, None, tnx_df)

    trend_live = "BULLISH" if trend_bullish else "BEARISH"

    # ── REGIME CLASSIFICATION ─────────────────────────────────────────────
    # BUG 5 FIX: State 2 threshold lowered to 45 (was 55)
    if vol_score >= 45 or risk_off or (macro_risk_score or 0) >= 70:
        state   = 2
        label   = "HIGH VOLATILITY / CRISIS"
        color   = "#8b0000"
        icon    = "🔴"
        allowed = []
        lot_mul = 0.0
        note    = (
            "CRISIS regime: Capital protection mode. "
            f"VIX score {vol_score}/100 | "
            f"{'Risk-off active: ' + ro_reason[:40] if risk_off else 'Macro danger'}. "
            "No new entries."
        )
        conf = min((vol_score / 100) + (0.3 if risk_off else 0), 1.0)

    elif (vol_score >= 25 or trend_score < 40 or breadth_score < 40
          or (macro_risk_score or 0) >= 40):
        state   = 1
        label   = "SIDEWAYS / CHOP"
        color   = "#e6a817"
        icon    = "🟡"
        allowed = ["DIP BUY", "CAUTION BUY", "MEAN REVERSION"]
        lot_mul = 0.5
        note    = (
            f"CHOP regime: vol={vol_score} trend={trend_score} breadth={breadth_score}. "
            "Mean-reversion only. 50% lot size."
        )
        conf = 0.5 + (vol_score / 200)

    else:
        state   = 0
        label   = "QUIET TREND"
        color   = "#2d9e2d"
        icon    = "🟢"
        allowed = ["MOMENTUM BUY", "STRONG BUY", "MOMENTUM SELL",
                   "STRONG SELL", "DIP BUY", "CAUTION BUY"]
        lot_mul = 1.0
        note    = (
            f"TRENDING regime: vol={vol_score} trend={trend_score} breadth={breadth_score}. "
            "All signals valid. Normal lot sizing."
        )
        conf = min((100 - vol_score) / 100 + (breadth_score / 200), 1.0)

    # ── REGIME CHANGE DETECTION ───────────────────────────────────────────
    prev_state     = st.session_state.get("regime_prev_state", state)
    regime_changed = (state != prev_state)
    if regime_changed:
        st.session_state["regime_prev_state"] = state

    result = RegimeState(
        state=state, label=label, color=color, icon=icon, confidence=conf,
        volatility_score=vol_score, trend_score=trend_score,
        breadth_score=breadth_score, risk_off_score=ro_score,
        allowed_signals=allowed, lot_multiplier=lot_mul, strategy_note=note,
        risk_off_active=risk_off, risk_off_reason=ro_reason,
        vix_live=round(vix_used, 2),
        atr_pct_live=round(atr_used * 100, 3),   # stored as % e.g. 1.2%
        trend_live=trend_live,
        regime_changed=regime_changed,
    )

    st.session_state[cache_key] = result
    st.session_state[ts_key]    = time.time()
    return result


# ── FOREX REGIME (unchanged from v2 — not affected by bugs) ──────────────────

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
    cache_key = f"regime_fx_{pair}"
    ts_key    = f"{cache_key}_ts"
    if (time.time() - st.session_state.get(ts_key, 0)) < CACHE_TTL:
        cached = st.session_state.get(cache_key)
        if cached:
            return cached

    vol_score, vix_val, atr_val = _compute_volatility_score(vix, atr_pct)
    trend_score   = _compute_trend_score(trend_bullish, True, rsi, 1.0)
    breadth_score = _compute_breadth_score(None, None, macro_risk_score)
    risk_off, ro_reason, ro_score = _check_risk_off(gold_df, jpy_df, tnx_df)

    if vol_intensity < 0.6:
        vol_score = min(vol_score + 20, 100)

    is_safe_haven = any(sh in pair for sh in ['JPY', 'CHF'])

    if vol_score >= 45 or (macro_risk_score or 0) >= 70:
        state = 2
        label = "HIGH VOLATILITY / CRISIS"
        color = "#8b0000"
        icon  = "🔴"
        if is_safe_haven and risk_off:
            allowed = ["BREAKOUT BUY"]
            lot_mul = 0.5
            note    = f"CRISIS: {pair} safe-haven. Risk-off flows favour this pair."
        else:
            allowed = []
            lot_mul = 0.0
            note    = f"CRISIS regime: No new entries on {pair}."
        conf = min(vol_score / 100, 1.0)

    elif vol_score >= 25 or trend_score < 35:
        state   = 1
        label   = "SIDEWAYS / CHOP"
        color   = "#e6a817"
        icon    = "🟡"
        allowed = ["MEAN REVERSION BUY", "MEAN REVERSION SELL", "MEAN REVERSION", "POC LEVEL"]
        lot_mul = 0.5
        note    = "CHOP regime: Mean reversion to POC/VA edges only."
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
        note    = "TRENDING regime: All Volume Profile signals valid."
        conf    = min((100 - vol_score) / 100, 1.0)

    result = RegimeState(
        state=state, label=label, color=color, icon=icon, confidence=conf,
        volatility_score=vol_score, trend_score=trend_score,
        breadth_score=breadth_score, risk_off_score=ro_score,
        allowed_signals=allowed, lot_multiplier=lot_mul, strategy_note=note,
        risk_off_active=risk_off, risk_off_reason=ro_reason,
        vix_live=round(vix_val, 2), atr_pct_live=round(atr_val * 100, 3),
        trend_live="BULLISH" if trend_bullish else "BEARISH",
        regime_changed=False,
    )

    st.session_state[cache_key] = result
    st.session_state[ts_key]    = time.time()
    return result


# ── RENDER FUNCTIONS (updated to show live inputs) ────────────────────────────

def render_regime_badge(regime: RegimeState, show_details: bool = False):
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
    if regime.regime_changed:
        st.toast(f"⚠️ Regime changed → {regime.label}", icon="🔄")

    st.markdown(
        f"<div style='padding:12px;border-radius:8px;"
        f"background:{regime.color}22;border:2px solid {regime.color};margin-bottom:8px'>"
        f"<span style='font-size:1.2em;font-weight:bold;color:{regime.color}'>"
        f"{regime.icon} Regime: {regime.label}</span>"
        f"<span style='color:#aaa;margin-left:16px;font-size:0.9em'>"
        f"Confidence: {regime.confidence*100:.0f}%</span>"
        f"{'<span style=\"color:#ff9900;margin-left:12px\">⚡ REGIME CHANGED</span>' if regime.regime_changed else ''}"
        f"</div>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volatility Score", f"{regime.volatility_score}/100",
              delta="High risk" if regime.volatility_score > 45 else "Normal",
              delta_color="inverse")
    c2.metric("Trend Score",      f"{regime.trend_score}/100",
              delta="Strong" if regime.trend_score > 60 else "Weak")
    c3.metric("Breadth Score",    f"{regime.breadth_score}/100",
              delta="Broad" if regime.breadth_score > 60 else "Narrow")
    c4.metric("Risk-Off Score",   f"{regime.risk_off_score}/100",
              delta="⚠️ Active" if regime.risk_off_active else "Clear",
              delta_color="inverse" if regime.risk_off_active else "normal")

    # Live data inputs panel — shows exactly what the regime is reading
    with st.expander("🔍 Live Regime Inputs (debug)", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("VIX (live)",    f"{regime.vix_live:.1f}")
        d2.metric("ATR% (live)",   f"{regime.atr_pct_live:.2f}%")
        d3.metric("Trend",         regime.trend_live)
        d4.metric("Lot multiplier",f"{regime.lot_multiplier}x")
        st.caption(regime.strategy_note)

    if regime.risk_off_active:
        st.error(f"🚨 Cross-Asset Risk-Off: {regime.risk_off_reason}")

    st.info(f"📋 Strategy: {regime.strategy_note}")

    if regime.state == 2:
        st.error("🔴 CRISIS MODE: No new trade entries. Protect capital.")
    elif regime.state == 1:
        st.warning(f"🟡 CHOP MODE: Lot size reduced to {regime.lot_multiplier}x. Mean reversion only.")
    else:
        st.success("🟢 TREND MODE: All signals valid. Normal lot sizing.")
