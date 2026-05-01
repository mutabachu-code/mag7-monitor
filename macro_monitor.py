"""
macro_monitor.py  —  v2
------------------------
Reads macro data from data_fetcher session_state cache.
No direct yfinance calls — zero extra API load.

Calculates:
  1. Yield-to-Growth signal  (^TNX vs Nasdaq correlation)
  2. Oil Margin Pressure     (BZ=F vs QQQ ratio)
  3. Market Breadth          (QQQ vs QQQE — Generals vs Soldiers)
  4. Breaking Point Score    (composite 0-100)
"""

import pandas as pd
import numpy as np
import streamlit as st
import time
from dataclasses import dataclass
from typing import Optional
from data_fetcher import (
    get_yield_10y, get_oil_price, get_qqqe_df,
    get_qqq_1d, get_macro_df
)


@dataclass
class MacroSnapshot:
    yield_10y: float
    yield_signal: str
    yield_color: str
    nasdaq_yield_corr: float
    oil_price: float
    oil_nasdaq_ratio: float
    oil_signal: str
    oil_color: str
    qqq_price: float
    qqqe_price: float
    breadth_ratio: float
    breadth_signal: str
    breadth_color: str
    risk_score: int
    risk_level: str
    risk_color: str
    risk_icon: str


CACHE_TTL = 65   # match data_fetcher TTL


def _macro_cache_valid() -> bool:
    return (time.time() - st.session_state.get("macro_snap_ts", 0)) < CACHE_TTL


def get_macro_snapshot() -> Optional[MacroSnapshot]:
    if _macro_cache_valid() and "macro_snap" in st.session_state:
        return st.session_state["macro_snap"]

    # ── READ FROM data_fetcher CACHE ─────────────────────────────────────────
    yield_10y   = get_yield_10y() or 4.30
    oil_price   = get_oil_price() or 75.0
    df_qqq      = get_qqq_1d()
    df_qqqe     = get_qqqe_df()
    df_tnx      = get_macro_df("tnx")
    df_ndx      = get_macro_df("ndx")

    # ── 1. 10Y YIELD ─────────────────────────────────────────────────────────
    nasdaq_corr = -0.5

    if df_tnx is not None and df_ndx is not None and len(df_tnx) >= 10 and len(df_ndx) >= 10:
        combined = pd.DataFrame({
            "yield": df_tnx["Close"].tail(20),
            "ndx":   df_ndx["Close"].tail(20),
        }).dropna()
        if len(combined) >= 10:
            nasdaq_corr = float(combined["yield"].rolling(10).corr(combined["ndx"]).iloc[-1])

    if yield_10y > 4.50 and nasdaq_corr < -0.8:
        yield_signal = "⚠️ YIELD TRAP"
        yield_color  = "#8b0000"
    elif yield_10y > 4.45:
        yield_signal = "🟡 ELEVATED"
        yield_color  = "#e6a817"
    else:
        yield_signal = "🟢 NORMAL"
        yield_color  = "#2d9e2d"

    # ── 2. OIL / MARGIN PRESSURE ──────────────────────────────────────────────
    qqq_price        = 460.0
    oil_nasdaq_ratio = 6.0

    if df_qqq is not None and not df_qqq.empty:
        qqq_price = float(df_qqq['Close'].iloc[-1])
        oil_nasdaq_ratio = qqq_price / oil_price if oil_price > 0 else 6.0

        # Ratio trend: falling = oil outpacing QQQ
        if len(df_qqq) >= 5:
            qqq_5d  = float(df_qqq['Close'].pct_change(5).iloc[-1]) * 100
            # Approximate oil 5d change
            oil_5d  = 0.0
            if df_tnx is not None and len(df_tnx) >= 5:   # reuse as proxy length check
                pass   # oil daily already has latest price; simple threshold sufficient
        else:
            qqq_5d = 0.0

    if oil_price > 110:
        oil_signal = "🔴 MARGIN PRESSURE"
        oil_color  = "#8b0000"
    elif oil_price > 95:
        oil_signal = "🟡 WATCH"
        oil_color  = "#e6a817"
    else:
        oil_signal = "🟢 CLEAR"
        oil_color  = "#2d9e2d"

    # ── 3. MARKET BREADTH ─────────────────────────────────────────────────────
    qqqe_price     = 80.0
    breadth_ratio  = 0.0
    breadth_signal = "🟢 HEALTHY — Broad participation"
    breadth_color  = "#2d9e2d"

    if df_qqqe is not None and not df_qqqe.empty and df_qqq is not None and not df_qqq.empty:
        qqqe_price = float(df_qqqe['Close'].iloc[-1])

        if len(df_qqq) >= 6 and len(df_qqqe) >= 6:
            qqq_ret  = float(df_qqq['Close'].pct_change(5).iloc[-1]) * 100
            qqqe_ret = float(df_qqqe['Close'].pct_change(5).iloc[-1]) * 100
            breadth_ratio = qqq_ret - qqqe_ret

            if qqq_ret > 0 and qqqe_ret < -1:
                breadth_signal = "🔴 EXHAUSTION — Generals vs Soldiers"
                breadth_color  = "#8b0000"
            elif breadth_ratio > 3:
                breadth_signal = "🟡 DIVERGING — Market narrowing"
                breadth_color  = "#e6a817"
            else:
                breadth_signal = "🟢 HEALTHY — Broad participation"
                breadth_color  = "#2d9e2d"

    # ── 4. BREAKING POINT SCORE ───────────────────────────────────────────────
    risk_score = 0

    if yield_10y > 4.50 and nasdaq_corr < -0.8:
        risk_score += 40
    elif yield_10y > 4.45:
        risk_score += 20

    if oil_price > 110:
        risk_score += 30
    elif oil_price > 95:
        risk_score += 15

    if "EXHAUSTION" in breadth_signal:
        risk_score += 30
    elif "DIVERGING" in breadth_signal:
        risk_score += 15

    if risk_score >= 70:
        risk_level = "DANGER: CRITICAL BREAKING POINT"
        risk_color = "#8b0000"
        risk_icon  = "🔴"
    elif risk_score >= 40:
        risk_level = "WARNING: FRAGILE EQUILIBRIUM"
        risk_color = "#e6a817"
        risk_icon  = "🟡"
    else:
        risk_level = "CLEAR: MOMENTUM CONTINUATION"
        risk_color = "#2d9e2d"
        risk_icon  = "🟢"

    snap = MacroSnapshot(
        yield_10y=yield_10y, yield_signal=yield_signal,
        yield_color=yield_color, nasdaq_yield_corr=nasdaq_corr,
        oil_price=oil_price, oil_nasdaq_ratio=oil_nasdaq_ratio,
        oil_signal=oil_signal, oil_color=oil_color,
        qqq_price=qqq_price, qqqe_price=qqqe_price,
        breadth_ratio=breadth_ratio, breadth_signal=breadth_signal,
        breadth_color=breadth_color, risk_score=risk_score,
        risk_level=risk_level, risk_color=risk_color, risk_icon=risk_icon,
    )

    st.session_state["macro_snap"]    = snap
    st.session_state["macro_snap_ts"] = time.time()
    return snap


def render_macro_panel():
    st.subheader("🌐 Macro Risk Monitor")

    snap = get_macro_snapshot()
    if snap is None:
        st.warning("Macro data unavailable.")
        return

    # Breaking Point Score banner
    st.markdown(
        f"<div style='padding:12px;border-radius:8px;"
        f"background:{snap.risk_color}22;border:2px solid {snap.risk_color};margin-bottom:8px'>"
        f"<span style='font-size:1.2em;font-weight:bold;color:{snap.risk_color}'>"
        f"{snap.risk_icon} {snap.risk_level}</span>"
        f"<span style='color:#aaa;margin-left:16px'>Score: {snap.risk_score}/100</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.progress(snap.risk_score / 100)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**📈 US 10Y Yield**")
        st.markdown(
            f"<span style='font-size:1.4em;font-weight:bold;color:{snap.yield_color}'>"
            f"{snap.yield_10y:.2f}%</span>",
            unsafe_allow_html=True
        )
        st.caption(snap.yield_signal)
        st.caption(f"Nasdaq corr (10d): {snap.nasdaq_yield_corr:.2f}")
        if "TRAP" in snap.yield_signal:
            st.error("Yields choking growth — reduce long exposure")

    with c2:
        st.markdown("**🛢️ Brent Crude**")
        st.markdown(
            f"<span style='font-size:1.4em;font-weight:bold;color:{snap.oil_color}'>"
            f"${snap.oil_price:.1f}</span>",
            unsafe_allow_html=True
        )
        st.caption(snap.oil_signal)
        st.caption(f"QQQ/Oil ratio: {snap.oil_nasdaq_ratio:.1f}x")
        if "MARGIN" in snap.oil_signal:
            st.error("Energy costs taxing AI margins — AMZN/META at risk")

    with c3:
        st.markdown("**📊 Breadth (QQQ vs QQQE)**")
        st.markdown(
            f"<span style='font-size:1.4em;font-weight:bold;color:{snap.breadth_color}'>"
            f"{'↑' if snap.breadth_ratio >= 0 else '↓'}{abs(snap.breadth_ratio):.1f}%</span>",
            unsafe_allow_html=True
        )
        st.caption(snap.breadth_signal)
        st.caption("5d QQQ vs QQQE performance gap")
        if "EXHAUSTION" in snap.breadth_signal:
            st.error("Generals only — classic exhaustion before reversal")

    st.divider()
