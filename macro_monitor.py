"""
macro_monitor.py
----------------
Tracks macro risk factors for Mag 7 dashboard:
  1. US 10Y Yield (^TNX) — Yield-to-Growth correlation with Nasdaq
  2. Oil/Nasdaq ratio (BZ=F vs QQQ) — margin pressure on tech
  3. Market Breadth — QQQ vs QQQE (Generals vs Soldiers)
  4. Breaking Point Risk Score — combines all three

All data fetched with timeout protection and cached in session_state.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class MacroSnapshot:
    # 10Y Yield
    yield_10y: float            # current yield %
    yield_signal: str           # "YIELD TRAP" | "ELEVATED" | "NORMAL"
    yield_color: str
    nasdaq_yield_corr: float    # 10-day rolling correlation (should be negative)

    # Oil pressure
    oil_price: float            # Brent crude USD
    oil_nasdaq_ratio: float     # QQQ/Oil ratio — falling = oil taxing tech
    oil_signal: str             # "MARGIN PRESSURE" | "WATCH" | "CLEAR"
    oil_color: str

    # Market breadth
    qqq_price: float
    qqqe_price: float
    breadth_ratio: float        # QQQ / QQQE — rising = narrow market
    breadth_signal: str         # "EXHAUSTION" | "DIVERGING" | "HEALTHY"
    breadth_color: str

    # Breaking point composite
    risk_score: int             # 0-100
    risk_level: str             # "DANGER: CRITICAL BREAKING POINT" | "WARNING: FRAGILE EQUILIBRIUM" | "CLEAR: MOMENTUM CONTINUATION"
    risk_color: str
    risk_icon: str


CACHE_TTL   = 300   # 5 min
TIMEOUT     = 10


def _fetch(func, timeout=TIMEOUT):
    result = [None]
    def t(): 
        try: result[0] = func()
        except Exception as e: print(f"[macro] {e}")
    th = threading.Thread(target=t, daemon=True)
    th.start(); th.join(timeout=timeout)
    return result[0]


def _cache_valid():
    return (time.time() - st.session_state.get("macro_ts", 0)) < CACHE_TTL


def get_macro_snapshot() -> Optional[MacroSnapshot]:
    """Main entry. Returns cached snapshot or fetches fresh data."""

    if _cache_valid() and "macro_snap" in st.session_state:
        return st.session_state["macro_snap"]

    # ── FETCH IN PARALLEL ─────────────────────────────────────────────────────
    data = {}

    def fetch_yield():
        df = yf.Ticker("^TNX").history(period="30d", interval="1d")
        data["yield"] = df if not df.empty else None

    def fetch_oil():
        df = yf.Ticker("BZ=F").history(period="5d", interval="1d")
        data["oil"] = df if not df.empty else None

    def fetch_qqq():
        df = yf.Ticker("QQQ").history(period="30d", interval="1d")
        data["qqq"] = df if not df.empty else None

    def fetch_qqqe():
        df = yf.Ticker("QQQE").history(period="30d", interval="1d")
        data["qqqe"] = df if not df.empty else None

    def fetch_ndx():
        df = yf.Ticker("^NDX").history(period="30d", interval="1d")
        data["ndx"] = df if not df.empty else None

    threads = [threading.Thread(target=f, daemon=True)
               for f in [fetch_yield, fetch_oil, fetch_qqq, fetch_qqqe, fetch_ndx]]
    for t in threads: t.start()
    for t in threads: t.join(timeout=TIMEOUT + 2)

    # ── 1. 10Y YIELD ─────────────────────────────────────────────────────────
    yield_10y      = 4.30   # fallback
    yield_signal   = "NORMAL"
    yield_color    = "#2d9e2d"
    nasdaq_corr    = -0.5

    df_yield = data.get("yield")
    df_ndx   = data.get("ndx")

    if df_yield is not None and len(df_yield) >= 2:
        yield_10y = float(df_yield["Close"].iloc[-1])

        # 10-day rolling correlation between TNX and NDX
        if df_ndx is not None and len(df_ndx) >= 10 and len(df_yield) >= 10:
            combined = pd.DataFrame({
                "yield": df_yield["Close"].tail(20),
                "ndx":   df_ndx["Close"].tail(20),
            }).dropna()
            if len(combined) >= 10:
                nasdaq_corr = float(combined["yield"].rolling(10).corr(combined["ndx"]).iloc[-1])

        # Yield Trap: yield > 4.45% AND correlation strongly negative
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
    oil_price       = 75.0
    oil_signal      = "CLEAR"
    oil_color       = "#2d9e2d"
    oil_nasdaq_ratio= 5.0

    df_oil = data.get("oil")
    df_qqq = data.get("qqq")

    if df_oil is not None and len(df_oil) >= 2:
        oil_price = float(df_oil["Close"].iloc[-1])

    if df_qqq is not None and len(df_qqq) >= 2:
        qqq_price = float(df_qqq["Close"].iloc[-1])
        oil_nasdaq_ratio = qqq_price / oil_price if oil_price > 0 else 5.0

        # Check if ratio is falling (oil rising faster than QQQ)
        if len(df_qqq) >= 5 and df_oil is not None and len(df_oil) >= 5:
            ratio_series = df_qqq["Close"].tail(10) / df_oil["Close"].tail(10)
            ratio_pct_change = float((ratio_series.iloc[-1] - ratio_series.iloc[0]) / ratio_series.iloc[0] * 100)
        else:
            ratio_pct_change = 0

        if oil_price > 110:
            oil_signal = "🔴 MARGIN PRESSURE"
            oil_color  = "#8b0000"
        elif oil_price > 95 or ratio_pct_change < -5:
            oil_signal = "🟡 WATCH"
            oil_color  = "#e6a817"
        else:
            oil_signal = "🟢 CLEAR"
            oil_color  = "#2d9e2d"
    else:
        qqq_price = 460.0

    # ── 3. MARKET BREADTH (QQQ vs QQQE) ──────────────────────────────────────
    qqqe_price     = 80.0
    breadth_ratio  = 1.0
    breadth_signal = "HEALTHY"
    breadth_color  = "#2d9e2d"

    df_qqqe = data.get("qqqe")

    if df_qqqe is not None and len(df_qqqe) >= 2 and df_qqq is not None and len(df_qqq) >= 5:
        qqqe_price = float(df_qqqe["Close"].iloc[-1])

        # Normalise both to same start point (5 days ago) for comparison
        qqq_ret  = float(df_qqq["Close"].pct_change(5).iloc[-1]) * 100
        qqqe_ret = float(df_qqqe["Close"].pct_change(5).iloc[-1]) * 100

        breadth_ratio = qqq_ret - qqqe_ret  # positive = QQQ outperforming = narrow market

        # Generals ahead while Soldiers fall = exhaustion
        if qqq_ret > 0 and qqqe_ret < -1:
            breadth_signal = "🔴 EXHAUSTION — Generals vs Soldiers"
            breadth_color  = "#8b0000"
        elif breadth_ratio > 3:
            breadth_signal = "🟡 DIVERGING — Market narrowing"
            breadth_color  = "#e6a817"
        else:
            breadth_signal = "🟢 HEALTHY — Broad participation"
            breadth_color  = "#2d9e2d"

    # ── 4. BREAKING POINT RISK SCORE ─────────────────────────────────────────
    risk_score = 0

    # Yield risk (40 pts max)
    if yield_10y > 4.50 and nasdaq_corr < -0.8:
        risk_score += 40
    elif yield_10y > 4.45:
        risk_score += 20

    # Oil risk (30 pts max)
    if oil_price > 110:
        risk_score += 30
    elif oil_price > 95:
        risk_score += 15

    # RSI/breadth risk (30 pts max) — use breadth signal as proxy
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

    st.session_state["macro_snap"] = snap
    st.session_state["macro_ts"]   = time.time()
    return snap


def render_macro_panel():
    """Render the full macro risk panel in the Mag7 dashboard."""
    st.subheader("🌐 Macro Risk Monitor")

    snap = get_macro_snapshot()
    if snap is None:
        st.warning("Macro data unavailable — check network.")
        return

    # ── BREAKING POINT SCORE ─────────────────────────────────────────────────
    st.markdown(
        f"<div style='padding:12px;border-radius:8px;background:{snap.risk_color}22;"
        f"border:2px solid {snap.risk_color};margin-bottom:12px'>"
        f"<span style='font-size:1.3em;font-weight:bold;color:{snap.risk_color}'>"
        f"{snap.risk_icon} {snap.risk_level}</span><br>"
        f"<span style='color:#aaa'>Risk Score: {snap.risk_score}/100</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.progress(snap.risk_score / 100)

    # ── THREE COLUMNS ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**📈 US 10Y Yield**")
        st.markdown(
            f"<span style='font-size:1.4em;font-weight:bold;color:{snap.yield_color}'>"
            f"{snap.yield_10y:.2f}%</span>",
            unsafe_allow_html=True
        )
        st.caption(snap.yield_signal)
        st.caption(f"Nasdaq corr: {snap.nasdaq_yield_corr:.2f}")
        if "TRAP" in snap.yield_signal:
            st.error("Yields choking growth — reduce long exposure")

    with c2:
        st.markdown("**🛢️ Brent Crude / Oil Pressure**")
        st.markdown(
            f"<span style='font-size:1.4em;font-weight:bold;color:{snap.oil_color}'>"
            f"${snap.oil_price:.1f}</span>",
            unsafe_allow_html=True
        )
        st.caption(snap.oil_signal)
        st.caption(f"QQQ/Oil ratio: {snap.oil_nasdaq_ratio:.1f}")
        if "MARGIN" in snap.oil_signal:
            st.error("Energy costs taxing AI margins — AMZN/META at risk")

    with c3:
        st.markdown("**📊 Market Breadth (QQQ vs QQQE)**")
        st.markdown(
            f"<span style='font-size:1.4em;font-weight:bold;color:{snap.breadth_color}'>"
            f"{'↑' if snap.breadth_ratio > 0 else '↓'}{abs(snap.breadth_ratio):.1f}%</span>",
            unsafe_allow_html=True
        )
        st.caption(snap.breadth_signal)
        st.caption("QQQ 5d vs QQQE 5d performance gap")
        if "EXHAUSTION" in snap.breadth_signal:
            st.error("Generals only — classic exhaustion before reversal")

    st.divider()
