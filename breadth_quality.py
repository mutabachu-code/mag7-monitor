"""
breadth_quality.py
------------------
Smart Breadth Engine upgrade per screenshot recommendations.

Current breadth (QQQ vs QQQE ratio) is decent but misses breadth QUALITY.
This module adds:

  1. Semiconductor Leadership Tracker
     - SOX (SOXX ETF) vs QQQ relative strength
     - Semis lead the Nasdaq — if semis are strong, rally is real
     - Semis weakening = distribution warning even if QQQ is up

  2. Mega-Cap Participation Tracker
     - Are all Mag 7 contributing or just 1-2 stocks carrying the index?
     - Breadth = count of Mag 7 stocks above their own 20-day SMA

  3. Momentum Decay Detector
     - RSI divergence: QQQ making new high but RSI lower = hidden weakness
     - Uses 5-day and 20-day RSI comparison

  4. Leadership Quality Score (0-100)
     - Composite: semi leadership + mag7 participation + no RSI divergence
     - HIGH (>70): Rally has broad, sector-confirmed support — trust signals
     - MEDIUM (40-70): Selective leadership — reduce lot size 20%
     - LOW (<40): Narrow / deteriorating breadth — treat all BUY signals as CAUTION

All data from yfinance daily (lightweight). Cached 10 minutes.
No interference with existing macro_monitor / regime_detector.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
from dataclasses import dataclass, field
from typing import Optional, List

from data_fetcher import get_1d, MAG7


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class SemiLeadership:
    soxx_5d_ret: float        # SOXX 5-day return %
    qqq_5d_ret: float         # QQQ 5-day return %
    relative_strength: float  # soxx_ret - qqq_ret
    signal: str               # "LEADING" | "INLINE" | "LAGGING"
    signal_color: str
    interpretation: str


@dataclass
class Mag7Participation:
    above_20sma_count: int    # number of Mag 7 stocks above their 20-day SMA
    total: int                # = 7
    participation_pct: float  # above_20sma_count / 7 * 100
    tickers_above: List[str]
    tickers_below: List[str]
    signal: str               # "BROAD" | "MODERATE" | "NARROW"
    signal_color: str


@dataclass
class MomentumDecay:
    qqq_price_5d_chg: float     # QQQ 5-day price change %
    qqq_rsi_5d_chg: float       # RSI change over same 5 days
    divergence_detected: bool   # price up but RSI down (or vice versa) = warning
    signal: str
    signal_color: str


@dataclass
class BreadthQuality:
    semi_leadership: Optional[SemiLeadership]
    mag7_participation: Optional[Mag7Participation]
    momentum_decay: Optional[MomentumDecay]
    quality_score: int          # 0-100 composite
    quality_label: str          # "HIGH" | "MEDIUM" | "LOW"
    quality_color: str
    lot_adjustment: float       # 1.0 | 0.8 | 0.6
    summary: str


# ── CACHE ─────────────────────────────────────────────────────────────────────

CACHE_TTL = 600   # 10 minutes


def _cache_valid() -> bool:
    return (time.time() - st.session_state.get("breadth_quality_ts", 0)) < CACHE_TTL


# ── SEMI LEADERSHIP ───────────────────────────────────────────────────────────

def _get_semi_leadership(qqq_1d: pd.DataFrame) -> Optional[SemiLeadership]:
    """Compare SOXX (semis ETF) vs QQQ 5-day performance."""
    try:
        soxx = yf.Ticker("SOXX").history(period="30d", interval="1d").ffill().bfill()
        if soxx.empty or qqq_1d is None or len(qqq_1d) < 6:
            return None

        soxx_ret = float(soxx['Close'].pct_change(5).iloc[-1]) * 100
        qqq_ret  = float(qqq_1d['Close'].pct_change(5).iloc[-1]) * 100
        rel_str  = round(soxx_ret - qqq_ret, 2)

        if rel_str > 1.5:
            signal    = "LEADING"
            color     = "#2d9e2d"
            interpret = "Semis outperforming QQQ — rally has sector confirmation. High-confidence BUY."
        elif rel_str > -1.5:
            signal    = "INLINE"
            color     = "#e6a817"
            interpret = "Semis in line with QQQ — neutral breadth confirmation."
        else:
            signal    = "LAGGING"
            color     = "#c9302c"
            interpret = "Semis underperforming QQQ — potential distribution. Treat BUY signals with caution."

        return SemiLeadership(
            soxx_5d_ret=round(soxx_ret, 2),
            qqq_5d_ret=round(qqq_ret, 2),
            relative_strength=rel_str,
            signal=signal,
            signal_color=color,
            interpretation=interpret,
        )
    except Exception as e:
        print(f"[breadth_quality] Semi leadership error: {e}")
        return None


# ── MAG 7 PARTICIPATION ───────────────────────────────────────────────────────

def _get_mag7_participation() -> Optional[Mag7Participation]:
    """Count Mag 7 stocks above their 20-day SMA from pre-fetched data."""
    try:
        above, below = [], []
        for ticker in MAG7:
            df = get_1d(ticker)
            if df is None or len(df) < 21:
                continue
            price   = float(df['Close'].iloc[-1])
            sma20   = float(df['Close'].rolling(20).mean().iloc[-1])
            if price > sma20:
                above.append(ticker)
            else:
                below.append(ticker)

        total = len(above) + len(below)
        if total == 0:
            return None

        pct = len(above) / total * 100

        if pct >= 70:
            signal = "BROAD"
            color  = "#2d9e2d"
        elif pct >= 43:
            signal = "MODERATE"
            color  = "#e6a817"
        else:
            signal = "NARROW"
            color  = "#c9302c"

        return Mag7Participation(
            above_20sma_count=len(above),
            total=total,
            participation_pct=round(pct, 1),
            tickers_above=above,
            tickers_below=below,
            signal=signal,
            signal_color=color,
        )
    except Exception as e:
        print(f"[breadth_quality] Mag7 participation error: {e}")
        return None


# ── MOMENTUM DECAY ────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def _get_momentum_decay(qqq_1d: pd.DataFrame) -> Optional[MomentumDecay]:
    """Detect RSI divergence: price higher but RSI lower (hidden weakness)."""
    try:
        if qqq_1d is None or len(qqq_1d) < 25:
            return None

        rsi_series = _rsi(qqq_1d['Close'])
        price_5d   = float(qqq_1d['Close'].pct_change(5).iloc[-1]) * 100
        rsi_now    = float(rsi_series.iloc[-1])
        rsi_5d_ago = float(rsi_series.iloc[-6]) if len(rsi_series) >= 6 else rsi_now
        rsi_chg    = rsi_now - rsi_5d_ago

        # Divergence: price up but RSI down (bearish divergence)
        #          or price down but RSI up (bullish divergence)
        divergence = (price_5d > 0.5 and rsi_chg < -3) or (price_5d < -0.5 and rsi_chg > 3)

        if divergence:
            if price_5d > 0:
                signal = "⚠️ Bearish divergence — price rising but momentum fading"
                color  = "#c9302c"
            else:
                signal = "🔍 Bullish divergence — price falling but momentum building"
                color  = "#2d9e2d"
        elif abs(price_5d) < 0.3:
            signal = "🟡 Flat momentum — no strong divergence signal"
            color  = "#e6a817"
        else:
            signal = "🟢 Price and momentum aligned — no divergence"
            color  = "#2d9e2d"

        return MomentumDecay(
            qqq_price_5d_chg=round(price_5d, 2),
            qqq_rsi_5d_chg=round(rsi_chg, 2),
            divergence_detected=divergence,
            signal=signal,
            signal_color=color,
        )
    except Exception as e:
        print(f"[breadth_quality] Momentum decay error: {e}")
        return None


# ── COMPOSITE QUALITY SCORE ───────────────────────────────────────────────────

def get_breadth_quality() -> Optional[BreadthQuality]:
    """
    Compute full breadth quality. Uses pre-fetched data_fetcher cache where possible.
    Only SOXX requires a new fetch (not in existing data layer).
    """
    if _cache_valid() and "breadth_quality" in st.session_state:
        return st.session_state["breadth_quality"]

    from data_fetcher import get_1d as _get_1d, NAS100_LABEL
    qqq_1d = _get_1d(NAS100_LABEL)

    semi   = _get_semi_leadership(qqq_1d)
    mag7   = _get_mag7_participation()
    decay  = _get_momentum_decay(qqq_1d)

    # Score components (each 0-40 / 0-40 / 0-20)
    score = 50   # neutral baseline

    if semi:
        if semi.signal == "LEADING":   score += 25
        elif semi.signal == "INLINE":  score += 10
        else:                          score -= 20

    if mag7:
        if mag7.signal == "BROAD":       score += 25
        elif mag7.signal == "MODERATE":  score += 10
        else:                            score -= 15

    if decay:
        if decay.divergence_detected:    score -= 15
        else:                            score += 5

    score = max(0, min(100, score))

    if score >= 70:
        label       = "HIGH"
        color       = "#2d9e2d"
        lot_adj     = 1.0
        summary     = ("Broad, sector-confirmed rally. Semiconductor leadership + Mag 7 participation strong. "
                       "Trust BUY signals — full lot size.")
    elif score >= 40:
        label       = "MEDIUM"
        color       = "#e6a817"
        lot_adj     = 0.8
        summary     = ("Selective leadership. Some breadth concerns. "
                       "Reduce lot size 20%. Prefer STRONG signals only.")
    else:
        label       = "LOW"
        color       = "#c9302c"
        lot_adj     = 0.6
        summary     = ("Narrow / deteriorating breadth. Rally may be unsustainable. "
                       "Treat all BUY signals as CAUTION. Reduce lot size 40%.")

    result = BreadthQuality(
        semi_leadership=semi,
        mag7_participation=mag7,
        momentum_decay=decay,
        quality_score=score,
        quality_label=label,
        quality_color=color,
        lot_adjustment=lot_adj,
        summary=summary,
    )

    st.session_state["breadth_quality"]    = result
    st.session_state["breadth_quality_ts"] = time.time()
    return result


# ── RENDER ────────────────────────────────────────────────────────────────────

def render_breadth_quality_panel(bq: Optional[BreadthQuality]):
    """Render the smart breadth quality panel in the dashboard."""
    st.subheader("🧠 Smart Breadth Quality Engine")

    if bq is None:
        st.warning("Breadth quality data unavailable.")
        return

    # Quality score banner
    st.markdown(
        f"<div style='padding:10px;border-radius:8px;"
        f"background:{bq.quality_color}22;border:2px solid {bq.quality_color};margin-bottom:8px'>"
        f"<span style='font-size:1.1em;font-weight:bold;color:{bq.quality_color}'>"
        f"Breadth Quality: {bq.quality_label} ({bq.quality_score}/100)"
        f"</span><br>"
        f"<span style='color:#ccc;font-size:0.9em'>{bq.summary}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.progress(bq.quality_score / 100)

    c1, c2, c3 = st.columns(3)

    # Semiconductor leadership
    with c1:
        st.markdown("**💾 Semiconductor Leadership**")
        if bq.semi_leadership:
            sl = bq.semi_leadership
            st.markdown(
                f"<span style='color:{sl.signal_color};font-weight:bold'>{sl.signal}</span>",
                unsafe_allow_html=True
            )
            c1a, c1b = st.columns(2)
            c1a.metric("SOXX 5d", f"{sl.soxx_5d_ret:+.1f}%")
            c1b.metric("QQQ 5d",  f"{sl.qqq_5d_ret:+.1f}%")
            st.caption(f"Rel. strength: {sl.relative_strength:+.1f}%")
            st.caption(sl.interpretation)
        else:
            st.caption("SOXX data unavailable")

    # Mag 7 participation
    with c2:
        st.markdown("**🏆 Mag 7 Participation**")
        if bq.mag7_participation:
            mp = bq.mag7_participation
            st.markdown(
                f"<span style='color:{mp.signal_color};font-weight:bold'>"
                f"{mp.signal} — {mp.above_20sma_count}/{mp.total} above 20-SMA</span>",
                unsafe_allow_html=True
            )
            st.progress(mp.participation_pct / 100,
                        text=f"{mp.participation_pct:.0f}% of Mag 7 above 20-SMA")
            if mp.tickers_above:
                st.caption(f"✅ Above: {', '.join(mp.tickers_above)}")
            if mp.tickers_below:
                st.caption(f"❌ Below: {', '.join(mp.tickers_below)}")
        else:
            st.caption("Mag 7 participation data unavailable")

    # Momentum decay
    with c3:
        st.markdown("**📉 Momentum Decay Check**")
        if bq.momentum_decay:
            md = bq.momentum_decay
            st.markdown(
                f"<span style='color:{md.signal_color}'>{md.signal}</span>",
                unsafe_allow_html=True
            )
            c3a, c3b = st.columns(2)
            c3a.metric("Price 5d", f"{md.qqq_price_5d_chg:+.1f}%")
            c3b.metric("RSI Δ 5d", f"{md.qqq_rsi_5d_chg:+.1f}")
            if md.divergence_detected:
                st.error("⚠️ RSI divergence active")
        else:
            st.caption("Momentum data unavailable")

    if bq.lot_adjustment < 1.0:
        st.warning(
            f"⚠️ Breadth adjustment: Lot size reduced to {bq.lot_adjustment}x "
            f"({bq.quality_label} breadth quality)"
        )

    st.divider()
