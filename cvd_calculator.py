"""
cvd_calculator.py
-----------------
Cumulative Volume Delta (CVD) — order flow tracking.

CVD measures the NET buying vs selling pressure by estimating
which side of the market is more aggressive:

  Buying pressure  = volume on UP candles (close > open)
  Selling pressure = volume on DOWN candles (close < open)
  Delta per candle = buy_vol - sell_vol
  CVD              = cumulative sum of deltas

Interpretation:
  CVD rising + price rising   = CONFIRMED BULLISH (buyers in control)
  CVD falling + price rising  = DIVERGENCE (hidden selling — rally suspect)
  CVD falling + price falling = CONFIRMED BEARISH (sellers in control)
  CVD rising + price falling  = DIVERGENCE (hidden buying — drop may reverse)

For Forex: uses 15m or 1H data (tick volume as proxy — actual volume unavailable)
For NAS100: uses 5m QQQ volume (real exchange volume)
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class CVDResult:
    # Current state
    cvd_current: float          # latest CVD value
    cvd_change_pct: float       # CVD change over lookback period (%)
    delta_last: float           # delta of last candle
    buy_volume: float           # session buy volume
    sell_volume: float          # session sell volume
    buy_sell_ratio: float       # buy/sell ratio (>1 = buyers winning)

    # Signal
    signal: str                 # "BULLISH_CONFIRM" | "BEARISH_CONFIRM" | "BULL_DIVERGENCE" | "BEAR_DIVERGENCE" | "NEUTRAL"
    signal_color: str
    signal_icon: str
    description: str

    # Trend
    cvd_trending_up: bool       # CVD making higher highs
    divergence: bool            # price and CVD disagree
    divergence_type: str        # "HIDDEN_SELLING" | "HIDDEN_BUYING" | "NONE"

    # Values for chart display
    cvd_series: list            # last 20 CVD values for mini sparkline


def calculate_cvd(df: pd.DataFrame, lookback: int = 20) -> Optional[CVDResult]:
    """
    Calculate CVD from OHLCV DataFrame.
    Works with any timeframe — 5m, 15m, 1H.
    Uses tick volume (proxy) for forex since real volume unavailable.
    """
    if df is None or len(df) < 5:
        return None

    try:
        df = df.copy()
        df.columns = [c.capitalize() for c in df.columns]

        # Ensure we have required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            return None

        # ── DELTA CALCULATION ─────────────────────────────────────────────────
        # Method: estimate buy/sell volume per candle
        # Bullish candle → most volume was buying
        # Bearish candle → most volume was selling
        # Doji → split 50/50

        candle_range = df['High'] - df['Low']
        candle_range = candle_range.replace(0, 0.00001)  # avoid division by zero

        # Proportion of candle that was up move (close vs open vs wick)
        up_move   = (df['Close'] - df['Low']) / candle_range
        down_move = (df['High'] - df['Close']) / candle_range

        buy_vol  = df['Volume'] * up_move
        sell_vol = df['Volume'] * down_move
        delta    = buy_vol - sell_vol

        # Cumulative delta
        cvd = delta.cumsum()

        # ── USE LAST N CANDLES ────────────────────────────────────────────────
        recent_cvd   = cvd.tail(lookback)
        recent_price = df['Close'].tail(lookback)
        recent_delta = delta.tail(lookback)
        recent_buy   = buy_vol.tail(lookback)
        recent_sell  = sell_vol.tail(lookback)

        cvd_current  = float(recent_cvd.iloc[-1])
        cvd_start    = float(recent_cvd.iloc[0])
        cvd_change   = ((cvd_current - cvd_start) / abs(cvd_start) * 100
                        if cvd_start != 0 else 0)

        price_current = float(recent_price.iloc[-1])
        price_start   = float(recent_price.iloc[0])
        price_up      = price_current > price_start

        # ── TREND DETECTION ───────────────────────────────────────────────────
        # Is CVD making higher highs?
        cvd_highs = [float(recent_cvd.iloc[max(0,i-3):i+1].max())
                     for i in range(3, len(recent_cvd))]
        cvd_trending_up = (len(cvd_highs) >= 2 and
                           cvd_highs[-1] > cvd_highs[0]) if cvd_highs else False

        # ── DIVERGENCE DETECTION ──────────────────────────────────────────────
        cvd_up    = cvd_change > 2   # CVD rising significantly
        cvd_down  = cvd_change < -2  # CVD falling significantly

        divergence      = False
        divergence_type = "NONE"

        if price_up and cvd_down:
            divergence      = True
            divergence_type = "HIDDEN_SELLING"   # price up but CVD falling = weak rally
        elif not price_up and cvd_up:
            divergence      = True
            divergence_type = "HIDDEN_BUYING"    # price down but CVD rising = potential reversal

        # ── SIGNAL CLASSIFICATION ─────────────────────────────────────────────
        buy_total  = float(recent_buy.sum())
        sell_total = float(recent_sell.sum())
        bs_ratio   = buy_total / sell_total if sell_total > 0 else 1.0

        if divergence_type == "HIDDEN_SELLING":
            signal      = "BEAR_DIVERGENCE"
            color       = "#e6a817"
            icon        = "⚠️"
            description = (f"HIDDEN SELLING: Price rising but CVD falling ({cvd_change:+.1f}%). "
                           f"Institutional distribution detected — rally may be exhausted.")
        elif divergence_type == "HIDDEN_BUYING":
            signal      = "BULL_DIVERGENCE"
            color       = "#3399ff"
            icon        = "💡"
            description = (f"HIDDEN BUYING: Price falling but CVD rising ({cvd_change:+.1f}%). "
                           f"Smart money accumulating — potential reversal setup.")
        elif cvd_up and price_up and bs_ratio > 1.2:
            signal      = "BULLISH_CONFIRM"
            color       = "#2d9e2d"
            icon        = "✅"
            description = (f"CONFIRMED BULLISH: CVD +{cvd_change:.1f}% with price rising. "
                           f"Buy/Sell ratio {bs_ratio:.2f} — buyers clearly in control.")
        elif cvd_down and not price_up and bs_ratio < 0.8:
            signal      = "BEARISH_CONFIRM"
            color       = "#c9302c"
            icon        = "✅"
            description = (f"CONFIRMED BEARISH: CVD {cvd_change:.1f}% with price falling. "
                           f"Buy/Sell ratio {bs_ratio:.2f} — sellers clearly in control.")
        else:
            signal      = "NEUTRAL"
            color       = "#888888"
            icon        = "➡️"
            description = (f"NEUTRAL order flow. CVD change: {cvd_change:+.1f}%. "
                           f"Buy/Sell ratio {bs_ratio:.2f} — no clear institutional bias.")

        # CVD series for sparkline (normalised to % for display)
        cvd_list = recent_cvd.tolist()

        return CVDResult(
            cvd_current=round(cvd_current, 2),
            cvd_change_pct=round(cvd_change, 2),
            delta_last=round(float(recent_delta.iloc[-1]), 2),
            buy_volume=round(buy_total, 0),
            sell_volume=round(sell_total, 0),
            buy_sell_ratio=round(bs_ratio, 3),
            signal=signal,
            signal_color=color,
            signal_icon=icon,
            description=description,
            cvd_trending_up=cvd_trending_up,
            divergence=divergence,
            divergence_type=divergence_type,
            cvd_series=cvd_list,
        )

    except Exception as e:
        print(f"[cvd_calculator] Error: {e}")
        return None


def render_cvd_badge(cvd: CVDResult):
    """Compact CVD badge for use inside pair cards."""
    import streamlit as st
    if cvd is None:
        st.caption("CVD: unavailable")
        return

    st.markdown(
        f"<div style='padding:4px 8px;border-radius:4px;"
        f"background:{cvd.signal_color}22;border:1px solid {cvd.signal_color};"
        f"margin:2px 0;font-size:0.85em'>"
        f"<b style='color:{cvd.signal_color}'>{cvd.signal_icon} CVD: "
        f"{cvd.signal.replace('_',' ')}</b> · "
        f"B/S {cvd.buy_sell_ratio:.2f} · Δ {cvd.cvd_change_pct:+.1f}%"
        f"</div>",
        unsafe_allow_html=True
    )
    if cvd.divergence:
        import streamlit as st
        st.caption(f"⚠️ {cvd.description}")


def render_cvd_panel(cvd: CVDResult, label: str = ""):
    """Full CVD panel for scalping expander."""
    import streamlit as st
    if cvd is None:
        st.caption("CVD data unavailable")
        return

    st.markdown(
        f"<div style='padding:8px;border-radius:6px;"
        f"background:{cvd.signal_color}22;border:1px solid {cvd.signal_color}'>"
        f"<b style='color:{cvd.signal_color}'>{cvd.signal_icon} "
        f"{cvd.signal.replace('_',' ')}</b>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.caption(cvd.description)

    c1, c2, c3 = st.columns(3)
    c1.metric("Buy Vol",      f"{cvd.buy_volume:,.0f}")
    c2.metric("Sell Vol",     f"{cvd.sell_volume:,.0f}")
    c3.metric("B/S Ratio",    f"{cvd.buy_sell_ratio:.2f}",
              delta="Buyers" if cvd.buy_sell_ratio > 1 else "Sellers",
              delta_color="normal" if cvd.buy_sell_ratio > 1 else "inverse")

    if cvd.divergence:
        if cvd.divergence_type == "HIDDEN_SELLING":
            st.warning("⚠️ Hidden Selling — be cautious on BUY entries")
        elif cvd.divergence_type == "HIDDEN_BUYING":
            st.info("💡 Hidden Buying — watch for reversal on SELL entries")
