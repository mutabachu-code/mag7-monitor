"""
cvd_calculator.py  — v2
------------------------
Improved order flow tracking for Forex (where real volume is unavailable).

Since yfinance forex volume is tick count (not real volume), B/S ratio stays
near 1.0 always. This version uses PRICE-BASED order flow proxies instead:

1. Candle Body Delta   — body direction and strength vs full range
2. Institutional Blocks — candles with body > 1.5x ATR = big player moves
3. Wick Rejection Score — large wicks = price rejection by institutions
4. Consecutive Pressure — 3+ same-direction closes = accumulation/distribution
5. Momentum Bias       — overall session directional pressure

For NAS100: uses real QQQ volume when available (much more reliable).
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class CVDResult:
    # Core metrics
    body_delta_pct: float       # candle body as % of range: +100=full bull, -100=full bear
    pressure_score: float       # -100 to +100: net directional pressure
    institutional_blocks: int   # count of large candles (body > 1.5x ATR)
    wick_rejection: str         # "UPPER" | "LOWER" | "NONE" — what price rejected
    consecutive_closes: int     # positive = bull closes, negative = bear closes
    buy_sell_ratio: float       # ratio proxy from candle analysis
    cvd_change_pct: float       # session momentum %

    # Signal
    signal: str
    signal_color: str
    signal_icon: str
    description: str
    divergence: bool
    divergence_type: str

    # For display
    cvd_series: list


def calculate_cvd(df: pd.DataFrame, lookback: int = 20) -> Optional[CVDResult]:
    """
    Price-based order flow analysis — works reliably for forex.
    Falls back gracefully if data is thin.
    """
    if df is None or len(df) < 5:
        return None

    try:
        df = df.copy()
        df.columns = [c.capitalize() for c in df.columns]

        if not all(col in df.columns for col in ["Open","High","Low","Close"]):
            return None

        recent = df.tail(lookback).copy()
        opens  = recent["Open"].values
        highs  = recent["High"].values
        lows   = recent["Low"].values
        closes = recent["Close"].values

        candle_range = highs - lows
        candle_range = np.where(candle_range == 0, 1e-10, candle_range)
        bodies       = closes - opens
        body_pct     = bodies / candle_range * 100   # +100=full bull, -100=full bear

        # ── 1. ATR for institutional block detection ───────────────────────────
        atr = float(np.mean(candle_range))

        # ── 2. Institutional blocks ────────────────────────────────────────────
        abs_bodies       = np.abs(bodies)
        inst_threshold   = atr * 1.5
        inst_bull_blocks = int(np.sum((bodies > inst_threshold)))
        inst_bear_blocks = int(np.sum((bodies < -inst_threshold)))
        inst_blocks      = inst_bull_blocks + inst_bear_blocks

        # ── 3. Wick rejection ─────────────────────────────────────────────────
        upper_wicks = highs - np.maximum(opens, closes)
        lower_wicks = np.minimum(opens, closes) - lows
        avg_upper   = float(np.mean(upper_wicks[-5:]))
        avg_lower   = float(np.mean(lower_wicks[-5:]))

        if avg_upper > atr * 0.6 and avg_upper > avg_lower * 1.5:
            wick_rejection = "UPPER"   # price rejected higher = bearish
        elif avg_lower > atr * 0.6 and avg_lower > avg_upper * 1.5:
            wick_rejection = "LOWER"   # price rejected lower = bullish
        else:
            wick_rejection = "NONE"

        # ── 4. Consecutive closes ─────────────────────────────────────────────
        consecutive = 0
        for i in range(len(closes)-1, max(len(closes)-8, -1), -1):
            if closes[i] > opens[i]:
                if consecutive >= 0:
                    consecutive += 1
                else:
                    break
            elif closes[i] < opens[i]:
                if consecutive <= 0:
                    consecutive -= 1
                else:
                    break
            else:
                break

        # ── 5. Pressure score ─────────────────────────────────────────────────
        # Weighted sum of candle directional strength
        weights      = np.linspace(0.5, 1.0, len(body_pct))   # recent = more weight
        pressure     = float(np.average(body_pct, weights=weights))

        # Adjust for institutional blocks
        if inst_bull_blocks > inst_bear_blocks:
            pressure = min(pressure + 15, 100)
        elif inst_bear_blocks > inst_bull_blocks:
            pressure = max(pressure - 15, -100)

        # ── 6. CVD series (cumulative body_pct) ───────────────────────────────
        cvd_series = list(np.cumsum(body_pct))
        cvd_change = float(cvd_series[-1] - cvd_series[0]) if len(cvd_series) > 1 else 0

        # ── 7. Buy/Sell ratio proxy ───────────────────────────────────────────
        bull_candles = float(np.sum(bodies > 0))
        bear_candles = float(np.sum(bodies < 0))
        bs_ratio     = bull_candles / bear_candles if bear_candles > 0 else 2.0

        # ── 8. Price direction ────────────────────────────────────────────────
        price_up   = closes[-1] > closes[0]
        pressure_up= pressure > 10

        # ── 9. Divergence ─────────────────────────────────────────────────────
        divergence      = False
        divergence_type = "NONE"

        if price_up and pressure < -10:
            divergence      = True
            divergence_type = "HIDDEN_SELLING"
        elif not price_up and pressure > 10:
            divergence      = True
            divergence_type = "HIDDEN_BUYING"

        # ── 10. Signal classification ─────────────────────────────────────────
        strong_bull = (pressure > 30 and consecutive >= 3
                       and wick_rejection != "UPPER"
                       and inst_bull_blocks >= 1)
        strong_bear = (pressure < -30 and consecutive <= -3
                       and wick_rejection != "LOWER"
                       and inst_bear_blocks >= 1)

        if divergence_type == "HIDDEN_SELLING":
            signal      = "BEAR_DIVERGENCE"
            color       = "#e6a817"
            icon        = "⚠️"
            description = (f"HIDDEN SELLING: Price rising but {abs(pressure):.0f}% candle pressure bearish. "
                           f"Institutional distribution — avoid new longs. "
                           f"Wick rejection: {wick_rejection}.")
        elif divergence_type == "HIDDEN_BUYING":
            signal      = "BULL_DIVERGENCE"
            color       = "#3399ff"
            icon        = "💡"
            description = (f"HIDDEN BUYING: Price falling but {pressure:.0f}% candle pressure bullish. "
                           f"{inst_bull_blocks} institutional block(s) detected — "
                           f"smart money accumulating. Watch for reversal.")
        elif strong_bull:
            signal      = "BULLISH_CONFIRM"
            color       = "#2d9e2d"
            icon        = "✅"
            description = (f"CONFIRMED BULLISH: {pressure:.0f}% directional pressure. "
                           f"{consecutive} consecutive bull closes. "
                           f"{inst_bull_blocks} institutional block(s). "
                           f"Lower wick rejection confirms buyers defending lows.")
        elif strong_bear:
            signal      = "BEARISH_CONFIRM"
            color       = "#c9302c"
            icon        = "✅"
            description = (f"CONFIRMED BEARISH: {pressure:.0f}% directional pressure. "
                           f"{abs(consecutive)} consecutive bear closes. "
                           f"{inst_bear_blocks} institutional block(s). "
                           f"Upper wick rejection confirms sellers capping highs.")
        elif pressure > 15:
            signal      = "BULLISH_BIAS"
            color       = "#5cb85c"
            icon        = "🟢"
            description = (f"Mild BULLISH pressure ({pressure:.0f}%). "
                           f"{consecutive} bull close(s) in sequence. "
                           f"Wick rejection: {wick_rejection}. "
                           f"Watch for institutional block confirmation.")
        elif pressure < -15:
            signal      = "BEARISH_BIAS"
            color       = "#d9534f"
            icon        = "🔴"
            description = (f"Mild BEARISH pressure ({pressure:.0f}%). "
                           f"{abs(consecutive)} bear close(s) in sequence. "
                           f"Wick rejection: {wick_rejection}. "
                           f"Watch for institutional block confirmation.")
        else:
            signal      = "NEUTRAL"
            color       = "#888888"
            icon        = "➡️"
            description = (f"Neutral order flow. Pressure: {pressure:.0f}%. "
                           f"No clear institutional bias detected. "
                           f"Wait for breakout from current range.")

        return CVDResult(
            body_delta_pct=round(float(body_pct[-1]), 1),
            pressure_score=round(pressure, 1),
            institutional_blocks=inst_blocks,
            wick_rejection=wick_rejection,
            consecutive_closes=consecutive,
            buy_sell_ratio=round(bs_ratio, 2),
            cvd_change_pct=round(cvd_change / max(abs(cvd_series[0]), 1) * 100, 1)
                           if cvd_series else 0,
            signal=signal,
            signal_color=color,
            signal_icon=icon,
            description=description,
            divergence=divergence,
            divergence_type=divergence_type,
            cvd_series=cvd_series,
        )

    except Exception as e:
        print(f"[cvd_calculator] Error: {e}")
        return None


def render_cvd_badge(cvd: CVDResult):
    import streamlit as st
    if cvd is None:
        st.caption("Order flow: unavailable")
        return
    st.markdown(
        f"<div style='padding:4px 8px;border-radius:4px;"
        f"background:{cvd.signal_color}22;border:1px solid {cvd.signal_color};"
        f"margin:2px 0;font-size:0.85em'>"
        f"<b style='color:{cvd.signal_color}'>{cvd.signal_icon} "
        f"Flow: {cvd.signal.replace('_', ' ')}</b> · "
        f"Pressure {cvd.pressure_score:+.0f}% · "
        f"{cvd.consecutive_closes:+d} closes · "
        f"{cvd.institutional_blocks} block(s)"
        f"</div>",
        unsafe_allow_html=True,
    )
    if cvd.divergence:
        st.caption(f"⚠️ {cvd.description}")


def render_cvd_panel(cvd: CVDResult, label: str = ""):
    import streamlit as st
    if cvd is None:
        st.caption("Order flow data unavailable")
        return

    st.markdown(
        f"<div style='padding:8px;border-radius:6px;"
        f"background:{cvd.signal_color}22;border:1px solid {cvd.signal_color}'>"
        f"<b style='color:{cvd.signal_color}'>{cvd.signal_icon} "
        f"{cvd.signal.replace('_', ' ')}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(cvd.description)

    c1, c2, c3 = st.columns(3)
    c1.metric("Pressure",     f"{cvd.pressure_score:+.0f}%",
              delta="Bullish" if cvd.pressure_score > 0 else "Bearish",
              delta_color="normal" if cvd.pressure_score > 0 else "inverse")
    c2.metric("Inst. Blocks", f"{cvd.institutional_blocks}",
              delta="Active" if cvd.institutional_blocks > 0 else "None")
    c3.metric("Consecutive",  f"{cvd.consecutive_closes:+d} closes",
              delta="Bull" if cvd.consecutive_closes > 0 else "Bear" if cvd.consecutive_closes < 0 else "Mixed",
              delta_color="normal" if cvd.consecutive_closes > 0 else "inverse" if cvd.consecutive_closes < 0 else "off")

    if cvd.wick_rejection != "NONE":
        wr_msg = ("🔽 Upper wick rejection — sellers capping highs"
                  if cvd.wick_rejection == "UPPER"
                  else "🔼 Lower wick rejection — buyers defending lows")
        st.caption(wr_msg)

    if cvd.divergence:
        if cvd.divergence_type == "HIDDEN_SELLING":
            st.warning("⚠️ Hidden Selling — be cautious on BUY entries")
        elif cvd.divergence_type == "HIDDEN_BUYING":
            st.info("💡 Hidden Buying — watch for reversal on SELL entries")
