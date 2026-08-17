"""
order_flow_sequence.py
-----------------------
Order-flow SEQUENCE detector for NAS100. Additive extension — does not modify
scalping_engine.py, cvd_calculator.py, or any other existing module. Reads
CPRLevels (for the BC anchor) and the same df_5m already cached in
session_state — zero extra yfinance calls.

Implements the 4-stage institutional footprint sequence:

  1. SELLING PRESSURE   (Price↓ + Delta↓ + Volume↑)
     Sellers are aggressive AND getting results. "Healthy" bearish move.

  2. ABSORPTION          (Volume↑ + Delta strongly negative + price fails to
     make proportional downside)
     Sellers are getting MORE aggressive but price stops responding.
     Classic sign of a large passive buyer soaking up supply — most
     meaningful when it forms at/near the CPR's BC level.

  3. SELLER EXHAUSTION   (bullish delta divergence: price makes a lower low
     while cumulative delta makes a higher/less-negative low)
     Selling volume is still there, but conviction is fading.

  4. REVERSAL CONFIRMED  (delta turns positive AND price reclaims BC)
     Selling → Absorption → Buyer Aggression → Reversal.

DELTA PROXY — IMPORTANT LIMITATION:
yfinance provides no bid/ask tape, so there is no true executed buy/sell
volume split. Per-bar delta is approximated the same way cvd_calculator.py
already does it elsewhere in this codebase: delta = volume * candle
body-position, i.e. volume * (close - open) / (high - low). This is a
volume-weighted directional-pressure proxy, not order-book delta. It is
internally consistent (same convention used for the existing CVD panel) and
is good enough to detect the shape of the sequence — but absolute delta
numbers won't match a real DOM/footprint tool.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

try:
    from scalping_engine import CPRLevels
except Exception:
    CPRLevels = None  # keeps this module importable even if scalping_engine changes


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class DeltaDivergence:
    detected: bool
    first_low_price: float
    first_low_delta: float
    second_low_price: float
    second_low_delta: float
    description: str


@dataclass
class OrderFlowSequence:
    ticker: str
    current_price: float

    # Snapshot (mirrors the ChatGPT spec: Volume / Delta / Delta% / Cum Delta)
    volume: float
    volume_avg: float
    volume_pace: float          # e.g. 2.4 == "2.4x average"
    delta: float                # last bar's signed delta proxy
    delta_pct: float            # delta as % of that bar's volume
    cumulative_delta: float     # net delta change over the lookback window

    # BC anchor (from CPRLevels)
    bc: Optional[float]
    distance_from_bc: Optional[float]   # +pts = price above BC

    # Divergence
    divergence: Optional[DeltaDivergence]

    # Stage classification
    stage: str                  # NEUTRAL | SELLING_PRESSURE | ABSORPTION | SELLER_EXHAUSTION | REVERSAL_CONFIRMED
    stage_color: str
    stage_icon: str
    description: str

    sequence_stages: List[str] = field(default_factory=lambda: [
        "SELLING_PRESSURE", "ABSORPTION", "SELLER_EXHAUSTION", "REVERSAL_CONFIRMED"
    ])


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _per_bar_delta(df: pd.DataFrame) -> pd.Series:
    """Volume-weighted directional-pressure proxy (see module docstring)."""
    o, h, l, c, v = df['Open'], df['High'], df['Low'], df['Close'], df['Volume']
    rng = (h - l).replace(0, np.nan)
    body_pos = ((c - o) / rng).fillna(0).clip(-1, 1)
    return body_pos * v


def _find_swing_lows(prices: np.ndarray, order: int = 3) -> List[int]:
    """Local minima with `order` bars of confirmation on each side."""
    lows = []
    n = len(prices)
    for i in range(order, n - order):
        window = prices[i - order:i + order + 1]
        if prices[i] == window.min():
            lows.append(i)
    return lows


# ── MAIN ──────────────────────────────────────────────────────────────────────

def compute_order_flow_sequence(df_5m: pd.DataFrame,
                                 cpr,
                                 current_price: float,
                                 ratio: float = 40.0,
                                 lookback: int = 40) -> Optional[OrderFlowSequence]:
    """
    df_5m: NAS100/QQQ 5-minute OHLCV (same df already cached for scalping_engine).
    cpr: CPRLevels from scalping_engine._compute_cpr (or None — degrades gracefully).
    current_price: NAS100 index points.
    ratio: qqq_to_nas100_ratio — same one used everywhere else in the app.
    """
    if df_5m is None or len(df_5m) < 15:
        return None

    try:
        df = df_5m.copy()
        df.columns = [c.capitalize() for c in df.columns]
        if not all(c in df.columns for c in ["Open", "High", "Low", "Close", "Volume"]):
            return None

        recent = df.tail(lookback).copy()
        recent['delta'] = _per_bar_delta(recent)
        recent['cum_delta'] = recent['delta'].cumsum()
        recent['price_scaled'] = recent['Close'] * ratio

        last = recent.iloc[-1]
        vol_avg = float(recent['Volume'].rolling(20, min_periods=5).mean().iloc[-1])
        vol_avg = vol_avg if vol_avg and vol_avg > 0 else 1.0
        volume_pace = float(last['Volume'] / vol_avg)

        delta_last = float(last['delta'])
        delta_pct = float(delta_last / last['Volume'] * 100) if last['Volume'] > 0 else 0.0

        n_trend = min(6, len(recent) - 1)
        cumulative_delta = float(recent['cum_delta'].iloc[-1] - recent['cum_delta'].iloc[-1 - n_trend])
        price_chg_pct = float(
            (recent['Close'].iloc[-1] - recent['Close'].iloc[-1 - n_trend])
            / recent['Close'].iloc[-1 - n_trend] * 100
        )

        bc = float(cpr.bc) if cpr is not None and getattr(cpr, "bc", None) else None
        distance_from_bc = round(current_price - bc, 0) if bc is not None else None
        near_bc = bc is not None and abs(current_price - bc) / max(current_price, 1) < 0.003

        # ── Bullish delta divergence on recent swing lows ─────────────────────
        price_scaled = recent['price_scaled'].values
        cum_delta_vals = recent['cum_delta'].values
        swing_idxs = _find_swing_lows(price_scaled, order=3)

        divergence = None
        if len(swing_idxs) >= 2:
            i1, i2 = swing_idxs[-2], swing_idxs[-1]
            p1, p2 = float(price_scaled[i1]), float(price_scaled[i2])
            d1, d2 = float(cum_delta_vals[i1]), float(cum_delta_vals[i2])
            if p2 < p1 and d2 > d1:
                divergence = DeltaDivergence(
                    detected=True,
                    first_low_price=round(p1, 0), first_low_delta=round(d1, 0),
                    second_low_price=round(p2, 0), second_low_delta=round(d2, 0),
                    description=(
                        f"Bullish delta divergence: price {p1:,.0f} \u2192 {p2:,.0f} (lower low) "
                        f"while cumulative delta {d1:,.0f} \u2192 {d2:,.0f} (higher low). "
                        f"Selling aggression weakening."
                    ),
                )

        # ── Stage classification ───────────────────────────────────────────────
        stage, color, icon = "NEUTRAL", "#888888", "\u26aa"
        desc = "No clear order-flow sequence forming."

        if cumulative_delta > 0 and price_chg_pct > 0.02 and (bc is None or current_price > bc):
            stage, color, icon = "REVERSAL_CONFIRMED", "#2d9e2d", "\U0001F7E2"
            bc_txt = f"BC ({bc:,.0f})" if bc is not None else "prior structure"
            desc = (f"Delta turned positive (cum \u0394 {cumulative_delta:+,.0f}) and price is "
                    f"reclaiming {bc_txt}. Buyer aggression confirmed \u2014 selling \u2192 "
                    f"absorption \u2192 buyer aggression \u2192 reversal.")
        elif divergence is not None and divergence.detected and (near_bc or bc is None):
            stage, color, icon = "SELLER_EXHAUSTION", "#5cb85c", "\U0001F7E2"
            desc = divergence.description + (f" Occurring right at BC ({bc:,.0f})." if near_bc else "")
        elif volume_pace > 1.3 and cumulative_delta < 0 and abs(price_chg_pct) < 0.05:
            stage, color, icon = "ABSORPTION", "#e6a817", "\U0001F7E1"
            bc_txt = f" near BC ({bc:,.0f})" if near_bc else ""
            desc = (f"Volume {volume_pace:.1f}x average, cumulative delta {cumulative_delta:+,.0f} "
                    f"(strongly negative) but price nearly flat ({price_chg_pct:+.2f}%). "
                    f"Sellers aggressive but losing effectiveness{bc_txt} \u2014 potential absorption zone.")
        elif price_chg_pct < -0.02 and cumulative_delta < 0 and volume_pace > 1.0:
            stage, color, icon = "SELLING_PRESSURE", "#c9302c", "\U0001F534"
            desc = (f"Price {price_chg_pct:+.2f}%, cumulative delta {cumulative_delta:+,.0f}, "
                    f"volume {volume_pace:.1f}x average. Healthy bearish pressure \u2014 "
                    f"sellers aggressive and effective.")

        return OrderFlowSequence(
            ticker="NAS100", current_price=current_price,
            volume=float(last['Volume']), volume_avg=round(vol_avg, 0), volume_pace=round(volume_pace, 2),
            delta=round(delta_last, 0), delta_pct=round(delta_pct, 1),
            cumulative_delta=round(cumulative_delta, 0),
            bc=bc, distance_from_bc=distance_from_bc,
            divergence=divergence,
            stage=stage, stage_color=color, stage_icon=icon, description=desc,
        )

    except Exception as e:
        print(f"[order_flow_sequence] Error: {e}")
        return None


# ── RENDER ────────────────────────────────────────────────────────────────────

def render_order_flow_sequence(ofs: Optional[OrderFlowSequence]):
    import streamlit as st

    st.markdown("**\U0001F52C Order Flow Sequence**")
    if ofs is None:
        st.caption("Order flow sequence unavailable")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Volume Pace", f"{ofs.volume_pace:.1f}x avg")
    c2.metric("Delta", f"{ofs.delta:+,.0f}", delta=f"{ofs.delta_pct:+.0f}%")
    c3.metric("Cum. Delta", f"{ofs.cumulative_delta:+,.0f}")
    c4.metric("Dist. from BC", f"{ofs.distance_from_bc:+.0f} pts" if ofs.distance_from_bc is not None else "n/a")

    st.markdown(
        f"<div style='padding:8px;border-radius:6px;background:{ofs.stage_color}22;"
        f"border-left:3px solid {ofs.stage_color};margin:6px 0'>"
        f"<b style='color:{ofs.stage_color}'>{ofs.stage_icon} {ofs.stage.replace('_', ' ')}</b><br>"
        f"<span style='font-size:0.88em'>{ofs.description}</span></div>",
        unsafe_allow_html=True,
    )

    # 4-stage progress strip: Selling Pressure -> Absorption -> Seller Exhaustion -> Reversal Confirmed
    stage_order = ["SELLING_PRESSURE", "ABSORPTION", "SELLER_EXHAUSTION", "REVERSAL_CONFIRMED"]
    labels = {
        "SELLING_PRESSURE": "Selling Pressure",
        "ABSORPTION": "Absorption",
        "SELLER_EXHAUSTION": "Seller Exhaustion",
        "REVERSAL_CONFIRMED": "Reversal Confirmed",
    }
    stage_colors = {
        "SELLING_PRESSURE": "#c9302c", "ABSORPTION": "#e6a817",
        "SELLER_EXHAUSTION": "#5cb85c", "REVERSAL_CONFIRMED": "#2d9e2d",
    }
    cols = st.columns(4)
    for i, s in enumerate(stage_order):
        active = (ofs.stage == s)
        bg = stage_colors[s]
        opacity = "" if active else "33"
        text_col = "white" if active else "#999"
        with cols[i]:
            st.markdown(
                f"<div style='text-align:center;padding:5px 2px;border-radius:4px;"
                f"background:{bg}{opacity};font-size:0.72em;font-weight:{'bold' if active else 'normal'};"
                f"color:{text_col}'>{labels[s]}</div>",
                unsafe_allow_html=True,
            )

    if ofs.divergence and ofs.divergence.detected and ofs.stage != "SELLER_EXHAUSTION":
        st.caption(f"\U0001F4D0 {ofs.divergence.description}")
