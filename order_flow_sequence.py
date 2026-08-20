"""
order_flow_sequence.py  — v2
-----------------------
Order-flow SEQUENCE detector for NAS100. Additive extension — does not modify
scalping_engine.py, cvd_calculator.py, breadth_quality.py, or any other
existing module. Reads the ScalpReport (for BC/TC/pivot/VWAP/key levels) and
the same df_5m already cached in session_state — zero extra yfinance calls.

Implements the 4-stage institutional footprint sequence:

  1. SELLING PRESSURE   (Price↓ + Delta↓ + Volume↑)
     Sellers are aggressive AND getting results. "Healthy" bearish move.

  2. ABSORPTION          (Volume↑ + Delta strongly negative + price fails to
     make proportional downside)
     Sellers are getting MORE aggressive but price stops responding.
     Classic sign of a large passive buyer soaking up supply — most
     meaningful when it forms at/near a LIQUIDITY ZONE (BC, VWAP, prior
     day high/low, round numbers, or a recent swing high/low).

  3. SELLER EXHAUSTION   (bullish delta divergence: price makes a lower low
     while cumulative delta makes a higher/less-negative low)
     Selling volume is still there, but conviction is fading. Corroborated
     when RSI also makes a bullish divergence over the same two lows.

  4. REVERSAL CONFIRMED  (delta turns positive AND price reclaims BC)
     Selling → Absorption → Buyer Aggression → Reversal.

v2 ADDITIONS (this revision):
  - Liquidity-zone mapping — "where orders are sitting": pulls
    ScalpReport.key_levels (prev-day high/low/close, round numbers, CPR
    pivot/TC/BC/R1/S1/R2/S2) + VWAP + this module's own recently-detected
    swing highs/lows, and flags whether the current stage is forming AT one
    of them. A stage that forms away from any liquidity zone is much weaker
    evidence than the same stage forming exactly at VWAP or BC.
  - RSI divergence — computed on the same 5m bars, same RSI formula
    breadth_quality.py already uses (Wilder-style rolling mean), so the two
    panels never disagree on what "RSI" means. Checked over the same two
    swing lows as the delta divergence, so the two can corroborate or
    contradict each other explicitly.
  - `confidence` (LOW/MEDIUM/HIGH) — a simple confluence count across delta
    divergence, RSI divergence, and liquidity-zone proximity. This is the
    field intended to gate whether a signal is even worth wiring into
    master_signal.py's scoring later.

DELTA PROXY — IMPORTANT LIMITATION:
yfinance provides no bid/ask tape, so there is no true executed buy/sell
volume split. Per-bar delta is approximated the same way cvd_calculator.py
already does it elsewhere in this codebase: delta = volume * candle
body-position, i.e. volume * (close - open) / (high - low). This is a
volume-weighted directional-pressure proxy, not order-book delta. It is
internally consistent (same convention used for the existing CVD panel) and
is good enough to detect the shape of the sequence — but absolute delta
numbers won't match a real DOM/footprint tool. The "aggression" language in
this module (buyer/seller aggressive) describes this proxy, not literal
ask/bid-hit tape, since that data isn't available from yfinance either.
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
class RSIDivergence:
    detected: bool
    kind: str                    # "BULLISH" | "BEARISH"
    first_price: float
    first_rsi: float
    second_price: float
    second_rsi: float
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

    # Divergences
    divergence: Optional[DeltaDivergence]
    rsi: Optional[float]
    rsi_divergence: Optional[RSIDivergence]

    # Liquidity — "where orders are sitting"
    liquidity_zones: List[float]
    at_liquidity_zone: bool
    nearest_liquidity_zone: Optional[float]
    liquidity_zone_dist: Optional[float]

    # Stage classification
    stage: str                  # NEUTRAL | SELLING_PRESSURE | ABSORPTION | SELLER_EXHAUSTION | REVERSAL_CONFIRMED
    stage_color: str
    stage_icon: str
    description: str
    confidence: str             # LOW | MEDIUM | HIGH — confluence count (divergence + RSI + liquidity zone)

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


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Same Wilder-style rolling-mean RSI breadth_quality.py uses, so the two
    panels never disagree on what 'RSI' means — just applied intraday here."""
    delta = series.diff()
    gain  = delta.where(delta > 0, 0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))


def _find_swing_lows(prices: np.ndarray, order: int = 3) -> List[int]:
    """Local minima with `order` bars of confirmation on each side."""
    lows = []
    n = len(prices)
    for i in range(order, n - order):
        window = prices[i - order:i + order + 1]
        if prices[i] == window.min():
            lows.append(i)
    return lows


def _find_swing_highs(prices: np.ndarray, order: int = 3) -> List[int]:
    """Local maxima with `order` bars of confirmation on each side."""
    highs = []
    n = len(prices)
    for i in range(order, n - order):
        window = prices[i - order:i + order + 1]
        if prices[i] == window.max():
            highs.append(i)
    return highs


def _build_liquidity_zones(scalp_report, cpr, ratio: float,
                            swing_price_levels: List[float]) -> List[float]:
    """
    'Liquidity → where orders are sitting': prior day high/low/close, round
    numbers, and CPR pivot/TC/BC/R1/S1/R2/S2 all come pre-merged in
    ScalpReport.key_levels (scalping_engine._detect_key_levels). We add VWAP
    and this module's own recently-detected swing highs/lows on top.
    """
    zones: List[float] = []
    if scalp_report is not None:
        if getattr(scalp_report, "key_levels", None):
            zones.extend([l for l in scalp_report.key_levels if l])
        if getattr(scalp_report, "vwap", None):
            zones.append(float(scalp_report.vwap))
    elif cpr is not None:
        # Fallback when only cpr was passed (v1 call signature)
        for attr in ("pivot", "tc", "bc", "r1", "s1", "r2", "s2"):
            v = getattr(cpr, attr, None)
            if v:
                zones.append(float(v))

    zones.extend(swing_price_levels)
    return sorted(set(round(z, 0) for z in zones if z))


def _nearest_zone(price: float, zones: List[float]):
    if not zones:
        return None, None, False
    nearest = min(zones, key=lambda z: abs(z - price))
    dist = abs(nearest - price)
    at_zone = (dist / max(price, 1)) < 0.0015   # within ~0.15%, same tolerance as key_levels
    return nearest, round(dist, 0), at_zone


# ── MAIN ──────────────────────────────────────────────────────────────────────

def compute_order_flow_sequence(df_5m: pd.DataFrame,
                                 cpr,
                                 current_price: float,
                                 ratio: float = 40.0,
                                 lookback: int = 40,
                                 scalp_report=None) -> Optional[OrderFlowSequence]:
    """
    df_5m: NAS100/QQQ 5-minute OHLCV (same df already cached for scalping_engine).
    cpr: CPRLevels from scalping_engine._compute_cpr (or None — degrades gracefully).
    current_price: NAS100 index points.
    ratio: qqq_to_nas100_ratio — same one used everywhere else in the app.
    scalp_report: optional ScalpReport (analyse_nas100_scalp output). When
        passed, liquidity zones are built from its key_levels + vwap
        (prior day H/L/C, round numbers, full CPR ladder, VWAP) instead of
        just CPR. Old call sites that only pass `cpr` keep working — they
        just get a smaller liquidity-zone set built from CPR alone.
    """
    if df_5m is None or len(df_5m) < 25:
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
        recent['rsi'] = _rsi(recent['Close'])

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

        cpr_obj = cpr if cpr is not None else getattr(scalp_report, "cpr", None)
        bc = float(cpr_obj.bc) if cpr_obj is not None and getattr(cpr_obj, "bc", None) else None
        distance_from_bc = round(current_price - bc, 0) if bc is not None else None
        near_bc = bc is not None and abs(current_price - bc) / max(current_price, 1) < 0.003

        # ── Bullish delta divergence on recent swing lows ─────────────────────
        price_scaled = recent['price_scaled'].values
        cum_delta_vals = recent['cum_delta'].values
        rsi_vals = recent['rsi'].values
        swing_low_idxs = _find_swing_lows(price_scaled, order=3)

        divergence = None
        if len(swing_low_idxs) >= 2:
            i1, i2 = swing_low_idxs[-2], swing_low_idxs[-1]
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

        # ── RSI divergence over the same two swing lows ────────────────────────
        rsi_now = float(rsi_vals[-1]) if not np.isnan(rsi_vals[-1]) else None
        rsi_divergence = None
        if len(swing_low_idxs) >= 2:
            i1, i2 = swing_low_idxs[-2], swing_low_idxs[-1]
            p1, p2 = float(price_scaled[i1]), float(price_scaled[i2])
            r1, r2 = rsi_vals[i1], rsi_vals[i2]
            if not (np.isnan(r1) or np.isnan(r2)):
                if p2 < p1 and r2 > r1:
                    rsi_divergence = RSIDivergence(
                        detected=True, kind="BULLISH",
                        first_price=round(p1, 0), first_rsi=round(float(r1), 1),
                        second_price=round(p2, 0), second_rsi=round(float(r2), 1),
                        description=(
                            f"Bullish RSI divergence: price {p1:,.0f} \u2192 {p2:,.0f} (lower low) "
                            f"while RSI {r1:.0f} \u2192 {r2:.0f} (higher low)."
                        ),
                    )
        if rsi_divergence is None:
            swing_high_idxs = _find_swing_highs(price_scaled, order=3)
            if len(swing_high_idxs) >= 2:
                i1, i2 = swing_high_idxs[-2], swing_high_idxs[-1]
                p1, p2 = float(price_scaled[i1]), float(price_scaled[i2])
                r1, r2 = rsi_vals[i1], rsi_vals[i2]
                if not (np.isnan(r1) or np.isnan(r2)) and p2 > p1 and r2 < r1:
                    rsi_divergence = RSIDivergence(
                        detected=True, kind="BEARISH",
                        first_price=round(p1, 0), first_rsi=round(float(r1), 1),
                        second_price=round(p2, 0), second_rsi=round(float(r2), 1),
                        description=(
                            f"Bearish RSI divergence: price {p1:,.0f} \u2192 {p2:,.0f} (higher high) "
                            f"while RSI {r1:.0f} \u2192 {r2:.0f} (lower high)."
                        ),
                    )

        # ── Liquidity zones — "where orders are sitting" ───────────────────────
        swing_idxs_all = sorted(set(_find_swing_lows(price_scaled, order=3)
                                     + _find_swing_highs(price_scaled, order=3)))
        recent_swing_levels = [float(price_scaled[i]) for i in swing_idxs_all[-6:]]
        liquidity_zones = _build_liquidity_zones(scalp_report, cpr_obj, ratio, recent_swing_levels)
        nearest_zone, zone_dist, at_zone = _nearest_zone(current_price, liquidity_zones)

        # ── Stage classification ───────────────────────────────────────────────
        stage, color, icon = "NEUTRAL", "#888888", "\u26aa"
        desc = "No clear order-flow sequence forming."

        if cumulative_delta > 0 and price_chg_pct > 0.02 and (bc is None or current_price > bc):
            stage, color, icon = "REVERSAL_CONFIRMED", "#2d9e2d", "\U0001F7E2"
            bc_txt = f"BC ({bc:,.0f})" if bc is not None else "prior structure"
            desc = (f"Delta turned positive (cum \u0394 {cumulative_delta:+,.0f}) and price is "
                    f"reclaiming {bc_txt}. Buyer aggression confirmed \u2014 selling \u2192 "
                    f"absorption \u2192 buyer aggression \u2192 reversal.")
        elif divergence is not None and divergence.detected and (near_bc or at_zone or bc is None):
            stage, color, icon = "SELLER_EXHAUSTION", "#5cb85c", "\U0001F7E2"
            zone_txt = f" Occurring right at BC ({bc:,.0f})." if near_bc else (
                f" Occurring at a liquidity zone ({nearest_zone:,.0f})." if at_zone else "")
            desc = divergence.description + zone_txt
        elif volume_pace > 1.3 and cumulative_delta < 0 and abs(price_chg_pct) < 0.05:
            stage, color, icon = "ABSORPTION", "#e6a817", "\U0001F7E1"
            zone_txt = f" near BC ({bc:,.0f})" if near_bc else (f" near a liquidity zone ({nearest_zone:,.0f})" if at_zone else "")
            desc = (f"Volume {volume_pace:.1f}x average, cumulative delta {cumulative_delta:+,.0f} "
                    f"(strongly negative) but price nearly flat ({price_chg_pct:+.2f}%). "
                    f"Sellers aggressive but losing effectiveness{zone_txt} \u2014 potential absorption zone.")
        elif price_chg_pct < -0.02 and cumulative_delta < 0 and volume_pace > 1.0:
            stage, color, icon = "SELLING_PRESSURE", "#c9302c", "\U0001F534"
            desc = (f"Price {price_chg_pct:+.2f}%, cumulative delta {cumulative_delta:+,.0f}, "
                    f"volume {volume_pace:.1f}x average. Healthy bearish pressure \u2014 "
                    f"sellers aggressive and effective.")

        # ── Confidence — confluence count ──────────────────────────────────────
        confluences = 0
        if divergence is not None and divergence.detected:
            confluences += 1
        if rsi_divergence is not None and rsi_divergence.detected:
            confluences += 1
        if at_zone or near_bc:
            confluences += 1
        confidence = "HIGH" if confluences >= 3 else ("MEDIUM" if confluences == 2 else "LOW")

        return OrderFlowSequence(
            ticker="NAS100", current_price=current_price,
            volume=float(last['Volume']), volume_avg=round(vol_avg, 0), volume_pace=round(volume_pace, 2),
            delta=round(delta_last, 0), delta_pct=round(delta_pct, 1),
            cumulative_delta=round(cumulative_delta, 0),
            bc=bc, distance_from_bc=distance_from_bc,
            divergence=divergence,
            rsi=round(rsi_now, 1) if rsi_now is not None else None,
            rsi_divergence=rsi_divergence,
            liquidity_zones=liquidity_zones,
            at_liquidity_zone=at_zone, nearest_liquidity_zone=nearest_zone,
            liquidity_zone_dist=zone_dist,
            stage=stage, stage_color=color, stage_icon=icon, description=desc,
            confidence=confidence,
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

    conf_colors = {"HIGH": "#2d9e2d", "MEDIUM": "#e6a817", "LOW": "#888888"}
    conf_col = conf_colors.get(ofs.confidence, "#888888")
    st.markdown(
        f"<div style='padding:8px;border-radius:6px;background:{ofs.stage_color}22;"
        f"border-left:3px solid {ofs.stage_color};margin:6px 0'>"
        f"<b style='color:{ofs.stage_color}'>{ofs.stage_icon} {ofs.stage.replace('_', ' ')}</b> "
        f"<span style='float:right;font-size:0.75em;padding:2px 6px;border-radius:3px;"
        f"background:{conf_col}33;color:{conf_col};font-weight:bold'>{ofs.confidence} CONFIDENCE</span><br>"
        f"<span style='font-size:0.88em'>{ofs.description}</span></div>",
        unsafe_allow_html=True,
    )

    # Liquidity zone + RSI strip
    zone_txt = (f"\U0001F4CD At liquidity zone {ofs.nearest_liquidity_zone:,.0f}"
                if ofs.at_liquidity_zone else
                (f"Nearest liquidity zone: {ofs.nearest_liquidity_zone:,.0f} ({ofs.liquidity_zone_dist:,.0f} pts away)"
                 if ofs.nearest_liquidity_zone is not None else "No liquidity zones mapped"))
    rsi_txt = f"RSI {ofs.rsi:.0f}" if ofs.rsi is not None else "RSI n/a"
    st.caption(f"{zone_txt}  \u00b7  {rsi_txt}")

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

    if ofs.rsi_divergence and ofs.rsi_divergence.detected:
        icon = "\U0001F7E2" if ofs.rsi_divergence.kind == "BULLISH" else "\U0001F534"
        st.caption(f"{icon} {ofs.rsi_divergence.description}")
