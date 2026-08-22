"""
mean_reversion_atr.py
----------------------
ATR-based mean-reversion fade setup for NAS100. Additive module — reads
order_flow_sequence.OrderFlowSequence (already computed) plus the same
ScalpReport/df_5m cached for everything else. No new yfinance calls.

STRATEGY (institutional-standard "VWAP/Pivot reversion, ATR-sized, order-flow
confirmed" — the same shape used by intraday mean-reversion desks):

  ANCHOR   Fair-value reference for the session: VWAP (primary — this is what
           most mean-reversion desks fade back toward, since it's the
           volume-weighted average price actually paid). Falls back to the
           CPR Pivot if VWAP isn't available.

  EXTENSION  How far price has stretched from the anchor, measured in ATR(14)
           units (Wilder true range, not the simplified high-low range
           cvd_calculator.py uses elsewhere — ATR needs the true-range gap
           term to size stops correctly). extension_atr = (price - anchor) / atr.

  TRIGGER  A fade is only considered once price is stretched >= 2.0 ATR from
           the anchor. Below that, reversion odds aren't good enough to risk
           capital against the trend.

  CONFIRMATION (this is what makes it an "order-flow confirmed" fade instead
  of a blind extension fade — extension alone is not a strategy). Gated as a
  confluence COUNT, not a rigid AND-chain — seller exhaustion structurally
  confirms right as RSI is already recovering off its trough, so demanding
  stage-confirmation AND live RSI<35 at the same instant would rarely both
  be true together:
    LONG (fade oversold) — needs >= 2 of 3:
      1. order_flow_sequence.stage in {SELLER_EXHAUSTION, ABSORPTION,
         REVERSAL_CONFIRMED} with confidence != LOW
      2. a detected bullish RSI divergence
      3. RSI < 40 (oversold)
    SHORT (fade overbought) — needs both of the 2 available:
      1. a detected bearish RSI divergence (order_flow_sequence only models
         the bearish→bullish sequence explicitly today, so short setups lean
         on RSI divergence + extension rather than a symmetric stage machine
         — stated here honestly rather than pretending one exists)
      2. RSI > 60 (overbought)
    Extra confluence either direction: forming at a liquidity zone
    (order_flow_sequence.at_liquidity_zone) adds to confidence but isn't
    required.

  STOPS & TARGETS — all ATR-denominated, not fixed points, so sizing scales
  with current volatility:
    STOP    entry -/+ (stop_atr_mult × ATR), pulled tighter to just beyond
            the nearest recent swing low/high if that's closer (structure-
            based stop beats a pure volatility stop when structure is closer).
    TARGET 1  reversion to the anchor (VWAP/Pivot) — the base case.
    TARGET 2  one ATR beyond the anchor on the far side — the stretch case
            for when reversion overshoots (common after a large extension).

  A setup is only marked `valid` if the R:R on Target 1 clears `min_rr`
  (default 1.3). Below that, it's shown as a diagnostic only, not a signal.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from order_flow_sequence import OrderFlowSequence, _find_swing_lows, _find_swing_highs


@dataclass
class MeanReversionSetup:
    ticker: str
    current_price: float

    anchor: Optional[float]
    anchor_label: str            # "VWAP" | "CPR Pivot" | "n/a"
    atr: Optional[float]
    extension_atr: Optional[float]   # signed: -2.4 = 2.4 ATR below anchor

    direction: Optional[str]     # "LONG" | "SHORT" | None
    entry: Optional[float]
    stop_loss: Optional[float]
    target_1: Optional[float]
    target_2: Optional[float]
    risk_pts: Optional[float]
    reward_1_pts: Optional[float]
    risk_reward_1: Optional[float]

    confluences: List[str]
    confidence: str              # LOW | MEDIUM | HIGH
    valid: bool                  # tradeable setup right now (extension + confirmation + min R:R)
    description: str


def _wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    True Wilder ATR (uses the gap term), distinct from the simplified
    mean(high-low) 'ATR' used in cvd_calculator.py for institutional-block
    sizing. Stops need the gap term; that module's use case doesn't.
    """
    h, l, c = df['High'], df['Low'], df['Close']
    prev_close = c.shift(1)
    tr = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def compute_mean_reversion_setup(df_5m: pd.DataFrame,
                                  ofs: Optional[OrderFlowSequence],
                                  scalp_report,
                                  current_price: float,
                                  ratio: float = 40.0,
                                  atr_period: int = 14,
                                  entry_ext: float = 2.0,
                                  stop_atr_mult: float = 1.0,
                                  min_rr: float = 1.3,
                                  lookback: int = 60) -> Optional[MeanReversionSetup]:
    if df_5m is None or len(df_5m) < atr_period + 5:
        return None

    try:
        df = df_5m.copy()
        df.columns = [c.capitalize() for c in df.columns]
        if not all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            return None

        recent = df.tail(lookback).copy()
        recent_scaled = recent[["Open", "High", "Low", "Close"]] * ratio
        atr_series = _wilder_atr(recent_scaled, period=atr_period)
        atr = float(atr_series.iloc[-1]) if not np.isnan(atr_series.iloc[-1]) else None

        # ── Anchor: VWAP preferred, CPR Pivot fallback ────────────────────────
        anchor, anchor_label = None, "n/a"
        vwap = getattr(scalp_report, "vwap", None) if scalp_report is not None else None
        cpr = getattr(scalp_report, "cpr", None) if scalp_report is not None else None
        if vwap:
            anchor, anchor_label = float(vwap), "VWAP"
        elif cpr is not None and getattr(cpr, "pivot", None):
            anchor, anchor_label = float(cpr.pivot), "CPR Pivot"

        if atr is None or atr <= 0 or anchor is None:
            return MeanReversionSetup(
                ticker="NAS100", current_price=current_price,
                anchor=anchor, anchor_label=anchor_label, atr=atr, extension_atr=None,
                direction=None, entry=None, stop_loss=None, target_1=None, target_2=None,
                risk_pts=None, reward_1_pts=None, risk_reward_1=None,
                confluences=[], confidence="LOW", valid=False,
                description="ATR or anchor unavailable — mean reversion setup not computable yet.",
            )

        extension_atr = (current_price - anchor) / atr

        # ── Confirmation inputs from order flow sequence ───────────────────────
        rsi = ofs.rsi if ofs is not None else None
        ofs_stage = ofs.stage if ofs is not None else "NEUTRAL"
        ofs_confidence = ofs.confidence if ofs is not None else "LOW"
        at_zone = bool(ofs is not None and ofs.at_liquidity_zone)
        bull_rsi_div = bool(ofs is not None and ofs.rsi_divergence and
                             ofs.rsi_divergence.detected and ofs.rsi_divergence.kind == "BULLISH")
        bear_rsi_div = bool(ofs is not None and ofs.rsi_divergence and
                             ofs.rsi_divergence.detected and ofs.rsi_divergence.kind == "BEARISH")

        direction = None
        confluences: List[str] = []

        # ── LONG: fade oversold extension ──────────────────────────────────────
        # Confluence-counted rather than a rigid AND-chain: seller exhaustion
        # structurally confirms right as RSI is already recovering off its
        # trough, so requiring stage-confirmation AND live RSI<35 at the same
        # instant would almost never both be true together. Instead, require
        # at least 2 of 3 independent confirmations.
        if extension_atr <= -entry_ext:
            stage_confirms = ofs_stage in ("SELLER_EXHAUSTION", "ABSORPTION", "REVERSAL_CONFIRMED") \
                and ofs_confidence != "LOW"
            rsi_oversold = rsi is not None and rsi < 40

            if stage_confirms:
                confluences.append(f"Order flow {ofs_stage.replace('_',' ').title()} ({ofs_confidence.lower()} confidence)")
            if bull_rsi_div:
                confluences.append("Bullish RSI divergence")
            if rsi_oversold:
                confluences.append(f"RSI oversold ({rsi:.0f})")
            if at_zone:
                confluences.append(f"At liquidity zone ({ofs.nearest_liquidity_zone:,.0f})")

            confirm_count = sum([stage_confirms, bull_rsi_div, rsi_oversold])
            if confirm_count >= 2:
                direction = "LONG"

        # ── SHORT: fade overbought extension ────────────────────────────────────
        elif extension_atr >= entry_ext:
            rsi_overbought = rsi is not None and rsi > 60

            if bear_rsi_div:
                confluences.append("Bearish RSI divergence")
            if rsi_overbought:
                confluences.append(f"RSI overbought ({rsi:.0f})")
            if at_zone:
                confluences.append(f"At liquidity zone ({ofs.nearest_liquidity_zone:,.0f})")

            # Short side has one fewer independent signal available (order_flow_sequence
            # doesn't yet model a symmetric buyer-exhaustion state machine — see module
            # docstring), so require both of what IS available rather than 2-of-3.
            if bear_rsi_div and rsi_overbought:
                direction = "SHORT"

        # ── Build entry / stop / targets (ATR-denominated) ─────────────────────
        entry = stop_loss = target_1 = target_2 = None
        risk_pts = reward_1_pts = risk_reward_1 = None

        price_scaled = recent_scaled['Close'].values
        if direction == "LONG":
            entry = current_price
            structure_stop = None
            lows_idx = _find_swing_lows(price_scaled, order=3)
            if lows_idx:
                structure_stop = float(price_scaled[lows_idx[-1]]) - 0.25 * atr
            vol_stop = entry - stop_atr_mult * atr
            stop_loss = max(vol_stop, structure_stop) if structure_stop else vol_stop
            target_1 = anchor
            target_2 = anchor + 1.0 * atr
            risk_pts = entry - stop_loss
            reward_1_pts = target_1 - entry

        elif direction == "SHORT":
            entry = current_price
            structure_stop = None
            highs_idx = _find_swing_highs(price_scaled, order=3)
            if highs_idx:
                structure_stop = float(price_scaled[highs_idx[-1]]) + 0.25 * atr
            vol_stop = entry + stop_atr_mult * atr
            stop_loss = min(vol_stop, structure_stop) if structure_stop else vol_stop
            target_1 = anchor
            target_2 = anchor - 1.0 * atr
            risk_pts = stop_loss - entry
            reward_1_pts = entry - target_1

        if risk_pts is not None and risk_pts > 0:
            risk_reward_1 = reward_1_pts / risk_pts

        valid = bool(direction and risk_reward_1 is not None and risk_reward_1 >= min_rr)
        if direction and not valid:
            confluences.append(f"R:R {risk_reward_1:.2f} below {min_rr:.1f} minimum \u2014 setup not actionable")

        # ── Confidence ───────────────────────────────────────────────────────────
        strong_confluences = sum(1 for c in confluences if "R:R" not in c)
        if valid and strong_confluences >= 3:
            confidence = "HIGH"
        elif valid and strong_confluences == 2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        if direction:
            desc = (f"Price is {abs(extension_atr):.1f} ATR {'below' if direction=='LONG' else 'above'} "
                    f"{anchor_label} ({anchor:,.0f}). " +
                    (f"Confirmed fade {direction.lower()} \u2014 " + "; ".join(confluences) + "."
                     if valid else
                     "Extension + some confirmation present but not enough for a valid setup: " +
                     "; ".join(confluences) + "."))
        else:
            desc = (f"Price is {extension_atr:+.1f} ATR from {anchor_label} ({anchor:,.0f}) \u2014 "
                    f"{'within normal range' if abs(extension_atr) < entry_ext else 'extended but unconfirmed by order flow/RSI'}.")

        return MeanReversionSetup(
            ticker="NAS100", current_price=current_price,
            anchor=round(anchor, 0), anchor_label=anchor_label,
            atr=round(atr, 1), extension_atr=round(extension_atr, 2),
            direction=direction,
            entry=round(entry, 0) if entry is not None else None,
            stop_loss=round(stop_loss, 0) if stop_loss is not None else None,
            target_1=round(target_1, 0) if target_1 is not None else None,
            target_2=round(target_2, 0) if target_2 is not None else None,
            risk_pts=round(risk_pts, 0) if risk_pts is not None else None,
            reward_1_pts=round(reward_1_pts, 0) if reward_1_pts is not None else None,
            risk_reward_1=round(risk_reward_1, 2) if risk_reward_1 is not None else None,
            confluences=confluences, confidence=confidence, valid=valid,
            description=desc,
        )

    except Exception as e:
        print(f"[mean_reversion_atr] Error: {e}")
        return None


def render_mean_reversion_setup(mr: Optional[MeanReversionSetup]):
    import streamlit as st

    st.markdown("**\U0001F3AF ATR Mean Reversion**")
    if mr is None:
        st.caption("Mean reversion setup unavailable")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Extension", f"{mr.extension_atr:+.1f} ATR" if mr.extension_atr is not None else "n/a",
               help=f"vs {mr.anchor_label} ({mr.anchor:,.0f})" if mr.anchor else None)
    c2.metric("ATR(14)", f"{mr.atr:,.0f} pts" if mr.atr is not None else "n/a")
    c3.metric("R:R (T1)", f"{mr.risk_reward_1:.2f}" if mr.risk_reward_1 is not None else "n/a")

    if mr.valid:
        conf_colors = {"HIGH": "#2d9e2d", "MEDIUM": "#e6a817", "LOW": "#888888"}
        col = conf_colors.get(mr.confidence, "#888888")
        dir_col = "#2d9e2d" if mr.direction == "LONG" else "#c9302c"
        st.markdown(
            f"<div style='padding:8px;border-radius:6px;background:{dir_col}22;"
            f"border-left:3px solid {dir_col};margin:6px 0'>"
            f"<b style='color:{dir_col}'>{'\U0001F7E2' if mr.direction=='LONG' else '\U0001F534'} "
            f"{mr.direction} FADE SETUP</b> "
            f"<span style='float:right;font-size:0.75em;padding:2px 6px;border-radius:3px;"
            f"background:{col}33;color:{col};font-weight:bold'>{mr.confidence} CONFIDENCE</span><br>"
            f"<span style='font-size:0.88em'>{mr.description}</span></div>",
            unsafe_allow_html=True,
        )
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Entry", f"{mr.entry:,.0f}")
        e2.metric("Stop", f"{mr.stop_loss:,.0f}")
        e3.metric("Target 1", f"{mr.target_1:,.0f}")
        e4.metric("Target 2", f"{mr.target_2:,.0f}")
    else:
        st.caption(mr.description)
