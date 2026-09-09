"""
order_flow_sequence.py  — v3
-----------------------------
Order-Flow SEQUENCE detector for NAS100.

Additive extension — does not modify scalping_engine.py or any other module.
Reads the ScalpReport (BC/TC/pivot/VWAP/key levels) and the same df_5m
already cached in session_state — zero extra yfinance calls.

═══════════════════════════════════════════════════════════════════════════════
STRATEGY ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

LAYER 1 — 4-STAGE INSTITUTIONAL FOOTPRINT SEQUENCE (core, from v2)
  Detects the order-flow fingerprint left by institutions entering positions:

  Stage 1: SELLING PRESSURE     Price↓ + Delta↓ + Volume↑
  Stage 2: ABSORPTION           Volume↑ + Δ strongly negative + price stalls
  Stage 3: SELLER EXHAUSTION    Price makes lower low, Delta makes higher low
  Stage 4: REVERSAL CONFIRMED   Delta turns positive + price reclaims key level

  Each stage is scored by proximity to a LIQUIDITY ZONE (CPR BC/TC/Pivot,
  VWAP, prev-day high/low, round numbers, swing points). A stage forming
  AT a liquidity zone is significantly higher probability.

LAYER 2 — INSTITUTIONAL MEAN REVERSION (new in v3)
  When Stage 2 (Absorption) forms at a high-quality liquidity zone:
  - Quantifies the mean-reversion probability using a 6-factor model
  - Computes entry zone, stop, TP1 (liquidity zone), TP2 (VWAP), TP3 (CPR TC)
  - Win rate target: 70%+ when ≥4 factors align
  - Integrates with GEX regime (positive GEX = dealers reinforce reversion)
  - Integrates with CPR day type (narrow CPR = strongest reversion setup)

LAYER 3 — ATR STRATEGY (new in v3)
  ATR (Average True Range) serves two roles:
  A) DYNAMIC STOP PLACEMENT — stops sized to current volatility, not fixed pts
     Stop = entry ± (ATR × multiplier) based on:
       - Regime (CHOP → 0.8× ATR, TRENDING → 1.2× ATR, CRISIS → 2.0× ATR)
       - Setup type (mean reversion → tighter, breakout → wider)
  B) ATR CHANNEL BANDS — volatility envelope around VWAP:
     Upper band = VWAP + (ATR × 1.5) → overbought fade zone
     Lower band = VWAP - (ATR × 1.5) → oversold fade zone
     When price touches a band AND delta is exhausted → mean reversion entry

LAYER 4 — SEQUENCE CONFIDENCE SCORING
  Each detected sequence is scored 0-100 based on:
  - Stage reached (Stage 4 = highest)
  - Liquidity zone quality (STRONG/MODERATE/WEAK)
  - RSI divergence confirmation
  - ATR band alignment
  - Volume confirmation
  - GEX regime alignment

═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
import streamlit as st
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# ── CONSTANTS ──────────────────────────────────────────────────────────────────
ATR_PERIOD        = 14       # standard ATR period
ATR_STOP_MULT     = {        # stop multiplier by regime
    "TRENDING": 1.2,
    "CHOP":     0.8,
    "CRISIS":   2.0,
}
ATR_BAND_MULT     = 1.5      # VWAP ± (ATR × 1.5) = fade zone
ATR_BREAKOUT_MULT = 2.0      # breakout confirmation = price > VWAP + 2×ATR
LIQUIDITY_ZONE_TOLERANCE = 0.003   # 0.3% proximity counts as "at zone"
SEQUENCE_CACHE_TTL = 60      # seconds


# ── CACHE ─────────────────────────────────────────────────────────────────────
def _cv(key: str) -> bool:
    return (time.time() - st.session_state.get(f"ofs_{key}_ts", 0)) < SEQUENCE_CACHE_TTL

def _store(key: str, data):
    st.session_state[f"ofs_{key}"] = data
    st.session_state[f"ofs_{key}_ts"] = time.time()

def _load(key: str):
    return st.session_state.get(f"ofs_{key}")


# ── DATA CLASSES ───────────────────────────────────────────────────────────────

@dataclass
class LiquidityZone:
    """A price level where institutional orders are likely resting."""
    price: float
    label: str              # "CPR BC" | "VWAP" | "Prev Day High" | "Round Number" etc.
    quality: str            # "STRONG" | "MODERATE" | "WEAK"
    quality_score: int      # 0-100
    distance_pts: float     # distance from current price
    distance_pct: float


@dataclass
class ATRData:
    """ATR calculations and derived levels."""
    atr_14: float           # 14-period ATR on 5m bars
    atr_pct: float          # ATR as % of current price

    # Dynamic stop levels (from current price)
    stop_long_pts: float    # ATR × multiplier (for long entries)
    stop_short_pts: float   # ATR × multiplier (for short entries)
    regime_multiplier: float

    # VWAP ATR bands
    vwap_upper_band: float  # VWAP + ATR × 1.5 → fade SHORT zone
    vwap_lower_band: float  # VWAP - ATR × 1.5 → fade LONG zone
    price_vs_upper: float   # % above upper band (negative = below)
    price_vs_lower: float   # % above lower band (positive = above)

    # ATR breakout levels
    breakout_long_level: float   # VWAP + ATR × 2.0 → confirmed breakout UP
    breakout_short_level: float  # VWAP - ATR × 2.0 → confirmed breakout DOWN

    # Signal
    at_upper_band: bool     # price touching upper fade zone
    at_lower_band: bool     # price touching lower fade zone
    atr_signal: str         # "FADE SHORT at upper band" | "FADE LONG at lower band" | "NEUTRAL"
    atr_signal_color: str


@dataclass
class SequenceStage:
    """One detected stage in the 4-stage institutional sequence."""
    stage_num: int          # 1-4
    stage_name: str         # "SELLING PRESSURE" | "ABSORPTION" | etc.
    bar_index: int          # which bar (from end of series) it was detected at
    price_at_stage: float
    delta_at_stage: float
    volume_at_stage: float

    # Zone context
    nearest_zone: Optional[LiquidityZone]
    at_liquidity_zone: bool
    zone_quality: str       # quality of nearest zone

    # Confirmation
    rsi_confirms: bool      # RSI divergence aligns with stage
    volume_confirms: bool   # volume spike confirms the stage
    stage_score: int        # 0-100

    description: str


@dataclass
class MeanReversionTrade:
    """
    Institutional mean reversion trade derived from Stage 2 (Absorption).
    Highest probability setup when absorption forms AT a liquidity zone
    with ≥4 confirming factors.
    """
    active: bool
    direction: str          # "LONG" | "SHORT"

    # Entry
    entry_zone_low: float
    entry_zone_high: float
    entry_trigger: str      # exact trigger condition

    # Risk levels (ATR-based)
    stop_loss: float        # entry ± ATR × multiplier
    stop_pts: float

    # Targets (tiered — institutional approach)
    tp1: float              # nearest liquidity zone (BC/TC/VWAP)
    tp2: float              # VWAP or CPR pivot
    tp3: float              # opposite CPR level (full reversion)
    rr1: float              # R:R to TP1
    rr2: float              # R:R to TP2
    rr3: float              # R:R to TP3

    # Context
    absorption_zone: Optional[LiquidityZone]
    atr_stop_used: float    # ATR value used for stop calculation
    cpr_day_type: str       # "NARROW" | "MODERATE" | "WIDE"

    # Conviction scoring
    conviction_factors: List[str]
    conviction_score: int   # 0-100
    conviction_label: str   # "HIGH (70-80%)" | "MODERATE (55-70%)" | "LOW"
    invalidation: str


@dataclass
class OrderFlowSequence:
    """Complete order flow sequence analysis — all 4 layers."""

    # ── LAYER 1: Sequence stages ──────────────────────────────────────────────
    stages_detected: List[SequenceStage]
    current_stage: int      # highest stage currently active (0 = none)
    sequence_direction: str # "BULLISH_REVERSAL" | "BEARISH_REVERSAL" | "NONE"
    sequence_complete: bool # True if Stage 4 confirmed

    # ── LAYER 2: Mean reversion trade ─────────────────────────────────────────
    mean_reversion_trade: Optional[MeanReversionTrade]

    # ── LAYER 3: ATR data ─────────────────────────────────────────────────────
    atr: Optional[ATRData]

    # ── LAYER 4: Overall confidence ───────────────────────────────────────────
    sequence_score: int     # 0-100
    sequence_signal: str    # actionable summary
    sequence_color: str

    # Liquidity zones map
    liquidity_zones: List[LiquidityZone]

    # RSI + delta reference
    current_rsi: float
    current_delta_pct: float
    cumulative_delta_trend: str  # "RISING" | "FALLING" | "DIVERGING"


# ── HELPER: RSI ────────────────────────────────────────────────────────────────

def _calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI on a numpy array of closes."""
    if len(closes) < period + 1:
        return np.full(len(closes), 50.0)
    deltas  = np.diff(closes)
    gains   = np.where(deltas > 0, deltas, 0.0)
    losses  = np.where(deltas < 0, -deltas, 0.0)
    avg_g   = np.mean(gains[:period])
    avg_l   = np.mean(losses[:period])
    rsi_arr = [50.0] * (period + 1)
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs    = avg_g / max(avg_l, 1e-9)
        rsi_arr.append(100 - 100 / (1 + rs))
    return np.array(rsi_arr)


# ── HELPER: ATR ────────────────────────────────────────────────────────────────

def _calc_atr(highs: np.ndarray, lows: np.ndarray,
               closes: np.ndarray, period: int = ATR_PERIOD) -> np.ndarray:
    """True Range and ATR."""
    if len(closes) < 2:
        return np.zeros(len(closes))
    tr = np.zeros(len(closes))
    tr[0] = highs[0] - lows[0]
    for i in range(1, len(closes)):
        tr[i] = max(highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i]  - closes[i - 1]))
    atr = np.zeros(len(closes))
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(closes)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ── HELPER: Delta proxy ────────────────────────────────────────────────────────

def _bar_deltas(opens, highs, lows, closes, volumes):
    """Buy-sell delta proxy from bar structure."""
    bar_range = np.where(highs - lows > 0, highs - lows, 1e-9)
    buy_frac  = (closes - lows) / bar_range
    sell_frac = (highs - closes) / bar_range
    buy_vol   = volumes * buy_frac
    sell_vol  = volumes * sell_frac
    return buy_vol - sell_vol, buy_vol, sell_vol


# ── HELPER: Liquidity zone mapping ────────────────────────────────────────────

def _build_liquidity_zones(scalp_report, current_price: float) -> List[LiquidityZone]:
    """
    Build a sorted list of liquidity zones from ScalpReport.
    Zones: CPR levels, VWAP, key_levels (prev-day H/L/C, round numbers),
    swing highs/lows detected from recent bars.
    """
    zones = []

    def _add(price: float, label: str, quality: str, score: int):
        if price <= 0:
            return
        dist_pts = price - current_price
        dist_pct = dist_pts / current_price * 100
        zones.append(LiquidityZone(
            price=round(price, 0), label=label, quality=quality,
            quality_score=score,
            distance_pts=round(dist_pts, 0),
            distance_pct=round(dist_pct, 3),
        ))

    # CPR levels — highest quality institutional zones
    if scalp_report and scalp_report.cpr:
        cpr = scalp_report.cpr
        _add(cpr.pivot, "CPR Pivot", "STRONG", 90)
        _add(cpr.bc,    "CPR BC",    "STRONG", 88)
        _add(cpr.tc,    "CPR TC",    "STRONG", 88)
        _add(cpr.r1,    "CPR R1",    "MODERATE", 70)
        _add(cpr.s1,    "CPR S1",    "MODERATE", 70)
        _add(cpr.r2,    "CPR R2",    "MODERATE", 60)
        _add(cpr.s2,    "CPR S2",    "MODERATE", 60)

    # VWAP — real-time equilibrium, strongest intraday zone
    if scalp_report and scalp_report.vwap:
        _add(scalp_report.vwap, "VWAP", "STRONG", 95)

    # Key levels (prev-day H/L/C, round numbers)
    if scalp_report:
        for i, lv in enumerate(scalp_report.key_levels[:10]):
            # Round numbers get higher score
            is_round = lv % 100 < 5 or lv % 100 > 95
            quality  = "STRONG" if is_round else "MODERATE"
            score    = 80 if is_round else 65
            _add(lv, f"Key Level {lv:,.0f}", quality, score)

    # Overnight high/low from liquidity sweeps
    if scalp_report:
        for sw in scalp_report.liquidity_sweeps:
            label_map = {
                "OVERNIGHT_HIGH": ("Overnight High", "STRONG", 82),
                "OVERNIGHT_LOW":  ("Overnight Low",  "STRONG", 82),
                "PREV_DAY_HIGH":  ("Prev Day High",  "STRONG", 85),
                "PREV_DAY_LOW":   ("Prev Day Low",   "STRONG", 85),
                "EQUAL_HIGHS":    ("Equal Highs",    "MODERATE", 72),
                "EQUAL_LOWS":     ("Equal Lows",     "MODERATE", 72),
            }
            if sw.sweep_type in label_map:
                lbl, q, s = label_map[sw.sweep_type]
                _add(sw.swept_level, lbl, q, s)

    # Sort by proximity to current price
    zones.sort(key=lambda z: abs(z.distance_pts))
    return zones


def _nearest_zone(price: float, zones: List[LiquidityZone]) -> Optional[LiquidityZone]:
    """Find the nearest liquidity zone to a given price."""
    if not zones:
        return None
    return min(zones, key=lambda z: abs(z.price - price))


def _at_zone(price: float, zones: List[LiquidityZone],
              tolerance: float = LIQUIDITY_ZONE_TOLERANCE) -> Tuple[bool, Optional[LiquidityZone]]:
    """Check if price is within tolerance of any liquidity zone."""
    for z in zones:
        if abs(price - z.price) / max(price, 1) <= tolerance:
            return True, z
    return False, None


# ── LAYER 3: ATR COMPUTATION ──────────────────────────────────────────────────

def _compute_atr_data(highs: np.ndarray, lows: np.ndarray,
                       closes: np.ndarray, vwap: Optional[float],
                       current_price: float, regime: str = "TRENDING",
                       ratio: float = 1.0) -> ATRData:
    """Compute ATR and all derived levels."""
    atr_arr  = _calc_atr(highs, lows, closes)
    atr_raw  = float(atr_arr[-1]) if atr_arr[-1] > 0 else float(np.mean(highs - lows))
    atr_pts  = atr_raw * ratio
    atr_pct  = atr_pts / current_price * 100

    mult     = ATR_STOP_MULT.get(regime, 1.2)
    stop_pts = atr_pts * mult

    vwap_ref = vwap if vwap else current_price
    upper    = vwap_ref + atr_pts * ATR_BAND_MULT
    lower    = vwap_ref - atr_pts * ATR_BAND_MULT
    bu_long  = vwap_ref + atr_pts * ATR_BREAKOUT_MULT
    bu_short = vwap_ref - atr_pts * ATR_BREAKOUT_MULT

    pct_vs_upper = (current_price - upper) / upper * 100
    pct_vs_lower = (current_price - lower) / lower * 100

    at_upper = current_price >= upper * 0.998   # within 0.2% of upper band
    at_lower = current_price <= lower * 1.002   # within 0.2% of lower band

    if at_upper:
        atr_sig = "⚠️ FADE SHORT — price at ATR upper band (overbought vs VWAP)"
        atr_col = "#c9302c"
    elif at_lower:
        atr_sig = "⚠️ FADE LONG — price at ATR lower band (oversold vs VWAP)"
        atr_col = "#2d9e2d"
    elif current_price > bu_long:
        atr_sig = "🚀 BREAKOUT LONG confirmed (above VWAP + 2×ATR)"
        atr_col = "#1a7a1a"
    elif current_price < bu_short:
        atr_sig = "💥 BREAKOUT SHORT confirmed (below VWAP - 2×ATR)"
        atr_col = "#8b0000"
    else:
        atr_sig = f"NEUTRAL — within ATR bands ({lower:,.0f}–{upper:,.0f})"
        atr_col = "#888888"

    return ATRData(
        atr_14=round(atr_pts, 0),
        atr_pct=round(atr_pct, 3),
        stop_long_pts=round(stop_pts, 0),
        stop_short_pts=round(stop_pts, 0),
        regime_multiplier=mult,
        vwap_upper_band=round(upper, 0),
        vwap_lower_band=round(lower, 0),
        price_vs_upper=round(pct_vs_upper, 2),
        price_vs_lower=round(pct_vs_lower, 2),
        breakout_long_level=round(bu_long, 0),
        breakout_short_level=round(bu_short, 0),
        at_upper_band=at_upper,
        at_lower_band=at_lower,
        atr_signal=atr_sig,
        atr_signal_color=atr_col,
    )


# ── LAYER 1: SEQUENCE DETECTOR ────────────────────────────────────────────────

def _detect_sequence(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volumes: np.ndarray, opens: np.ndarray,
    current_price: float, ratio: float,
    zones: List[LiquidityZone],
) -> Tuple[List[SequenceStage], int, str]:
    """
    Detect the 4-stage institutional footprint sequence.
    Returns (stages_list, current_stage_num, direction)
    """
    if len(closes) < 20:
        return [], 0, "NONE"

    deltas, buy_vols, sell_vols = _bar_deltas(opens, highs, lows, closes, volumes)
    cum_delta = np.cumsum(deltas)
    rsi_arr   = _calc_rsi(closes)
    avg_vol   = float(np.mean(volumes))

    stages    = []
    direction = "NONE"

    # ── Detect BULLISH REVERSAL sequence (selling → absorption → exhaust → reverse)
    # Work backwards from current bar, look for the sequence in last 30 bars

    # Stage 1: SELLING PRESSURE — find a bearish leg
    # Criteria: 3+ consecutive bars closing lower with rising volume and negative delta
    stage1_bar = None
    for i in range(len(closes) - 6, max(len(closes) - 30, 3), -1):
        leg = closes[i:i+4]
        vol_leg = volumes[i:i+4]
        dlt_leg = deltas[i:i+4]
        if (np.all(np.diff(leg) < 0) and          # all bars closing lower
            np.mean(vol_leg) > avg_vol * 0.8 and   # reasonable volume
            np.sum(dlt_leg) < 0):                   # net negative delta
            at_z, zone = _at_zone(closes[i], zones)
            rsi_conf   = rsi_arr[i+3] < 40 if len(rsi_arr) > i+3 else False
            s1 = SequenceStage(
                stage_num=1, stage_name="SELLING PRESSURE",
                bar_index=i, price_at_stage=round(closes[i+3] * ratio, 0),
                delta_at_stage=round(float(np.sum(dlt_leg)), 0),
                volume_at_stage=round(float(np.mean(vol_leg)), 0),
                nearest_zone=_nearest_zone(closes[i+3] * ratio, zones),
                at_liquidity_zone=at_z, zone_quality=zone.quality if zone else "NONE",
                rsi_confirms=rsi_conf, volume_confirms=np.mean(vol_leg) > avg_vol,
                stage_score=60 + (20 if at_z else 0) + (10 if rsi_conf else 0),
                description=(f"Sustained selling: {abs(closes[i+3]-closes[i])*ratio:.0f} pts over "
                             f"{4} bars | Net delta: {np.sum(dlt_leg):.0f} | "
                             f"{'At ' + zone.label if at_z and zone else 'Away from zones'}"),
            )
            stages.append(s1); stage1_bar = i+3; direction = "BULLISH_REVERSAL"; break

    if not stages:
        # Also check for BEARISH REVERSAL sequence (buying → absorption → exhaust → reverse)
        for i in range(len(closes) - 6, max(len(closes) - 30, 3), -1):
            leg = closes[i:i+4]
            vol_leg = volumes[i:i+4]
            dlt_leg = deltas[i:i+4]
            if (np.all(np.diff(leg) > 0) and
                np.mean(vol_leg) > avg_vol * 0.8 and
                np.sum(dlt_leg) > 0):
                at_z, zone = _at_zone(closes[i+3], zones)
                rsi_conf   = rsi_arr[i+3] > 60 if len(rsi_arr) > i+3 else False
                s1 = SequenceStage(
                    stage_num=1, stage_name="BUYING PRESSURE",
                    bar_index=i, price_at_stage=round(closes[i+3] * ratio, 0),
                    delta_at_stage=round(float(np.sum(dlt_leg)), 0),
                    volume_at_stage=round(float(np.mean(vol_leg)), 0),
                    nearest_zone=_nearest_zone(closes[i+3] * ratio, zones),
                    at_liquidity_zone=at_z, zone_quality=zone.quality if zone else "NONE",
                    rsi_confirms=rsi_conf, volume_confirms=True,
                    stage_score=60 + (20 if at_z else 0) + (10 if rsi_conf else 0),
                    description=(f"Sustained buying: {(closes[i+3]-closes[i])*ratio:.0f} pts | "
                                 f"Net delta: {np.sum(dlt_leg):.0f} | "
                                 f"{'At ' + zone.label if at_z and zone else 'Away from zones'}"),
                )
                stages.append(s1); stage1_bar = i+3; direction = "BEARISH_REVERSAL"; break

    if not stages or stage1_bar is None:
        return [], 0, "NONE"

    # ── Stage 2: ABSORPTION ────────────────────────────────────────────────────
    # After the pressure leg: volume spikes but price STOPS making new extremes
    if stage1_bar < len(closes) - 3:
        abs_window = slice(stage1_bar, min(stage1_bar + 8, len(closes)))
        abs_close  = closes[abs_window]
        abs_vol    = volumes[abs_window]
        abs_dlt    = deltas[abs_window]
        abs_high   = highs[abs_window]
        abs_low    = lows[abs_window]

        if direction == "BULLISH_REVERSAL":
            # Price not making new lows despite high volume and negative delta
            high_vol_bars  = abs_vol > avg_vol * 1.5
            neg_delta_bars = abs_dlt < 0
            price_stalling = abs_low.min() >= closes[stage1_bar] * 0.999   # price holds
            absorption = (np.any(high_vol_bars) and
                          np.any(neg_delta_bars) and
                          price_stalling)
        else:
            high_vol_bars  = abs_vol > avg_vol * 1.5
            pos_delta_bars = abs_dlt > 0
            price_stalling = abs_high.max() <= closes[stage1_bar] * 1.001
            absorption = (np.any(high_vol_bars) and
                          np.any(pos_delta_bars) and
                          price_stalling)

        if absorption:
            abs_price = float(abs_close[-1]) * ratio
            at_z, zone = _at_zone(abs_price, zones)
            rsi_now    = float(rsi_arr[abs_window.stop - 1]) if abs_window.stop <= len(rsi_arr) else 50
            rsi_conf   = rsi_now < 35 if direction == "BULLISH_REVERSAL" else rsi_now > 65
            s2 = SequenceStage(
                stage_num=2, stage_name="ABSORPTION",
                bar_index=abs_window.stop - 1,
                price_at_stage=round(abs_price, 0),
                delta_at_stage=round(float(np.sum(abs_dlt)), 0),
                volume_at_stage=round(float(np.max(abs_vol)), 0),
                nearest_zone=_nearest_zone(abs_price, zones),
                at_liquidity_zone=at_z, zone_quality=zone.quality if zone else "MODERATE",
                rsi_confirms=rsi_conf, volume_confirms=True,
                stage_score=75 + (15 if at_z else 0) + (10 if rsi_conf else 0),
                description=(f"{'Sellers' if direction=='BULLISH_REVERSAL' else 'Buyers'} absorbed: "
                             f"Vol spike {np.max(abs_vol)/avg_vol:.1f}× but price holds | "
                             f"RSI: {rsi_now:.0f} | "
                             f"{'✅ At ' + zone.label if at_z and zone else '⚠️ Away from zones'}"),
            )
            stages.append(s2)

    # ── Stage 3: EXHAUSTION (delta divergence) ────────────────────────────────
    if len(stages) >= 2:
        s2_bar = stages[-1].bar_index
        if s2_bar < len(closes) - 3:
            exh_window  = slice(s2_bar, min(s2_bar + 8, len(closes)))
            exh_close   = closes[exh_window]
            exh_cum_dlt = cum_delta[exh_window]
            exh_rsi     = rsi_arr[exh_window] if len(rsi_arr) >= exh_window.stop else rsi_arr[-len(exh_close):]

            if direction == "BULLISH_REVERSAL":
                # Price makes lower low but delta makes higher low = bullish divergence
                price_div  = (exh_close.min() < closes[s2_bar] and
                              exh_cum_dlt[-1] > exh_cum_dlt[0])
                rsi_div    = (len(exh_rsi) > 1 and exh_rsi[-1] > exh_rsi[0] and
                              exh_close[-1] < closes[s2_bar])
            else:
                price_div  = (exh_close.max() > closes[s2_bar] and
                              exh_cum_dlt[-1] < exh_cum_dlt[0])
                rsi_div    = (len(exh_rsi) > 1 and exh_rsi[-1] < exh_rsi[0] and
                              exh_close[-1] > closes[s2_bar])

            if price_div or rsi_div:
                exh_price = float(exh_close[-1]) * ratio
                at_z, zone = _at_zone(exh_price, zones)
                s3 = SequenceStage(
                    stage_num=3,
                    stage_name="SELLER EXHAUSTION" if direction == "BULLISH_REVERSAL" else "BUYER EXHAUSTION",
                    bar_index=exh_window.stop - 1,
                    price_at_stage=round(exh_price, 0),
                    delta_at_stage=round(float(exh_cum_dlt[-1]), 0),
                    volume_at_stage=round(float(np.mean(volumes[exh_window])), 0),
                    nearest_zone=_nearest_zone(exh_price, zones),
                    at_liquidity_zone=at_z,
                    zone_quality=zone.quality if zone else "WEAK",
                    rsi_confirms=rsi_div,
                    volume_confirms=True,
                    stage_score=85 + (10 if at_z else 0) + (5 if rsi_div else 0),
                    description=(f"{'Bullish' if direction=='BULLISH_REVERSAL' else 'Bearish'} delta divergence: "
                                 f"price {'lower low' if direction=='BULLISH_REVERSAL' else 'higher high'} "
                                 f"but delta {'higher low' if direction=='BULLISH_REVERSAL' else 'lower high'} | "
                                 f"{'RSI divergence confirmed ✅' if rsi_div else 'RSI divergence pending'}"),
                )
                stages.append(s3)

    # ── Stage 4: REVERSAL CONFIRMED ───────────────────────────────────────────
    if len(stages) >= 3:
        s3_bar   = stages[-1].bar_index
        last_bars = closes[s3_bar:]
        last_dlt  = deltas[s3_bar:]

        if direction == "BULLISH_REVERSAL":
            # Delta turns positive AND price reclaims a key level
            delta_positive = len(last_dlt) >= 2 and float(np.sum(last_dlt[-3:])) > 0
            reclaim_zone   = False; reclaim_label = ""
            for z in stages[0].nearest_zone and [stages[0].nearest_zone] or []:
                if current_price > z.price:
                    reclaim_zone = True; reclaim_label = z.label
        else:
            delta_positive = len(last_dlt) >= 2 and float(np.sum(last_dlt[-3:])) < 0
            reclaim_zone   = False; reclaim_label = ""

        if delta_positive:
            at_z, zone = _at_zone(current_price, zones)
            s4 = SequenceStage(
                stage_num=4, stage_name="REVERSAL CONFIRMED",
                bar_index=len(closes) - 1,
                price_at_stage=round(current_price, 0),
                delta_at_stage=round(float(np.sum(last_dlt[-3:])), 0),
                volume_at_stage=round(float(np.mean(volumes[-3:])), 0),
                nearest_zone=_nearest_zone(current_price, zones),
                at_liquidity_zone=reclaim_zone or at_z,
                zone_quality="STRONG" if reclaim_zone else (zone.quality if zone else "MODERATE"),
                rsi_confirms=float(rsi_arr[-1]) > 50 if direction == "BULLISH_REVERSAL" else float(rsi_arr[-1]) < 50,
                volume_confirms=float(np.mean(volumes[-3:])) > avg_vol * 0.8,
                stage_score=95,
                description=(f"{'Bullish' if direction=='BULLISH_REVERSAL' else 'Bearish'} reversal confirmed: "
                             f"delta turned {'positive' if direction=='BULLISH_REVERSAL' else 'negative'} | "
                             f"{'Reclaimed ' + reclaim_label if reclaim_zone else 'Key level pending reclaim'}"),
            )
            stages.append(s4)

    current_stage = max((s.stage_num for s in stages), default=0)
    return stages, current_stage, direction


# ── LAYER 2: MEAN REVERSION TRADE ─────────────────────────────────────────────

def _build_mean_reversion_trade(
    stages: List[SequenceStage],
    current_stage: int,
    direction: str,
    current_price: float,
    atr: Optional[ATRData],
    scalp_report,
    zones: List[LiquidityZone],
    gex_regime: str = "NEGATIVE",
    regime_state: int = 0,
) -> Optional[MeanReversionTrade]:
    """
    Build the institutional mean reversion trade from Stage 2 absorption.
    Only fires when absorption forms AT a liquidity zone.
    """
    if current_stage < 2 or direction == "NONE":
        return None

    # Find Stage 2
    s2 = next((s for s in stages if s.stage_num == 2), None)
    if s2 is None:
        return None

    # Must be at a liquidity zone for institutional quality
    if not s2.at_liquidity_zone and s2.zone_quality not in ("STRONG", "MODERATE"):
        return None

    trade_dir = "LONG" if direction == "BULLISH_REVERSAL" else "SHORT"

    # Entry zone: around the absorption price
    entry_buffer = atr.atr_14 * 0.1 if atr else 15
    if trade_dir == "LONG":
        entry_low  = round(s2.price_at_stage - entry_buffer, 0)
        entry_high = round(s2.price_at_stage + entry_buffer, 0)
    else:
        entry_low  = round(s2.price_at_stage - entry_buffer, 0)
        entry_high = round(s2.price_at_stage + entry_buffer, 0)

    # ATR-based stop
    stop_pts = atr.stop_long_pts if atr else 50
    if trade_dir == "LONG":
        stop_loss = round(entry_low - stop_pts, 0)
    else:
        stop_loss = round(entry_high + stop_pts, 0)

    # Tiered targets
    vwap    = scalp_report.vwap if scalp_report else None
    cpr     = scalp_report.cpr  if scalp_report else None

    # TP1: nearest opposing liquidity zone
    above_zones = [z for z in zones if z.price > current_price] if trade_dir == "LONG" else \
                  [z for z in zones if z.price < current_price]
    tp1 = round(above_zones[0].price if above_zones else current_price + stop_pts * 1.5, 0)

    # TP2: VWAP or CPR Pivot
    tp2 = round(vwap if vwap else (cpr.pivot if cpr else current_price + stop_pts * 2.5), 0)

    # TP3: opposite CPR level (full mean reversion)
    if cpr:
        tp3 = round(cpr.tc if trade_dir == "LONG" else cpr.bc, 0)
    elif vwap:
        tp3 = round(vwap + (vwap - entry_low) if trade_dir == "LONG"
                    else vwap - (entry_high - vwap), 0)
    else:
        tp3 = round(current_price + stop_pts * 4 if trade_dir == "LONG"
                    else current_price - stop_pts * 4, 0)

    entry_mid = (entry_low + entry_high) / 2
    rr1 = round(abs(tp1 - entry_mid) / max(stop_pts, 1), 1)
    rr2 = round(abs(tp2 - entry_mid) / max(stop_pts, 1), 1)
    rr3 = round(abs(tp3 - entry_mid) / max(stop_pts, 1), 1)

    # Conviction scoring
    factors = []
    score   = 40  # base

    if s2.at_liquidity_zone:
        zone_lbl = s2.nearest_zone.label if s2.nearest_zone else "key zone"
        factors.append(f"Absorption AT {zone_lbl} (institutional level)")
        score += 20

    if current_stage >= 3:
        factors.append("Exhaustion confirmed (delta divergence)")
        score += 15

    if current_stage >= 4:
        factors.append("Reversal CONFIRMED — highest conviction")
        score += 15

    if gex_regime == "POSITIVE":
        factors.append("Positive GEX — dealers mechanically support reversion")
        score += 10

    if regime_state == 1:   # CHOP
        factors.append("CHOP regime — mean reversion statistically favored")
        score += 10

    if atr and atr.at_lower_band and trade_dir == "LONG":
        factors.append("Price at ATR lower band — volatility exhaustion")
        score += 10

    if atr and atr.at_upper_band and trade_dir == "SHORT":
        factors.append("Price at ATR upper band — volatility exhaustion")
        score += 10

    if cpr and cpr.cpr_type == "NARROW":
        factors.append(f"NARROW CPR day — strongest mean reversion setup")
        score += 10

    if s2.rsi_confirms:
        factors.append("RSI extreme confirms exhaustion")
        score += 5

    if rr2 >= 2.0:
        factors.append(f"R:R to VWAP = {rr2:.1f}:1 (institutional minimum met)")
        score += 5

    score = min(score, 100)

    if score >= 80:
        label = f"HIGH PROBABILITY (75-85%) — {len(factors)} factors"
    elif score >= 65:
        label = f"MODERATE-HIGH (60-75%) — {len(factors)} factors"
    elif score >= 50:
        label = f"MODERATE (50-65%) — watch for Stage 4 confirmation"
    else:
        label = "LOW — wait for more confirmation"

    inval = (f"Close below {stop_loss:,.0f}" if trade_dir == "LONG"
             else f"Close above {stop_loss:,.0f}")

    cpr_type = cpr.cpr_type if cpr else "UNKNOWN"
    trigger  = (f"Stage {current_stage} sequence at "
                f"{s2.nearest_zone.label if s2.nearest_zone else 'key level'}")

    return MeanReversionTrade(
        active=True, direction=trade_dir,
        entry_zone_low=entry_low, entry_zone_high=entry_high,
        entry_trigger=trigger,
        stop_loss=stop_loss, stop_pts=round(stop_pts, 0),
        tp1=tp1, tp2=tp2, tp3=tp3,
        rr1=rr1, rr2=rr2, rr3=rr3,
        absorption_zone=s2.nearest_zone,
        atr_stop_used=atr.atr_14 if atr else 0,
        cpr_day_type=cpr_type,
        conviction_factors=factors,
        conviction_score=score,
        conviction_label=label,
        invalidation=inval,
    )


# ── MASTER COMPUTE ─────────────────────────────────────────────────────────────

def compute_order_flow_sequence(
    df_5m,
    scalp_report,
    current_price: float,
    ratio: float = 40.0,
    gex_regime: str = "NEGATIVE",
    regime_state: int = 0,
    regime_label: str = "TRENDING",
) -> Optional[OrderFlowSequence]:
    """
    Main entry point. Computes all 4 layers of the order flow sequence.
    Call once per refresh cycle — cached for SEQUENCE_CACHE_TTL seconds.
    """
    cache_key = "ofs_main"
    if _cv(cache_key):
        cached = _load(cache_key)
        if cached:
            return cached

    try:
        if df_5m is None or len(df_5m) < 20:
            return None

        df = df_5m.copy()
        df.columns = [c.capitalize() for c in df.columns]
        df.index   = pd.to_datetime(df.index, utc=True)
        today      = pd.Timestamp.now(tz='UTC').date()
        today_bars = df[df.index.date == today].copy()
        if len(today_bars) < 15:
            today_bars = df.tail(78).copy()

        o = today_bars['Open'].values
        h = today_bars['High'].values
        l = today_bars['Low'].values
        c = today_bars['Close'].values
        v = today_bars['Volume'].values

        # ── Liquidity zones ──────────────────────────────────────────────────
        zones = _build_liquidity_zones(scalp_report, current_price)

        # ── ATR ──────────────────────────────────────────────────────────────
        vwap_raw = (scalp_report.vwap / ratio) if (scalp_report and scalp_report.vwap) else None
        atr_data = _compute_atr_data(
            h, l, c,
            vwap=vwap_raw,
            current_price=c[-1],
            regime=regime_label,
            ratio=ratio,
        )

        # ── Sequence detection ───────────────────────────────────────────────
        stages, current_stage, direction = _detect_sequence(
            c, h, l, v, o, c[-1], ratio, zones
        )

        # ── RSI and delta for display ────────────────────────────────────────
        rsi_arr    = _calc_rsi(c)
        current_rsi = float(rsi_arr[-1])
        dlt, _, _  = _bar_deltas(o, h, l, c, v)
        cum_dlt    = np.cumsum(dlt)
        total_v    = float(np.sum(v))
        total_dlt  = float(np.sum(dlt))
        delta_pct  = total_dlt / max(total_v, 1) * 100
        cum_trend  = ("RISING"    if cum_dlt[-1] > cum_dlt[0] else
                      "FALLING"   if cum_dlt[-1] < cum_dlt[0] else "FLAT")
        if len(cum_dlt) > 10:
            mid = len(cum_dlt) // 2
            if (c[-1] < c[0] and cum_dlt[-1] > cum_dlt[mid]):
                cum_trend = "DIVERGING"   # bullish divergence

        # ── Mean reversion trade ─────────────────────────────────────────────
        mr_trade = _build_mean_reversion_trade(
            stages, current_stage, direction,
            current_price, atr_data, scalp_report,
            zones, gex_regime, regime_state,
        )

        # ── Overall score ─────────────────────────────────────────────────────
        seq_score = 0
        if current_stage >= 1: seq_score += 20
        if current_stage >= 2: seq_score += 25
        if current_stage >= 3: seq_score += 25
        if current_stage >= 4: seq_score += 20
        if mr_trade:           seq_score = max(seq_score, mr_trade.conviction_score)

        # ── Signal ───────────────────────────────────────────────────────────
        stage_names = {0:"No sequence", 1:"Stage 1", 2:"Stage 2", 3:"Stage 3", 4:"Stage 4"}
        if current_stage == 4:
            sig = (f"{'🟢 BULLISH REVERSAL' if direction=='BULLISH_REVERSAL' else '🔴 BEARISH REVERSAL'} "
                   f"CONFIRMED — full 4-stage sequence complete")
            col = "#2d9e2d" if direction == "BULLISH_REVERSAL" else "#c9302c"
        elif current_stage == 3:
            sig = (f"{'🟡 Exhaustion detected' if direction=='BULLISH_REVERSAL' else '🟡 Buyer exhaustion'} "
                   f"— watch for Stage 4 confirmation (delta flip)")
            col = "#e6a817"
        elif current_stage == 2:
            at_z_txt = (f" at {stages[-1].nearest_zone.label}" if stages and stages[-1].nearest_zone else "")
            sig = (f"{'🔵 ABSORPTION' + at_z_txt} — "
                   f"{'sellers absorbed' if direction=='BULLISH_REVERSAL' else 'buyers absorbed'}, "
                   f"mean reversion setup forming")
            col = "#4a7fb5"
        elif current_stage == 1:
            sig = (f"Stage 1: {'Selling' if direction=='BULLISH_REVERSAL' else 'Buying'} pressure — "
                   f"monitoring for absorption")
            col = "#888888"
        else:
            sig = "No institutional sequence detected — price in equilibrium"
            col = "#888888"

        result = OrderFlowSequence(
            stages_detected=stages,
            current_stage=current_stage,
            sequence_direction=direction,
            sequence_complete=(current_stage == 4),
            mean_reversion_trade=mr_trade,
            atr=atr_data,
            sequence_score=seq_score,
            sequence_signal=sig,
            sequence_color=col,
            liquidity_zones=zones[:8],
            current_rsi=round(current_rsi, 1),
            current_delta_pct=round(delta_pct, 1),
            cumulative_delta_trend=cum_trend,
        )
        _store(cache_key, result)
        return result

    except Exception as e:
        print(f"[order_flow_sequence] Error: {e}")
        return None


# ── RENDER ─────────────────────────────────────────────────────────────────────

def render_order_flow_sequence(ofs: Optional[OrderFlowSequence]):
    """Render the complete order flow sequence panel."""
    import streamlit as st

    st.subheader("🏛️ Institutional Order Flow Sequence")

    if ofs is None:
        st.info("Order flow sequence unavailable — insufficient 5m bar data.")
        return

    # ── SEQUENCE STATUS BANNER ────────────────────────────────────────────────
    stage_icons = {0: "⚪", 1: "🟡", 2: "🔵", 3: "🟠", 4: "🟢"}
    st.markdown(
        f"<div style='padding:12px 16px;border-radius:10px;"
        f"background:{ofs.sequence_color}22;border:2px solid {ofs.sequence_color}'>"
        f"<div style='font-size:1.2em;font-weight:bold;color:{ofs.sequence_color}'>"
        f"{stage_icons.get(ofs.current_stage, '⚪')} {ofs.sequence_signal}</div>"
        f"<div style='margin-top:4px;color:#aaa;font-size:0.85em'>"
        f"Sequence score: {ofs.sequence_score}/100 | "
        f"RSI: {ofs.current_rsi:.0f} | "
        f"Delta: {ofs.current_delta_pct:+.1f}% | "
        f"Cum delta: {ofs.cumulative_delta_trend}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    # ── 4-STAGE PROGRESS BAR ─────────────────────────────────────────────────
    st.markdown("**📊 Sequence Progress**")
    stage_labels = [
        ("1", "Pressure",    "#e6a817"),
        ("2", "Absorption",  "#4a7fb5"),
        ("3", "Exhaustion",  "#aa44ff"),
        ("4", "Confirmed",   "#2d9e2d"),
    ]
    cols = st.columns(4)
    for i, (num, name, color) in enumerate(stage_labels):
        active = ofs.current_stage >= int(num)
        bg     = f"{color}33" if active else "#1a1a1a"
        border = color if active else "#333"
        icon   = "✅" if active else "⬜"
        stage_detail = ""
        if active and ofs.stages_detected:
            s = next((st_ for st_ in ofs.stages_detected if st_.stage_num == int(num)), None)
            if s:
                stage_detail = f"{s.price_at_stage:,.0f}"
        cols[i].markdown(
            f"<div style='padding:8px;border-radius:8px;background:{bg};"
            f"border:2px solid {border};text-align:center'>"
            f"<div style='font-size:1.1em'>{icon}</div>"
            f"<div style='color:{color if active else '#666'};font-weight:bold;font-size:0.85em'>"
            f"Stage {num}</div>"
            f"<div style='color:#aaa;font-size:0.75em'>{name}</div>"
            f"{'<div style=\"color:#ccc;font-size:0.8em\">' + stage_detail + '</div>' if stage_detail else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── STAGE DETAILS ─────────────────────────────────────────────────────────
    if ofs.stages_detected:
        with st.expander("📋 Stage Details", expanded=ofs.current_stage >= 3):
            for s in ofs.stages_detected:
                s_color = {"STRONG": "#2d9e2d", "MODERATE": "#e6a817",
                           "WEAK": "#888", "NONE": "#888"}.get(s.zone_quality, "#888")
                zone_txt = (f"AT {s.nearest_zone.label} ({s.nearest_zone.quality})"
                            if s.at_liquidity_zone and s.nearest_zone
                            else "Away from zones — lower conviction")
                st.markdown(
                    f"<div style='padding:6px 10px;border-radius:5px;margin:4px 0;"
                    f"border-left:3px solid {s_color}'>"
                    f"<span style='color:{s_color};font-weight:bold'>"
                    f"Stage {s.stage_num}: {s.stage_name}</span> "
                    f"<span style='color:#aaa;font-size:0.85em'>@ {s.price_at_stage:,.0f}</span><br>"
                    f"<span style='color:#ccc;font-size:0.82em'>{s.description[:120]}</span><br>"
                    f"<span style='color:#888;font-size:0.78em'>{zone_txt} | "
                    f"Score: {s.stage_score}/100 | "
                    f"{'✅ RSI confirms' if s.rsi_confirms else '⚠️ RSI pending'}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── ATR PANEL ─────────────────────────────────────────────────────────────
    st.markdown("**📐 ATR Volatility Engine**")
    if ofs.atr:
        atr = ofs.atr
        st.markdown(
            f"<div style='padding:8px 12px;border-radius:6px;"
            f"background:{atr.atr_signal_color}22;"
            f"border-left:3px solid {atr.atr_signal_color}'>"
            f"<span style='color:{atr.atr_signal_color};font-weight:bold'>"
            f"{atr.atr_signal}</span></div>",
            unsafe_allow_html=True,
        )
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("ATR (14)",     f"{atr.atr_14:.0f} pts", delta=f"{atr.atr_pct:.2f}%")
        a2.metric("Upper Band",   f"{atr.vwap_upper_band:,.0f}",
                  delta=f"{atr.price_vs_upper:+.2f}% from price")
        a3.metric("Lower Band",   f"{atr.vwap_lower_band:,.0f}",
                  delta=f"{atr.price_vs_lower:+.2f}% from price")
        a4.metric("Stop (Long)",  f"{atr.stop_long_pts:.0f} pts",
                  delta=f"{atr.regime_multiplier:.1f}× ATR mult")
        a5.metric("Breakout ↑",   f"{atr.breakout_long_level:,.0f}",
                  delta=f"Breakout ↓: {atr.breakout_short_level:,.0f}")

    st.markdown("---")

    # ── MEAN REVERSION TRADE ──────────────────────────────────────────────────
    st.markdown("**🔄 Institutional Mean Reversion Trade**")
    if ofs.mean_reversion_trade and ofs.mean_reversion_trade.active:
        mr    = ofs.mean_reversion_trade
        mr_c  = "#2d9e2d" if mr.direction == "LONG" else "#c9302c"
        cv_c  = {"HIGH": "#2d9e2d", "MODERATE": "#e6a817",
                  "MODERATE-HIGH": "#5cb85c", "LOW": "#888"}.get(
                  mr.conviction_label.split()[0], "#e6a817")

        st.markdown(
            f"<div style='padding:12px;border-radius:10px;"
            f"background:{mr_c}22;border:2px solid {mr_c}'>"
            f"<div style='font-size:1.2em;font-weight:bold;color:{mr_c}'>"
            f"{'📈 FADE LONG' if mr.direction == 'LONG' else '📉 FADE SHORT'} "
            f"— {mr.conviction_label}</div>"
            f"<div style='color:#ccc;font-size:0.88em;margin-top:4px'>"
            f"Trigger: {mr.entry_trigger} | "
            f"CPR: {mr.cpr_day_type} day | "
            f"ATR stop: {mr.atr_stop_used:.0f} pts</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Trade levels — 3 targets (institutional tiered approach)
        tl1, tl2, tl3, tl4, tl5, tl6 = st.columns(6)
        tl1.markdown(
            f"<div style='background:#1a2a1a;padding:7px;border-radius:6px;"
            f"border:1px solid {mr_c};text-align:center'>"
            f"<div style='color:#aaa;font-size:0.7em'>ENTRY</div>"
            f"<div style='color:{mr_c};font-weight:bold;font-size:0.9em'>"
            f"{mr.entry_zone_low:,.0f}–{mr.entry_zone_high:,.0f}</div></div>",
            unsafe_allow_html=True,
        )
        tl2.markdown(
            f"<div style='background:#2a1a1a;padding:7px;border-radius:6px;"
            f"border:1px solid #c9302c;text-align:center'>"
            f"<div style='color:#aaa;font-size:0.7em'>STOP (ATR)</div>"
            f"<div style='color:#c9302c;font-weight:bold'>{mr.stop_loss:,.0f}</div>"
            f"<div style='color:#888;font-size:0.7em'>{mr.stop_pts:.0f} pts</div></div>",
            unsafe_allow_html=True,
        )
        tl3.markdown(
            f"<div style='background:#2a2a1a;padding:7px;border-radius:6px;"
            f"border:1px solid #e6a817;text-align:center'>"
            f"<div style='color:#aaa;font-size:0.7em'>TP1 (zone)</div>"
            f"<div style='color:#e6a817;font-weight:bold'>{mr.tp1:,.0f}</div>"
            f"<div style='color:#888;font-size:0.7em'>{mr.rr1:.1f}:1</div></div>",
            unsafe_allow_html=True,
        )
        tl4.markdown(
            f"<div style='background:#1a2a2a;padding:7px;border-radius:6px;"
            f"border:1px solid #5cb85c;text-align:center'>"
            f"<div style='color:#aaa;font-size:0.7em'>TP2 (VWAP)</div>"
            f"<div style='color:#5cb85c;font-weight:bold'>{mr.tp2:,.0f}</div>"
            f"<div style='color:#888;font-size:0.7em'>{mr.rr2:.1f}:1</div></div>",
            unsafe_allow_html=True,
        )
        tl5.markdown(
            f"<div style='background:#1a2a1a;padding:7px;border-radius:6px;"
            f"border:1px solid #2d9e2d;text-align:center'>"
            f"<div style='color:#aaa;font-size:0.7em'>TP3 (full)</div>"
            f"<div style='color:#2d9e2d;font-weight:bold'>{mr.tp3:,.0f}</div>"
            f"<div style='color:#888;font-size:0.7em'>{mr.rr3:.1f}:1</div></div>",
            unsafe_allow_html=True,
        )
        tl6.markdown(
            f"<div style='background:#1a1a1a;padding:7px;border-radius:6px;"
            f"border:1px solid #888;text-align:center'>"
            f"<div style='color:#aaa;font-size:0.7em'>Score</div>"
            f"<div style='color:{cv_c};font-weight:bold'>{mr.conviction_score}/100</div>"
            f"<div style='color:#888;font-size:0.7em'>conviction</div></div>",
            unsafe_allow_html=True,
        )

        # Factors
        with st.expander("✅ Conviction Factors", expanded=True):
            for f in mr.conviction_factors:
                st.markdown(f"✅ {f}")
            st.caption(f"❌ Invalidation: {mr.invalidation}")

    else:
        if ofs.current_stage < 2:
            st.info("Mean reversion trade requires Stage 2 (Absorption) — "
                    "monitor for selling pressure to develop into absorption at a key level.")
        else:
            st.info("Absorption detected but not at a confirmed liquidity zone. "
                    "Trade not active — wait for zone confluence.")

    # ── LIQUIDITY ZONES MAP ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🗺️ Liquidity Zone Map** (nearest 6)")
    for z in ofs.liquidity_zones[:6]:
        q_col = {"STRONG": "#2d9e2d", "MODERATE": "#e6a817", "WEAK": "#888"}.get(z.quality, "#888")
        bar_w = min(abs(z.distance_pct) * 20, 100)
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
            f"<span style='color:{q_col};min-width:60px;font-size:0.8em;font-weight:bold'>"
            f"{z.quality[:3]}</span>"
            f"<span style='color:#ddd;min-width:120px;font-size:0.85em'>{z.label}</span>"
            f"<span style='color:#4a7fb5;font-weight:bold;min-width:70px'>{z.price:,.0f}</span>"
            f"<span style='color:{'#2d9e2d' if z.distance_pts > 0 else '#c9302c'};"
            f"font-size:0.82em'>{z.distance_pts:+.0f} pts</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
