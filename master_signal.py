"""
master_signal.py
----------------
Harmonized Master Signal Aggregator.

Collects evidence from every layer of the dashboard and produces ONE
authoritative signal with conviction score, direction, and precise
entry / SL / TP price ranges.

Signal layers (weighted):
  Layer 1 — Macro Risk          (20 pts max)  yield, oil, breadth ratio, risk score
  Layer 2 — Market Regime       (15 pts max)  VIX regime state, lot multiplier
  Layer 3 — Options Intelligence(25 pts max)  GEX regime, OI walls, expected move
  Layer 4 — Breadth Quality     (15 pts max)  semi leadership, mag7 participation
  Layer 5 — Technical / Price   (25 pts max)  SMA200, MACD, RSI, vol surge, signal
  Layer 6 — QQQ Options+Vol     (15 pts max)  PCR, IV skew, unusual options activity
  Layer 7 — NQ Futures          (15 pts max)  NQ/QQQ leadership, VWAP slope confirmation
  Layer 8 — Order Flow Sequence (12 pts max)  4-stage delta/liquidity-zone footprint sequence
  Layer 9 — Mean Reversion(ATR) (13 pts max)  ATR-extension fade, order-flow + RSI confirmed

Total possible: 155 raw pts, scaled to ±100

Output:
  • MASTER DIRECTION:  LONG | SHORT | HOLD
  • CONVICTION:        STRONG (≥65) | MODERATE (40-64) | WEAK (20-39) | AVOID (<20)
  • ENTRY ZONE:        price range derived from VWAP, OI walls, gamma flip
  • STOP LOSS:         below put wall / above call wall / ATR-based
  • TAKE PROFIT:       scaled to expected move remaining, GEX regime, R:R
  • REASONING:         plain-text summary of each layer's contribution
  • LAYER SCORES:      breakdown for transparency
"""

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class LayerScore:
    name: str
    score: int          # positive = bullish, negative = bearish, 0 = neutral
    max_pts: int        # maximum absolute contribution
    verdict: str        # short human label
    detail: str         # one-line explanation


@dataclass
class EntryZone:
    low: float
    high: float
    description: str    # why this zone


@dataclass
class MasterSignal:
    # Core output
    direction: str              # "LONG" | "SHORT" | "HOLD"
    conviction: str             # "STRONG" | "MODERATE" | "WEAK" | "AVOID"
    conviction_score: int       # -100 to +100 (positive = bullish)
    confidence_pct: int         # 0-100 display value

    # Price levels
    entry_zone: EntryZone
    stop_loss: float
    take_profit_1: float        # conservative TP (1.5:1 R:R)
    take_profit_2: float        # aggressive TP  (2.5:1 R:R)
    risk_pts: float             # pts from entry mid to SL
    reward_1_pts: float
    reward_2_pts: float
    rr_1: float
    rr_2: float

    # Context
    current_price: float
    gamma_flip_zone: Optional[float]
    oi_support: Optional[float]
    oi_resistance: Optional[float]
    expected_move_remaining_pts: float
    vwap: Optional[float]

    # Layer breakdown
    layers: List[LayerScore]

    # Narrative
    summary: str                # 3-4 sentence plain-English rationale
    warnings: List[str]         # specific risk flags
    lot_adjustment: float       # composite lot multiplier from all layers

    # Meta
    timestamp: str
    blockers: List[str]         # reasons that prevented a LONG/SHORT call


# ── CACHE ─────────────────────────────────────────────────────────────────────

CACHE_TTL = 60   # 1 minute — refreshes with each dashboard cycle


def _cache_valid() -> bool:
    return (time.time() - st.session_state.get("master_signal_ts", 0)) < CACHE_TTL


def _store(sig: "MasterSignal"):
    st.session_state["master_signal"]    = sig
    st.session_state["master_signal_ts"] = time.time()


def _load() -> Optional["MasterSignal"]:
    return st.session_state.get("master_signal")


# ── LAYER SCORERS ─────────────────────────────────────────────────────────────

def _score_macro(macro_snap) -> LayerScore:
    """Layer 1: Macro Risk — max ±20 pts."""
    if macro_snap is None:
        return LayerScore("Macro Risk", 0, 20, "Unavailable", "No macro data")

    score = 0
    details = []

    # Risk score: 0=clear(+8), 40=warning(-4), 70=danger(-20)
    rs = macro_snap.risk_score
    if rs < 30:
        score += 8
        details.append(f"Macro clear ({rs}/100)")
    elif rs < 40:
        score += 4
        details.append(f"Macro OK ({rs}/100)")
    elif rs < 70:
        score -= 8
        details.append(f"Macro warning ({rs}/100)")
    else:
        score -= 20
        details.append(f"MACRO DANGER ({rs}/100)")

    # Yield signal
    if "TRAP" in macro_snap.yield_signal:
        score -= 8
        details.append("Yield trap active")
    elif "ELEVATED" in macro_snap.yield_signal:
        score -= 3
        details.append("Elevated yields")
    else:
        score += 4
        details.append("Yields normal")

    # Oil
    if "MARGIN" in macro_snap.oil_signal:
        score -= 5
        details.append("Oil margin pressure")
    elif "WATCH" in macro_snap.oil_signal:
        score -= 2

    # Breadth
    if "EXHAUSTION" in macro_snap.breadth_signal:
        score -= 7
        details.append("Breadth exhaustion")
    elif "DIVERGING" in macro_snap.breadth_signal:
        score -= 3
    else:
        score += 3
        details.append("Broad participation")

    score = max(-20, min(20, score))
    verdict = "Bullish" if score > 5 else ("Bearish" if score < -5 else "Neutral")
    return LayerScore(
        "Macro Risk", score, 20, verdict,
        " | ".join(details[:3])
    )


def _score_regime(regime) -> LayerScore:
    """Layer 2: Market Regime — max ±15 pts."""
    if regime is None:
        return LayerScore("Regime", 0, 15, "Unavailable", "No regime data")

    if regime.state == 2:
        return LayerScore("Regime", -15, 15, "CRISIS — No trade",
                          f"VIX/crisis regime active. {regime.strategy_note[:60]}")
    elif regime.state == 1:
        score = -5   # chop = slight bearish bias (harder to trade)
        verdict = "Choppy"
        detail = f"Sideways regime. Lot ×{regime.lot_multiplier}. Mean-reversion only."
    else:
        score = 12
        verdict = "Trending"
        detail = f"Quiet trend regime. All signals valid. Lot ×{regime.lot_multiplier}."

    # Adjust for risk-off
    if regime.risk_off_active:
        score -= 8
        detail += f" | Risk-off: {regime.risk_off_reason[:40]}"

    score = max(-15, min(15, score))
    return LayerScore("Regime", score, 15, verdict, detail)


def _score_options(gex, heatmap, expected_move, current_price) -> tuple:
    """
    Layer 3: Options Intelligence — max ±25 pts.
    Returns (LayerScore, oi_support, oi_resistance, gamma_flip, em_remaining)
    """
    score = 0
    details = []
    oi_support    = None
    oi_resistance = None
    gamma_flip    = None
    em_remaining  = 9999.0

    if gex is None and heatmap is None and expected_move is None:
        return (LayerScore("Options Intelligence", 0, 25, "Unavailable",
                           "Options data not loaded"), None, None, None, em_remaining)

    # GEX regime
    if gex:
        gamma_flip = gex.gamma_flip_price
        if gex.gamma_regime == "POSITIVE":
            score += 5
            details.append("Positive GEX: mean-revert favored")
        else:
            # Negative GEX amplifies the prevailing direction
            # Determine if price is above or below gamma flip
            if current_price > gex.gamma_flip_price:
                score += 8   # above flip in negative gamma = strong momentum up
                details.append(f"Above gamma flip ({gex.gamma_flip_price:,.0f}) — momentum UP")
            else:
                score -= 8
                details.append(f"Below gamma flip ({gex.gamma_flip_price:,.0f}) — momentum DOWN")

    # OI walls — support/resistance relative to current price
    if heatmap:
        oi_support    = heatmap.max_put_strike
        oi_resistance = heatmap.max_call_strike

        if current_price > heatmap.max_put_strike:
            score += 6
            details.append(f"Above put wall ({heatmap.max_put_strike:,.0f}) = OI support below")
        else:
            score -= 6
            details.append(f"Below put wall ({heatmap.max_put_strike:,.0f}) = bearish OI")

        if current_price < heatmap.max_call_strike:
            score += 3
            details.append(f"Call wall ({heatmap.max_call_strike:,.0f}) overhead — room to run")
        else:
            score -= 2   # above call wall = resistance flipped but next wall unknown

    # Expected move
    if expected_move:
        exh = expected_move.exhaustion_pct
        em_remaining = max(0.0, expected_move.expected_daily_move_pts - expected_move.actual_move_today_pts)

        if exh < 40:
            score += 6
            details.append(f"Only {exh:.0f}% of expected move used — full range ahead")
        elif exh < 70:
            score += 2
            details.append(f"{exh:.0f}% of expected move used")
        elif exh < 100:
            score -= 4
            details.append(f"⚠️ {exh:.0f}% of expected move used — approaching limit")
        else:
            score -= 10
            details.append(f"🔴 Expected move exceeded ({exh:.0f}%) — reversal risk HIGH")

    score = max(-25, min(25, score))
    verdict = "Bullish" if score > 8 else ("Bearish" if score < -8 else "Neutral")
    return (
        LayerScore("Options Intelligence", score, 25, verdict, " | ".join(details[:3])),
        oi_support, oi_resistance, gamma_flip, em_remaining
    )


def _score_breadth(breadth_quality) -> tuple:
    """
    Layer 4: Breadth Quality — max ±15 pts.
    Returns (LayerScore, lot_adjustment)
    """
    if breadth_quality is None:
        return LayerScore("Breadth Quality", 0, 15, "Unavailable", "No breadth data"), 1.0

    bq    = breadth_quality
    score = 0
    details = []

    # Overall quality score → points
    if bq.quality_score >= 70:
        score += 10
        details.append(f"HIGH breadth quality ({bq.quality_score}/100)")
    elif bq.quality_score >= 40:
        score += 4
        details.append(f"MEDIUM breadth quality ({bq.quality_score}/100)")
    else:
        score -= 8
        details.append(f"LOW breadth quality ({bq.quality_score}/100)")

    # Semi leadership
    if bq.semi_leadership:
        sl = bq.semi_leadership
        if sl.signal == "LEADING":
            score += 5
            details.append(f"Semis leading QQQ +{sl.relative_strength:.1f}%")
        elif sl.signal == "LAGGING":
            score -= 5
            details.append(f"Semis lagging QQQ {sl.relative_strength:.1f}%")

    # Momentum decay
    if bq.momentum_decay and bq.momentum_decay.divergence_detected:
        score -= 5
        details.append("RSI divergence detected")

    score = max(-15, min(15, score))
    verdict = "Bullish" if score > 5 else ("Bearish" if score < -5 else "Neutral")
    return (
        LayerScore("Breadth Quality", score, 15, verdict, " | ".join(details[:2])),
        bq.lot_adjustment
    )


def _score_technicals(ind: dict) -> LayerScore:
    """Layer 5: Technical/Price Action — max ±25 pts."""
    if ind is None:
        return LayerScore("Technicals", 0, 25, "Unavailable", "No indicator data")

    score = 0
    details = []

    # Trend vs SMA200
    if ind["trend_status"] == "BULLISH":
        score += 8
        details.append("Above SMA200")
    else:
        score -= 8
        details.append("Below SMA200")

    # MACD
    if ind["macd_bullish"]:
        score += 5
        details.append("MACD bullish")
    else:
        score -= 5
        details.append("MACD bearish")

    # RSI zones
    rsi = ind["rsi"]
    if 45 < rsi < 70:
        score += 5
        details.append(f"RSI healthy ({rsi:.0f})")
    elif rsi <= 35:
        score += 3   # oversold bounce potential
        details.append(f"RSI oversold ({rsi:.0f})")
    elif rsi >= 78:
        score -= 5
        details.append(f"RSI overbought ({rsi:.0f})")
    elif 35 < rsi <= 45:
        score += 1
    else:
        score -= 2   # 70-78 = extended

    # Volume surge
    vr = ind["vol_ratio"]
    if vr >= 1.3:
        score += 4
        details.append(f"Vol surge {vr:.1f}x")
    elif vr >= 1.1:
        score += 2
    elif vr < 0.8:
        score -= 3
        details.append("Low volume")

    # Raw signal type
    raw = ind["signal"]
    if "STRONG BUY" in raw or "MOMENTUM BUY" in raw or "TREND BUY" in raw:
        score += 3
    elif "STRONG SELL" in raw or "MOMENTUM SELL" in raw or "TREND SELL" in raw:
        score -= 3
    elif "Overbought" in raw:
        score -= 2

    score = max(-25, min(25, score))
    verdict = "Bullish" if score > 8 else ("Bearish" if score < -8 else "Neutral")
    return LayerScore("Technicals", score, 25, verdict, " | ".join(details[:3]))


# ── PRICE LEVEL CALCULATOR ────────────────────────────────────────────────────

def _score_qqq_options(qqq_report) -> LayerScore:
    """
    Layer 6: QQQ Options Intelligence — max ±15 pts.
    Reads PCR, IV skew, and unusual volume from qqq_intelligence.py.
    Adds signals that master_signal previously missed entirely.
    """
    if qqq_report is None:
        return LayerScore("QQQ Options", 0, 15, "Unavailable", "No QQQ report")

    score   = 0
    details = []
    opt     = qqq_report.options
    iv      = qqq_report.intraday
    vol     = qqq_report.volume

    # ── PUT-CALL RATIO ────────────────────────────────────────────────────────
    if opt:
        pcr = opt.put_call_ratio_oi
        if pcr > 1.4:
            # Heavy put buying = fear = bearish signal
            score -= 8
            details.append(f"PCR {pcr:.2f} — heavy put buying (fear)")
        elif pcr > 1.2:
            score -= 4
            details.append(f"PCR {pcr:.2f} — elevated puts (caution)")
        elif pcr < 0.6:
            # Extreme call buying = complacency = fade signal
            score -= 5
            details.append(f"PCR {pcr:.2f} — extreme complacency (fade risk)")
        elif pcr < 0.8:
            score += 3
            details.append(f"PCR {pcr:.2f} — mild call lean (moderate bullish)")
        else:
            score += 5
            details.append(f"PCR {pcr:.2f} — balanced (neutral)")

        # ── IV SKEW ───────────────────────────────────────────────────────────
        # Positive skew = puts more expensive than calls = hedging/fear
        # Negative skew = calls more expensive = bullish positioning
        skew = opt.iv_skew
        if skew > 5:
            score -= 5
            details.append(f"IV skew +{skew:.1f}% — put premium high (bearish fear)")
        elif skew > 2:
            score -= 2
            details.append(f"IV skew +{skew:.1f}% — mild put premium")
        elif skew < -3:
            score += 4
            details.append(f"IV skew {skew:.1f}% — call premium (bullish positioning)")
        else:
            score += 2
            details.append(f"IV skew {skew:.1f}% — balanced")

        # ── UNUSUAL OPTIONS ACTIVITY ──────────────────────────────────────────
        if opt.unusual_puts and not opt.unusual_calls:
            score -= 3
            details.append("Unusual PUT activity only — institutional hedging")
        elif opt.unusual_calls and not opt.unusual_puts:
            score += 3
            details.append("Unusual CALL activity only — institutional buying")

    # ── QQQ VOLUME SIGNAL ─────────────────────────────────────────────────────
    if iv:
        vr = iv.vol_pace_ratio
        if vr >= 2.0 and iv.change_pct < -0.5:
            # High volume sell-off = institutional distribution
            score -= 8
            details.append(f"Unusual vol ({vr:.1f}x) on down move — distribution")
        elif vr >= 2.0 and iv.change_pct > 0.5:
            # High volume rally = institutional accumulation
            score += 5
            details.append(f"Unusual vol ({vr:.1f}x) on up move — accumulation")
        elif vr >= 1.5:
            # Elevated volume — amplifies price direction
            if iv.change_pct < 0:
                score -= 3
                details.append(f"Elevated vol ({vr:.1f}x) on decline")
            else:
                score += 3
                details.append(f"Elevated vol ({vr:.1f}x) on advance")

    score = max(-15, min(15, score))
    verdict = "Bullish" if score > 4 else ("Bearish" if score < -4 else "Neutral")
    return LayerScore("QQQ Options+Vol", score, 15, verdict,
                      " | ".join(details[:3]))


def _score_nq_futures(nq_report) -> LayerScore:
    """
    Layer 7: NQ Futures Confirmation — max ±15 pts.
    Uses the NQ Futures Confirmation Score directly.
    Most important signal: is NQ leading or lagging QQQ?
    """
    if nq_report is None or not nq_report.available:
        return LayerScore("NQ Futures", 0, 15, "Unavailable",
                          "NQ=F data not accessible — QQQ used as proxy")

    score   = 0
    details = []
    nqs     = nq_report.score

    if nqs:
        # Use the pre-computed master_signal_pts directly (already ±15 range)
        score = max(-15, min(15, nqs.master_signal_pts))
        details.append(f"NQ score {nqs.score}/100 ({nqs.direction_bias})")

    # Leadership override — most important signal
    if nq_report.leadership:
        lead = nq_report.leadership
        if lead.confirmation == "DIVERGENT":
            score = int(score * 0.3)   # heavily discount conflicted signal
            details.append(f"⚠️ NQ/QQQ DIVERGENT ({lead.spread:+.2f}%)")
        elif lead.confirmation == "CONFIRMED":
            bonus = 3 if lead.spread > 0 else -3
            score = max(-15, min(15, score + bonus))
            details.append(f"NQ leading {'bullish' if lead.spread > 0 else 'bearish'}")

    # VWAP slope confirmation
    if nq_report.price:
        p = nq_report.price
        if p.price_vs_vwap == "ABOVE" and p.vwap_slope == "RISING":
            score = min(15, score + 3)
            details.append("NQ above rising VWAP")
        elif p.price_vs_vwap == "BELOW" and p.vwap_slope == "FALLING":
            score = max(-15, score - 3)
            details.append("NQ below falling VWAP")

    # Overnight range position
    if nq_report.displacement:
        pos = nq_report.displacement.position_in_overnight_range
        if pos > 75:
            score = min(15, score + 2)
        elif pos < 25:
            score = max(-15, score - 2)

    verdict = "Bullish" if score > 4 else ("Bearish" if score < -4 else "Neutral")
    return LayerScore("NQ Futures", score, 15, verdict,
                      " | ".join(details[:2]))


def _score_order_flow_sequence(ofs) -> LayerScore:
    """
    Layer 8: Order Flow Sequence — max ±12 pts.
    Reads the OrderFlowSequence from order_flow_sequence.py (duck-typed, no
    import needed — same pattern as every other layer here). The sequence
    only models the bearish→bullish footprint explicitly (selling pressure →
    absorption → seller exhaustion → reversal confirmed), so this layer is
    asymmetric by design: it can argue for LONG with real conviction, but
    only ever argues mildly against SHORT via unresolved selling pressure.
    """
    if ofs is None or ofs.stage == "NEUTRAL":
        return LayerScore("Order Flow Sequence", 0, 12, "Neutral",
                          "No clear order-flow sequence forming" if ofs else "Unavailable")

    weight = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.25}.get(ofs.confidence, 0.25)
    base = {
        "SELLING_PRESSURE":  -8,
        "ABSORPTION":         3,   # ambiguous — sellers losing effectiveness, not yet confirmed reversal
        "SELLER_EXHAUSTION":  7,
        "REVERSAL_CONFIRMED": 10,
    }.get(ofs.stage, 0)

    score = int(round(base * weight))
    score = max(-12, min(12, score))
    verdict = "Bullish" if score > 3 else ("Bearish" if score < -3 else "Neutral")
    detail = f"{ofs.stage.replace('_',' ').title()} ({ofs.confidence.lower()} confidence)"
    return LayerScore("Order Flow Sequence", score, 12, verdict, detail)


def _score_mean_reversion(mr) -> LayerScore:
    """
    Layer 9: ATR Mean Reversion — max ±13 pts.
    Reads the MeanReversionSetup from mean_reversion_atr.py (duck-typed).
    Only contributes when a setup is `valid` — i.e. extension trigger met,
    confluence-count confirmed, and R:R clears the module's min_rr floor.
    An extended-but-unconfirmed reading (valid=False) scores 0, same as no
    setup at all — extension alone isn't evidence.
    """
    if mr is None or not mr.valid or mr.direction is None:
        detail = mr.description[:110] if mr is not None else "Unavailable"
        return LayerScore("Mean Reversion (ATR)", 0, 13, "No Setup", detail)

    magnitude = 13 if mr.confidence == "HIGH" else 8   # valid setups are MEDIUM or HIGH by construction
    score = magnitude if mr.direction == "LONG" else -magnitude
    verdict = "Bullish" if score > 0 else "Bearish"
    detail = (f"{mr.direction} fade at {mr.extension_atr:+.1f} ATR from {mr.anchor_label} "
              f"(R:R {mr.risk_reward_1:.1f}, {mr.confidence.lower()} confidence)")
    return LayerScore("Mean Reversion (ATR)", score, 13, verdict, detail)


def _resolve_conflicts(
    direction: str,
    conviction: str,
    total_score: int,
    layers: List[LayerScore],
    regime,
    gex,
    expected_move,
    qqq_report,
    breadth_quality,
) -> Tuple[str, str, List[str], List[str]]:
    """
    Explicit conflict resolver — catches situations where multiple layers
    give contradictory signals and resolves them with clear rules.

    Returns (resolved_direction, resolved_conviction, warnings, blockers)

    Conflict rules:
      C1: LONG signal + CHOP regime → downgrade to WEAK or HOLD
      C2: MOMENTUM BUY signal + POSITIVE GEX → downgrade (GEX favors fades)
      C3: Expected move >95% consumed + any entry signal → add warning, reduce conviction
      C4: PCR >1.4 (fear) + LONG signal → warning, don't block but downgrade
      C5: QQQ unusual volume DOWN + LONG signal → blocker (distribution)
      C6: LONG signal + LOW breadth quality + bearish IV skew → downgrade to WEAK
      C7: All layers agree same direction → upgrade conviction
    """
    warnings = []
    blockers = []
    new_dir  = direction
    new_conv = conviction

    tech_layer   = next((l for l in layers if l.name == "Technicals"), None)
    opt_layer    = next((l for l in layers if l.name == "Options Intelligence"), None)
    qqq_layer    = next((l for l in layers if l.name == "QQQ Options+Vol"), None)
    regime_layer = next((l for l in layers if l.name == "Regime"), None)

    raw_signal = tech_layer.detail if tech_layer else ""

    # ── C1: LONG + CHOP regime ────────────────────────────────────────────────
    if direction == "LONG" and regime and regime.state == 1:
        warnings.append(
            "CHOP regime active — momentum/breakout BUY signals have low win rate. "
            "Only mean-reversion dip entries valid."
        )
        if conviction in ("STRONG", "MODERATE"):
            new_conv = "WEAK"

    # ── C2: MOMENTUM BUY + POSITIVE GEX ──────────────────────────────────────
    if (direction == "LONG" and gex and gex.gamma_regime == "POSITIVE"
            and ("MOMENTUM" in raw_signal or "BREAKOUT" in raw_signal)):
        warnings.append(
            f"Positive GEX contradicts momentum signal — "
            f"dealers will SELL into this rally (fade regime). "
            f"Consider mean-reversion SELL above {gex.gamma_flip_price:,.0f} instead."
        )
        if conviction == "STRONG":
            new_conv = "MODERATE"

    # ── C3: Expected move >95% consumed ──────────────────────────────────────
    if expected_move and expected_move.exhaustion_pct >= 95:
        rem = getattr(expected_move, 'expected_move_remaining_pts',
                      max(0.0, expected_move.expected_daily_move_pts
                          - expected_move.actual_move_today_pts))
        warnings.append(
            f"Expected move {expected_move.exhaustion_pct:.0f}% consumed — "
            f"only ±{rem:.0f} pts remain. "
            "High reversal risk. Tight stops or avoid new entries."
        )
        if direction in ("LONG", "SHORT") and conviction in ("STRONG", "MODERATE"):
            new_conv = "WEAK"

    # ── C4: High PCR fear + LONG ──────────────────────────────────────────────
    if qqq_report and qqq_report.options:
        pcr = qqq_report.options.put_call_ratio_oi
        if direction == "LONG" and pcr > 1.4:
            warnings.append(
                f"PCR {pcr:.2f} — institutions are heavily hedged with puts. "
                "This suggests fear or distribution ahead. Reduce long size."
            )
            if conviction == "STRONG":
                new_conv = "MODERATE"

        # ── C5: Vol-Price matrix signals ──────────────────────────────────────
        if qqq_report.intraday:
            iv_c5 = qqq_report.intraday
            pace  = getattr(iv_c5, 'vol_pace_ratio', iv_c5.vol_surge_ratio)
            accel = getattr(iv_c5, 'vol_accel', 1.0)
            chg   = iv_c5.change_pct

            # Distribution: elevated vol + falling price + LONG signal → block
            if direction == "LONG" and pace >= 1.4 and chg < -0.3:
                blockers.append(
                    f"QQQ distribution: volume pace {pace:.1f}× "
                    f"on a {chg:.1f}% decline — institutional selling. "
                    "No long entries."
                )
                new_dir  = "HOLD"
                new_conv = "AVOID"

            # Accumulation: elevated vol + rising price → confirm LONG, warn SHORT
            elif direction == "SHORT" and pace >= 1.4 and chg > 0.3:
                warnings.append(
                    f"QQQ accumulation: volume pace {pace:.1f}× "
                    f"on a +{chg:.1f}% rally — shorting into institutional buying."
                )
                if new_conv == "STRONG":
                    new_conv = "MODERATE"

            # Low conviction rally: low vol + rising price → warn not to chase
            elif direction == "LONG" and pace < 0.75 and chg > 0.2:
                warnings.append(
                    f"Low-conviction rally: vol only {pace:.2f}× session pace "
                    f"on +{chg:.1f}% move. Fade at resistance, don't chase."
                )

            # Accelerating vol into move
            elif accel >= 1.4 and pace >= 0.85:
                if chg > 0.1 and direction == "LONG":
                    pass   # confirmation — no warning needed
                elif chg < -0.1 and direction == "LONG":
                    warnings.append(
                        f"Volume accelerating {accel:.1f}× into decline — "
                        "momentum building to the downside."
                    )

    # ── C6: LONG + LOW breadth + bearish IV skew ──────────────────────────────
    if direction == "LONG" and breadth_quality and breadth_quality.quality_label == "LOW":
        if qqq_report and qqq_report.options and qqq_report.options.iv_skew > 3:
            warnings.append(
                f"LOW breadth quality + bearish IV skew ({qqq_report.options.iv_skew:.1f}%) — "
                "triple confirmation of weakness. Treat as WEAK signal only."
            )
            if new_conv in ("STRONG", "MODERATE"):
                new_conv = "WEAK"

    # ── C7: Layer agreement → upgrade ─────────────────────────────────────────
    bull_layers = sum(1 for l in layers if l.score > l.max_pts * 0.4)
    bear_layers = sum(1 for l in layers if l.score < -l.max_pts * 0.4)

    if direction == "LONG"  and bull_layers >= 5 and new_conv == "MODERATE" and not warnings:
        new_conv = "STRONG"
    if direction == "SHORT" and bear_layers >= 5 and new_conv == "MODERATE" and not warnings:
        new_conv = "STRONG"

    return new_dir, new_conv, warnings, blockers


def _build_price_levels(
    direction: str,
    current_price: float,
    conviction_score: int,
    oi_support: Optional[float],
    oi_resistance: Optional[float],
    gamma_flip: Optional[float],
    vwap: Optional[float],
    em_remaining: float,
    ind: dict,
    scalp_report,
) -> tuple:
    """
    Derive entry zone, SL, TP1, TP2 from all available structural levels.
    Returns (EntryZone, sl, tp1, tp2)
    """
    p = current_price

    # ── ATR proxy from vol surge ratio ───────────────────────────────────────
    # Use 0.3% of price as base ATR unit (~90 pts on NAS100 at 30000)
    # Scale down if low volatility, up if high vol
    vol_ratio  = ind.get("vol_ratio", 1.0) if ind else 1.0
    base_atr   = p * 0.003 * max(vol_ratio, 0.7)

    if direction == "LONG":
        # ── ENTRY ZONE ────────────────────────────────────────────────────────
        # Prefer: at or just above VWAP / put wall / gamma flip (whichever is closest below price)
        anchors_below = [x for x in [vwap, oi_support, gamma_flip] if x and x < p]
        if anchors_below:
            anchor = max(anchors_below)   # closest below = strongest near-term floor
            entry_low  = anchor
            entry_high = anchor + base_atr * 0.5
            zone_desc  = (
                f"Between {anchor:,.0f} (structural floor: "
                f"{'VWAP' if anchor == vwap else ('put wall' if anchor == oi_support else 'gamma flip')})"
                f" and {entry_high:,.0f}"
            )
        else:
            # No anchors below — enter on current price ±0.1%
            entry_low  = p * 0.999
            entry_high = p * 1.001
            zone_desc  = f"At market ({entry_low:,.0f}–{entry_high:,.0f}) — no structural anchor below"

        entry_mid = (entry_low + entry_high) / 2

        # ── STOP LOSS ─────────────────────────────────────────────────────────
        # Below the lowest anchor or 1× ATR below entry
        if oi_support and oi_support < entry_low:
            sl = oi_support - base_atr * 0.3    # below put wall
        elif vwap and vwap < entry_low:
            sl = vwap - base_atr * 0.5           # below VWAP
        else:
            sl = entry_low - base_atr            # pure ATR stop

        # Minimum stop: at least 0.2% below entry
        sl = min(sl, entry_mid * 0.998)

        # ── TAKE PROFITS ──────────────────────────────────────────────────────
        risk = entry_mid - sl

        # TP1: 1.5× R or next resistance (call wall / gamma flip above), whichever is closer
        anchors_above = [x for x in [oi_resistance, gamma_flip] if x and x > p]
        natural_tp1   = entry_mid + risk * 1.5
        if anchors_above:
            nearest_resist = min(anchors_above)
            # Use whichever is more conservative
            tp1 = min(natural_tp1, nearest_resist * 0.999)
        else:
            tp1 = natural_tp1

        # TP2: 2.5× R or expected move upper bound if available
        tp2 = entry_mid + risk * 2.5
        # Cap TP2 to expected move remaining so we don't target beyond IV-implied range
        if em_remaining > 0:
            tp2 = min(tp2, p + em_remaining * 0.85)

        # Sanity: TP2 must be > TP1 > entry
        tp2 = max(tp2, tp1 * 1.001)
        tp1 = max(tp1, entry_high * 1.001)

    else:  # SHORT
        anchors_above = [x for x in [vwap, oi_resistance, gamma_flip] if x and x > p]
        if anchors_above:
            anchor = min(anchors_above)
            entry_high = anchor
            entry_low  = anchor - base_atr * 0.5
            zone_desc  = (
                f"Between {entry_low:,.0f} and {anchor:,.0f} (structural ceiling: "
                f"{'VWAP' if anchor == vwap else ('call wall' if anchor == oi_resistance else 'gamma flip')})"
            )
        else:
            entry_low  = p * 0.999
            entry_high = p * 1.001
            zone_desc  = f"At market ({entry_low:,.0f}–{entry_high:,.0f}) — no structural anchor above"

        entry_mid = (entry_low + entry_high) / 2

        if oi_resistance and oi_resistance > entry_high:
            sl = oi_resistance + base_atr * 0.3
        elif vwap and vwap > entry_high:
            sl = vwap + base_atr * 0.5
        else:
            sl = entry_high + base_atr

        sl = max(sl, entry_mid * 1.002)

        risk = sl - entry_mid

        natural_tp1  = entry_mid - risk * 1.5
        anchors_below = [x for x in [oi_support, gamma_flip] if x and x < p]
        if anchors_below:
            nearest_support = max(anchors_below)
            tp1 = max(natural_tp1, nearest_support * 1.001)
        else:
            tp1 = natural_tp1

        tp2 = entry_mid - risk * 2.5
        if em_remaining > 0:
            tp2 = max(tp2, p - em_remaining * 0.85)

        tp2 = min(tp2, tp1 * 0.999)
        tp1 = min(tp1, entry_low * 0.999)

    entry_zone = EntryZone(low=round(entry_low, 0), high=round(entry_high, 0),
                           description=zone_desc)

    return (
        entry_zone,
        round(sl,  0),
        round(tp1, 0),
        round(tp2, 0),
    )


# ── NARRATIVE BUILDER ─────────────────────────────────────────────────────────

def _build_narrative(direction: str, conviction: str, layers: List[LayerScore],
                     warnings: List[str], blockers: List[str]) -> str:
    if blockers:
        return (
            f"HOLD — {'; '.join(blockers[:2])}. "
            "No directional trade recommended until blockers resolve. "
            "Monitor for regime or macro improvement before re-evaluating."
        )

    bull_layers = [l for l in layers if l.score > 0]
    bear_layers = [l for l in layers if l.score < 0]
    top_bull    = sorted(bull_layers, key=lambda l: l.score, reverse=True)[:2]
    top_bear    = sorted(bear_layers, key=lambda l: l.score)[:2]

    if direction == "LONG":
        leading  = " and ".join([l.name for l in top_bull]) if top_bull else "technical setup"
        friction = " and ".join([l.name for l in top_bear]) if top_bear else "none"
        base = (
            f"{conviction} LONG bias. "
            f"Primary support: {leading}. "
            f"Friction from: {friction}. "
        )
    else:
        leading  = " and ".join([l.name for l in top_bear]) if top_bear else "technical setup"
        friction = " and ".join([l.name for l in top_bull]) if top_bull else "none"
        base = (
            f"{conviction} SHORT bias. "
            f"Pressure from: {leading}. "
            f"Counter-support: {friction}. "
        )

    warn_str = " | ".join(warnings[:2]) if warnings else ""
    if warn_str:
        base += f"Risk flags: {warn_str}."

    return base


# ── MASTER AGGREGATOR ─────────────────────────────────────────────────────────

def compute_master_signal(
    ind: dict,
    macro_snap,
    regime,
    gex,
    heatmap,
    expected_move,
    breadth_quality,
    scalp_report,
    qqq_report=None,        # from get_qqq_report()
    nq_report=None,         # from get_nq_report() — Layer 7
    ofs=None,                # OrderFlowSequence from order_flow_sequence.py — Layer 8
    mr_setup=None,           # MeanReversionSetup from mean_reversion_atr.py — Layer 9
) -> Optional["MasterSignal"]:
    """
    Aggregate all dashboard signals into one Master Signal.
    Safe to call every refresh — internal 60s cache prevents redundant work.
    """
    if _cache_valid():
        cached = _load()
        if cached:
            return cached

    if ind is None:
        return None

    current_price = float(ind["curr_p"])
    timestamp     = pd.Timestamp.now().strftime("%H:%M:%S")

    # ── SCORE EACH LAYER ─────────────────────────────────────────────────────
    l1              = _score_macro(macro_snap)
    l2              = _score_regime(regime)
    l3_tuple        = _score_options(gex, heatmap, expected_move, current_price)
    l3, oi_support, oi_resistance, gamma_flip, em_remaining = l3_tuple
    l4, lot_adj_bq  = _score_breadth(breadth_quality)
    l5              = _score_technicals(ind)
    l6              = _score_qqq_options(qqq_report)   # Layer 6
    l7              = _score_nq_futures(nq_report)      # Layer 7
    l8              = _score_order_flow_sequence(ofs)   # Layer 8
    l9              = _score_mean_reversion(mr_setup)   # Layer 9

    layers      = [l1, l2, l3, l4, l5, l6, l7, l8, l9]
    total_score = sum(l.score for l in layers)   # ±155 max

    # ── INITIAL DIRECTION & CONVICTION ───────────────────────────────────────
    # Scale score to ±100 range (total_score can be ±155 with 9 layers)
    scaled_score = int(total_score * 100 / 155)
    blockers = []
    warnings = []

    if regime and regime.state == 2:
        blockers.append("CRISIS regime active — no new entries")
    if macro_snap and macro_snap.risk_score >= 70:
        blockers.append(f"Macro danger ({macro_snap.risk_score}/100)")

    if blockers:
        direction  = "HOLD"
        conviction = "AVOID"
    elif scaled_score >= 50:
        direction, conviction = "LONG",  "STRONG"
    elif scaled_score >= 25:
        direction, conviction = "LONG",  "MODERATE"
    elif scaled_score >= 10:
        direction, conviction = "LONG",  "WEAK"
    elif scaled_score <= -50:
        direction, conviction = "SHORT", "STRONG"
    elif scaled_score <= -25:
        direction, conviction = "SHORT", "MODERATE"
    elif scaled_score <= -10:
        direction, conviction = "SHORT", "WEAK"
    else:
        direction, conviction = "HOLD",  "WEAK"

    if conviction == "WEAK":
        strong_layers = sum(1 for l in layers if abs(l.score) >= l.max_pts * 0.5)
        if strong_layers < 2:
            direction = "HOLD"

    # ── CONFLICT RESOLUTION (replaces manual warnings) ────────────────────────
    if not blockers:
        direction, conviction, conflict_warnings, conflict_blockers = _resolve_conflicts(
            direction=direction,
            conviction=conviction,
            total_score=scaled_score,
            layers=layers,
            regime=regime,
            gex=gex,
            expected_move=expected_move,
            qqq_report=qqq_report,
            breadth_quality=breadth_quality,
        )
        warnings.extend(conflict_warnings)
        blockers.extend(conflict_blockers)

    # ── REMAINING STANDARD WARNINGS ───────────────────────────────────────────
    if expected_move and expected_move.exhaustion_pct >= 110 and not any(
        "expected move" in w.lower() for w in warnings
    ):
        warnings.append(f"Expected move exceeded ({expected_move.exhaustion_pct:.0f}%)")
    if regime and regime.risk_off_active:
        warnings.append(f"Risk-off: {regime.risk_off_reason[:40]}")
    if breadth_quality and breadth_quality.quality_label == "LOW" and not any(
        "breadth" in w.lower() for w in warnings
    ):
        warnings.append("LOW breadth quality")

    # ── PRICE LEVELS ─────────────────────────────────────────────────────────
    vwap = scalp_report.vwap if scalp_report else None

    if direction in ("LONG", "SHORT"):
        entry_zone, sl, tp1, tp2 = _build_price_levels(
            direction, current_price,
            total_score, oi_support, oi_resistance, gamma_flip,
            vwap, em_remaining, ind, scalp_report,
        )
    else:
        # HOLD — still provide informational levels (no active trade)
        mid = current_price
        entry_zone = EntryZone(
            low=round(mid * 0.998, 0),
            high=round(mid * 1.002, 0),
            description="No entry recommended — monitoring zone only",
        )
        sl  = round(mid * 0.990, 0)
        tp1 = round(mid * 1.010, 0)
        tp2 = round(mid * 1.020, 0)

    entry_mid    = (entry_zone.low + entry_zone.high) / 2
    risk_pts     = abs(entry_mid - sl)
    reward_1_pts = abs(tp1 - entry_mid)
    reward_2_pts = abs(tp2 - entry_mid)
    rr_1         = reward_1_pts / risk_pts if risk_pts > 0 else 0
    rr_2         = reward_2_pts / risk_pts if risk_pts > 0 else 0

    # ── COMPOSITE LOT MULTIPLIER ──────────────────────────────────────────────
    regime_mult = regime.lot_multiplier if regime else 1.0
    lot_adj     = round(regime_mult * lot_adj_bq, 2)
    lot_adj     = max(0.0, min(1.0, lot_adj))
    if direction == "HOLD":
        lot_adj = 0.0

    # ── CONFIDENCE % (display) ───────────────────────────────────────────────
    confidence_pct = min(100, int(abs(scaled_score)))

    # ── NARRATIVE ────────────────────────────────────────────────────────────
    summary = _build_narrative(direction, conviction, layers, warnings, blockers)

    result = MasterSignal(
        direction=direction,
        conviction=conviction,
        conviction_score=scaled_score,
        confidence_pct=confidence_pct,
        entry_zone=entry_zone,
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp2,
        risk_pts=round(risk_pts, 0),
        reward_1_pts=round(reward_1_pts, 0),
        reward_2_pts=round(reward_2_pts, 0),
        rr_1=round(rr_1, 1),
        rr_2=round(rr_2, 1),
        current_price=current_price,
        gamma_flip_zone=gamma_flip,
        oi_support=oi_support,
        oi_resistance=oi_resistance,
        expected_move_remaining_pts=round(em_remaining, 0),
        vwap=vwap,
        layers=layers,
        summary=summary,
        warnings=warnings,
        blockers=blockers,
        lot_adjustment=lot_adj,
        timestamp=timestamp,
    )

    _store(result)
    return result


# ── RENDER ────────────────────────────────────────────────────────────────────

def render_master_signal(ms: Optional["MasterSignal"], risk_config=None):
    """Render the full Master Signal panel."""
    import streamlit as st

    st.subheader("🎯 Master Signal — Harmonized Conviction Engine")

    if ms is None:
        st.warning("Master signal unavailable — waiting for data.")
        return

    # ── DIRECTION BANNER ─────────────────────────────────────────────────────
    dir_colors = {
        "LONG":  ("#1a7a1a", "#2d9e2d", "📈 LONG"),
        "SHORT": ("#5a0000", "#c9302c", "📉 SHORT"),
        "HOLD":  ("#2a2a2a", "#888888", "⏸️ HOLD"),
    }
    bg, border, label = dir_colors.get(ms.direction, dir_colors["HOLD"])
    conv_colors = {
        "STRONG": border, "MODERATE": "#e6a817",
        "WEAK": "#888888", "AVOID": "#c9302c",
    }
    conv_color = conv_colors.get(ms.conviction, "#888888")

    st.markdown(
        f"""
        <div style='padding:16px;border-radius:10px;background:{bg};
                    border:2px solid {border};margin-bottom:12px'>
          <div style='display:flex;align-items:center;gap:16px;flex-wrap:wrap'>
            <span style='font-size:2em;font-weight:900;color:{border}'>{label}</span>
            <span style='font-size:1.3em;font-weight:700;color:{conv_color}'>
              {ms.conviction}</span>
            <span style='color:#aaa;font-size:0.95em'>
              Score: {ms.conviction_score:+d}/100 &nbsp;|&nbsp;
              Confidence: {ms.confidence_pct}% &nbsp;|&nbsp;
              {ms.timestamp}
            </span>
          </div>
          <div style='color:#ddd;margin-top:8px;font-size:0.95em'>{ms.summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Blockers / Warnings
    for b in ms.blockers:
        st.error(f"🚫 BLOCKER: {b}")
    for w in ms.warnings:
        st.warning(f"⚠️ {w}")

    if ms.direction == "HOLD" and not ms.blockers:
        st.info("No high-conviction setup detected. Remain flat until score moves outside ±25.")
        return

    st.markdown("---")

    # ── PRICE LEVELS GRID ────────────────────────────────────────────────────
    st.markdown("**📐 Trade Levels**")
    col_ez, col_sl, col_tp1, col_tp2 = st.columns(4)

    with col_ez:
        st.markdown(
            f"<div style='background:#1a2a1a;padding:10px;border-radius:8px;"
            f"border:1px solid #2d9e2d'>"
            f"<div style='color:#aaa;font-size:0.8em'>ENTRY ZONE</div>"
            f"<div style='color:#2d9e2d;font-size:1.3em;font-weight:bold'>"
            f"{ms.entry_zone.low:,.0f} – {ms.entry_zone.high:,.0f}</div>"
            f"<div style='color:#888;font-size:0.75em;margin-top:4px'>"
            f"{ms.entry_zone.description[:55]}...</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_sl:
        sl_color = "#c9302c"
        st.markdown(
            f"<div style='background:#2a1a1a;padding:10px;border-radius:8px;"
            f"border:1px solid {sl_color}'>"
            f"<div style='color:#aaa;font-size:0.8em'>STOP LOSS</div>"
            f"<div style='color:{sl_color};font-size:1.3em;font-weight:bold'>"
            f"{ms.stop_loss:,.0f}</div>"
            f"<div style='color:#888;font-size:0.75em;margin-top:4px'>"
            f"Risk: {ms.risk_pts:.0f} pts</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_tp1:
        tp1_color = "#e6a817"
        rr1_txt   = f"R:R {ms.rr_1:.1f}:1"
        st.markdown(
            f"<div style='background:#2a2a1a;padding:10px;border-radius:8px;"
            f"border:1px solid {tp1_color}'>"
            f"<div style='color:#aaa;font-size:0.8em'>TP1 (Conservative)</div>"
            f"<div style='color:{tp1_color};font-size:1.3em;font-weight:bold'>"
            f"{ms.take_profit_1:,.0f}</div>"
            f"<div style='color:#888;font-size:0.75em;margin-top:4px'>"
            f"+{ms.reward_1_pts:.0f} pts | {rr1_txt}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col_tp2:
        tp2_color = "#5cb85c"
        rr2_txt   = f"R:R {ms.rr_2:.1f}:1"
        st.markdown(
            f"<div style='background:#1a2a1a;padding:10px;border-radius:8px;"
            f"border:1px solid {tp2_color}'>"
            f"<div style='color:#aaa;font-size:0.8em'>TP2 (Aggressive)</div>"
            f"<div style='color:{tp2_color};font-size:1.3em;font-weight:bold'>"
            f"{ms.take_profit_2:,.0f}</div>"
            f"<div style='color:#888;font-size:0.75em;margin-top:4px'>"
            f"+{ms.reward_2_pts:.0f} pts | {rr2_txt}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── STRUCTURAL CONTEXT ROW ───────────────────────────────────────────────
    st.markdown("**🗺️ Structural Context**")
    c1, c2, c3, c4, c5 = st.columns(5)

    def _level_badge(col, label, value, color="#888"):
        with col:
            if value:
                dist = value - ms.current_price
                col.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='color:#aaa;font-size:0.75em'>{label}</div>"
                    f"<div style='color:{color};font-weight:bold'>{value:,.0f}</div>"
                    f"<div style='color:#666;font-size:0.75em'>{dist:+.0f} pts</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                col.caption(f"{label}: N/A")

    _level_badge(c1, "VWAP",       ms.vwap,          "#e6a817")
    _level_badge(c2, "Put Wall",   ms.oi_support,     "#2d9e2d")
    _level_badge(c3, "Gamma Flip", ms.gamma_flip_zone,"#aa44ff")
    _level_badge(c4, "Call Wall",  ms.oi_resistance,  "#c9302c")
    c5.markdown(
        f"<div style='text-align:center'>"
        f"<div style='color:#aaa;font-size:0.75em'>EM Remaining</div>"
        f"<div style='color:#5cb85c;font-weight:bold'>±{ms.expected_move_remaining_pts:.0f}</div>"
        f"<div style='color:#666;font-size:0.75em'>pts today</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── LOT SIZE RECOMMENDATION ──────────────────────────────────────────────
    st.markdown("---")
    lot_base = risk_config.lot_size if risk_config else 0.02
    lot_rec  = round(lot_base * ms.lot_adjustment, 2)
    lot_rec  = max(lot_rec, 0.01) if ms.direction != "HOLD" else 0.0
    lot_color = "#2d9e2d" if ms.lot_adjustment >= 0.9 else (
                "#e6a817" if ms.lot_adjustment >= 0.6 else "#c9302c")
    st.markdown(
        f"**Recommended Lot:** "
        f"<span style='color:{lot_color};font-size:1.1em;font-weight:bold'>"
        f"{lot_rec} lots</span> "
        f"<span style='color:#888'>(base {lot_base} × {ms.lot_adjustment:.2f} composite multiplier)</span>",
        unsafe_allow_html=True,
    )

    # ── LAYER SCORE BREAKDOWN ────────────────────────────────────────────────
    with st.expander("📊 Layer Score Breakdown", expanded=False):
        total_max = sum(l.max_pts for l in ms.layers)
        for layer in ms.layers:
            bar_pct = abs(layer.score) / layer.max_pts if layer.max_pts > 0 else 0
            bar_col = "#2d9e2d" if layer.score > 0 else ("#c9302c" if layer.score < 0 else "#888")
            sign    = "+" if layer.score >= 0 else ""
            st.markdown(
                f"<div style='margin:4px 0'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='color:#ddd'>{layer.name}</span>"
                f"<span style='color:{bar_col};font-weight:bold'>"
                f"{sign}{layer.score}/{layer.max_pts} — {layer.verdict}</span>"
                f"</div>"
                f"<div style='background:#333;border-radius:4px;height:6px;margin:2px 0'>"
                f"<div style='background:{bar_col};width:{bar_pct*100:.0f}%;height:6px;"
                f"border-radius:4px'></div></div>"
                f"<div style='color:#777;font-size:0.8em'>{layer.detail}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<div style='margin-top:8px;color:#aaa'>"
            f"<b>Total: {ms.conviction_score:+d}</b> / ±{total_max}"
            f"</div>",
            unsafe_allow_html=True,
        )
