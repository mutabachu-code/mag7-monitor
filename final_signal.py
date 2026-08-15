"""
final_signal.py
---------------
The Unified Harmonized Final Signal.

Replaces the current scattered signal outputs with ONE mini-dashboard
that gathers inputs from every layer and produces:

  SCALPING SIGNAL  — for intraday entries (5m / 15m timeframe)
  GENERAL SIGNAL   — for session-level positioning (1H / daily view)

Sources aggregated:
  1. NAS100 Technical Signal     (SMA200, MACD, RSI, vol surge)
  2. Smart Breadth Quality       (semi leadership, Mag7 participation, divergence)
  3. Macro Signals               (yield, oil, risk score)
  4. NAS100 Internal Breadth     (component analysis, A/D, SMA50%)
  5. Options Intelligence        (GEX regime, OI walls, expected move, IV skew)
  6. NAS100 Sniper Scalping      (VWAP setup, CPR signal, liquidity sweeps, gap fill)
  7. CPR Signal                  (narrow/wide day, TC/BC position)
  8. QQQ Intelligence            (PCR, unusual volume, IV skew)
  9. NQ Futures                  (leadership vs QQQ, VWAP, overnight range, basis)
  10. Regime Detector            (trending/chop/crisis)

Output format: a compact mini-dashboard panel with:
  - Overall bias bar (STRONG LONG → STRONG SHORT)
  - Scalping signal: entry zone, SL, TP1, TP2, lot
  - General signal: directional bias + session context
  - Conflict alerts: when layers disagree
  - Per-source signal summary table (all 10 sources in one view)
  - Quick-read conviction gauge
"""

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import time


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class SourceSignal:
    """One row in the signal summary table."""
    source: str          # e.g. "NQ Futures"
    signal: str          # "LONG" | "SHORT" | "NEUTRAL" | "CAUTION" | "AVOID"
    detail: str          # one-line summary
    score: int           # -10 to +10 contribution
    weight: float        # 0.5–2.0 (importance multiplier)
    color: str


@dataclass
class ScalpingSignal:
    """Intraday scalping signal — actionable within current session."""
    direction: str           # "LONG" | "SHORT" | "WAIT"
    conviction: str          # "STRONG" | "MODERATE" | "WEAK"
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    take_profit_1: float     # 1.5:1 R:R
    take_profit_2: float     # 2.5:1 R:R
    risk_pts: float
    rr_1: float
    rr_2: float
    lot_size: float
    primary_setup: str       # e.g. "CPR Narrow BUY at TC" or "VWAP bounce"
    key_level: str           # e.g. "CPR TC: 19,450 | VWAP: 19,420"
    invalidation_note: str   # what kills the trade
    scalp_window: str        # "Next 30-60 min" | "NY Open drive" etc.


@dataclass
class GeneralSignal:
    """Session-level directional bias."""
    direction: str           # "BULLISH" | "BEARISH" | "NEUTRAL"
    strength: str            # "STRONG" | "MODERATE" | "WEAK"
    session_bias: str        # "Favour LONG scalps" | "Favour SHORT" | "Range trade"
    trend_note: str          # 1-line macro/structure context
    key_levels: List[str]    # 3-4 levels to watch this session


@dataclass
class UnifiedSignal:
    """Complete unified final signal — both scalping and general."""
    # Core
    overall_score: int           # -100 to +100 (positive = bullish)
    overall_direction: str       # "STRONG LONG" | "LONG" | "NEUTRAL" | "SHORT" | "STRONG SHORT" | "AVOID"
    confidence_pct: int          # 0-100

    # The two signal outputs
    scalping: Optional[ScalpingSignal]
    general: GeneralSignal

    # Source breakdown
    sources: List[SourceSignal]

    # Conflict detection
    conflicting_sources: List[str]
    conflict_severity: str       # "NONE" | "MINOR" | "MAJOR"
    conflict_note: str

    # Risk context
    regime_override: bool        # True if crisis/chop overrides signals
    regime_note: str
    blockers: List[str]

    # Meta
    timestamp: str


# ── SIGNAL EXTRACTOR ──────────────────────────────────────────────────────────

def _sig_color(s: str) -> str:
    if s in ("LONG", "BULLISH", "STRONG LONG"):   return "#2d9e2d"
    if s in ("SHORT", "BEARISH", "STRONG SHORT"): return "#c9302c"
    if s == "CAUTION":                             return "#e6a817"
    if s == "AVOID":                               return "#8b0000"
    return "#888888"


def _score_to_dir(score: int) -> str:
    if score >= 6:    return "LONG"
    if score <= -6:   return "SHORT"
    if score >= 3:    return "LONG"
    if score <= -3:   return "SHORT"
    return "NEUTRAL"


def _extract_sources(
    nas100_ind:      Optional[dict],
    breadth_quality,
    macro_snap,
    nas100_breadth,
    gex,
    heatmap,
    expected_move,
    scalp_report,
    qqq_report,
    nq_report,
    regime,
    vix_value:       Optional[float],
) -> List[SourceSignal]:
    """Extract a SourceSignal from each of the 10 data layers."""
    sources = []

    # ── 1. NAS100 Technicals ──────────────────────────────────────────────────
    if nas100_ind:
        raw = nas100_ind.get("signal", "⚪ Neutral")
        rsi = nas100_ind.get("rsi", 50)
        if "STRONG BUY" in raw or "MOMENTUM BUY" in raw or "TREND BUY" in raw:
            sig, sc = "LONG",    8
        elif "STRONG SELL" in raw or "MOMENTUM SELL" in raw or "TREND SELL" in raw:
            sig, sc = "SHORT",  -8
        elif "Caution" in raw or "Overbought" in raw:
            sig, sc = "CAUTION", -2
        else:
            sig, sc = "NEUTRAL", 0
        sources.append(SourceSignal(
            source="NAS100 Technicals", signal=sig,
            detail=f"{raw[:50]} | RSI:{rsi:.0f} | "
                   f"{'Above' if nas100_ind.get('trend_status')=='BULLISH' else 'Below'} SMA200",
            score=sc, weight=2.0, color=_sig_color(sig),
        ))

    # ── 2. Breadth Quality ────────────────────────────────────────────────────
    if breadth_quality:
        bq = breadth_quality
        if bq.quality_label == "HIGH":     sig, sc = "LONG",    6
        elif bq.quality_label == "MEDIUM": sig, sc = "NEUTRAL", 0
        else:                               sig, sc = "CAUTION", -5
        div = bq.momentum_decay.divergence_detected if bq.momentum_decay else False
        if div: sc -= 3
        sources.append(SourceSignal(
            source="Breadth Quality", signal=sig,
            detail=f"{bq.quality_label} ({bq.quality_score}/100) | "
                   f"Semi: {bq.semi_leadership.signal if bq.semi_leadership else 'N/A'} | "
                   f"{'⚠️ RSI divergence' if div else 'No divergence'}",
            score=sc, weight=1.5, color=_sig_color(sig),
        ))

    # ── 3. Macro ──────────────────────────────────────────────────────────────
    if macro_snap:
        rs = macro_snap.risk_score
        if rs >= 70:     sig, sc = "AVOID",   -10
        elif rs >= 40:   sig, sc = "CAUTION",  -4
        else:            sig, sc = "LONG",      5
        if "TRAP" in (macro_snap.yield_signal or ""):
            sig, sc = "CAUTION", -6
        sources.append(SourceSignal(
            source="Macro", signal=sig,
            detail=f"Risk {rs}/100 | {macro_snap.yield_signal} | "
                   f"{macro_snap.breadth_signal[:30]}",
            score=sc, weight=1.5, color=_sig_color(sig),
        ))

    # ── 4. NAS100 Internal Breadth ────────────────────────────────────────────
    if nas100_breadth:
        nb = nas100_breadth
        if nb.breadth_trend_bias == "BULLISH":   sig, sc = "LONG",  7
        elif nb.breadth_trend_bias == "BEARISH":  sig, sc = "SHORT",-7
        else:                                      sig, sc = "NEUTRAL", 0
        if nb.sell_off_confirmed: sc -= 4; sig = "SHORT"
        if nb.rally_confirmed:    sc += 4; sig = "LONG"
        if nb.divergence_warning: sc -= 2
        sources.append(SourceSignal(
            source="NAS100 Breadth", signal=sig,
            detail=f"{nb.pct_above_sma50:.0f}% above SMA50 | "
                   f"A/D {nb.advance_decline:.0f}% | "
                   f"{'🔴 Sell-off' if nb.sell_off_confirmed else ('🟢 Rally confirmed' if nb.rally_confirmed else nb.breadth_label)}",
            score=sc, weight=1.5, color=_sig_color(sig),
        ))

    # ── 5. Options Intelligence ───────────────────────────────────────────────
    if gex:
        above_flip = False
        if heatmap and gex.gamma_flip_price:
            above_flip = True  # simplified — compute from price in caller
        if gex.gamma_regime == "NEGATIVE":
            sig, sc = "LONG", 5   # negative GEX amplifies trend
        else:
            sig, sc = "NEUTRAL", 2
        em_warn = expected_move and expected_move.exhaustion_pct >= 90
        if em_warn: sc -= 4; sig = "CAUTION"
        sources.append(SourceSignal(
            source="Options / GEX", signal=sig,
            detail=f"GEX: {gex.gamma_regime} | Flip: {gex.gamma_flip_price:,.0f} | "
                   f"EM: {expected_move.exhaustion_pct:.0f}% used" if expected_move
                   else f"GEX: {gex.gamma_regime}",
            score=sc, weight=1.5, color=_sig_color(sig),
        ))

    # ── 6. Scalping Engine (VWAP + Gap) ──────────────────────────────────────
    if scalp_report:
        sr = scalp_report
        sc = 0; sig = "NEUTRAL"; detail_parts = []
        if sr.vwap_setup:
            vws = sr.vwap_setup
            sc += 5 if vws.direction == "BUY" else -5
            sig = "LONG" if vws.direction == "BUY" else "SHORT"
            detail_parts.append(f"VWAP {vws.direction} ({vws.strength})")
        if sr.gap_fill_setup:
            gfs = sr.gap_fill_setup
            sc += 3 if gfs.direction == "BUY" else -3
            detail_parts.append(f"Gap fill {gfs.direction}")
        if sr.active_fade_setup:
            afs = sr.active_fade_setup
            detail_parts.append(f"Liq sweep {afs.fade_direction}")
        drive_map = {"BULLISH": 3, "BEARISH": -3, "CHOPPY": -1}
        sc += drive_map.get(sr.open_drive or "", 0)
        if sr.open_drive == "CHOPPY": sig = "CAUTION"
        sources.append(SourceSignal(
            source="Scalping Engine", signal=sig,
            detail=" | ".join(detail_parts) or f"Drive: {sr.open_drive}",
            score=min(max(sc, -8), 8), weight=1.5, color=_sig_color(sig),
        ))

    # ── 7. CPR Signal ─────────────────────────────────────────────────────────
    if scalp_report and scalp_report.cpr:
        cpr = scalp_report.cpr
        sc = 0; sig = "NEUTRAL"
        if cpr.setup:
            sig = "LONG" if cpr.setup.direction == "BUY" else "SHORT"
            sc  = 6 if cpr.setup.strength == "STRONG" else 3
            if cpr.setup.direction == "SELL": sc = -sc
        pos_map = {"ABOVE_TC": 3, "INSIDE": 0, "BELOW_BC": -3}
        sc += pos_map.get(cpr.price_vs_cpr, 0)
        sources.append(SourceSignal(
            source="CPR", signal=sig,
            detail=f"{cpr.cpr_type} CPR | {cpr.price_vs_cpr} | "
                   f"TC:{cpr.tc:,.0f} BC:{cpr.bc:,.0f}"
                   + (f" | {cpr.setup.description[:40]}" if cpr.setup else ""),
            score=min(max(sc, -8), 8), weight=1.5, color=_sig_color(sig),
        ))

    # ── 8. QQQ Intelligence ───────────────────────────────────────────────────
    if qqq_report:
        sc = 0; sig = "NEUTRAL"; detail_parts = []
        if qqq_report.options:
            pcr = qqq_report.options.put_call_ratio_oi
            skew = qqq_report.options.iv_skew
            if pcr > 1.4:   sc -= 5; sig = "CAUTION"; detail_parts.append(f"PCR {pcr:.2f} fear")
            elif pcr < 0.7: sc -= 3; detail_parts.append(f"PCR {pcr:.2f} complacency")
            else:           sc += 3; detail_parts.append(f"PCR {pcr:.2f} OK")
            if skew > 4:    sc -= 3; detail_parts.append(f"Skew +{skew:.1f}% bearish")
        if qqq_report.intraday:
            iv2 = qqq_report.intraday
            pace = getattr(iv2, 'vol_pace_ratio', iv2.vol_surge_ratio)
            accel = getattr(iv2, 'vol_accel', 1.0)
            chg   = iv2.change_pct

            if pace >= 1.4 and chg > 0.2:
                sc += 6; sig = "LONG"
                detail_parts.append(f"ACCUMULATION {pace:.1f}× pace + rising")
            elif pace >= 1.4 and chg < -0.2:
                sc -= 6; sig = "SHORT"
                detail_parts.append(f"DISTRIBUTION {pace:.1f}× pace + falling")
            elif accel >= 1.3 and chg > 0.1:
                sc += 4; sig = "LONG"
                detail_parts.append(f"Vol accelerating {accel:.1f}× into rally")
            elif accel >= 1.3 and chg < -0.1:
                sc -= 4; sig = "SHORT"
                detail_parts.append(f"Vol accelerating {accel:.1f}× into decline")
            elif pace < 0.75 and chg > 0.2:
                sc -= 2  # low conviction rally — don't chase
                detail_parts.append(f"Low-vol rally (pace {pace:.2f}×) — low conviction")
            elif pace < 0.75 and chg < -0.2:
                sc += 1  # low vol decline = sellers not committed
                detail_parts.append(f"Low-vol decline — possible exhaustion")
        sources.append(SourceSignal(
            source="QQQ Intelligence", signal=sig,
            detail=" | ".join(detail_parts) or "QQQ normal",
            score=min(max(sc, -8), 8), weight=1.5, color=_sig_color(sig),
        ))

    # ── 9. NQ Futures ─────────────────────────────────────────────────────────
    if nq_report and nq_report.available and nq_report.score:
        nqs = nq_report.score
        if nqs.direction_bias == "BULLISH":   sig, sc = "LONG",    7
        elif nqs.direction_bias == "BEARISH":  sig, sc = "SHORT",  -7
        else:                                   sig, sc = "NEUTRAL", 0
        # Leadership is the most important NQ signal
        if nq_report.leadership:
            if nq_report.leadership.confirmation == "DIVERGENT":
                sig = "CAUTION"; sc = int(sc * 0.4)
        sources.append(SourceSignal(
            source="NQ Futures", signal=sig,
            detail=f"Score {nqs.score}/100 | "
                   f"{nq_report.leadership.leadership_signal[:50] if nq_report.leadership else 'N/A'}",
            score=min(max(nqs.master_signal_pts, -10), 10),
            weight=2.0, color=_sig_color(sig),
        ))
    elif nq_report and not nq_report.available:
        sources.append(SourceSignal(
            source="NQ Futures", signal="NEUTRAL",
            detail="NQ=F data unavailable — QQQ used as proxy",
            score=0, weight=1.0, color="#888",
        ))

    # ── 10. Regime ────────────────────────────────────────────────────────────
    if regime:
        if regime.state == 2:    sig, sc = "AVOID",   -10
        elif regime.state == 1:  sig, sc = "CAUTION",  -3
        else:                     sig, sc = "LONG",      4
        if regime.risk_off_active: sc -= 5; sig = "AVOID"
        sources.append(SourceSignal(
            source="Regime", signal=sig,
            detail=f"{regime.label} | {regime.strategy_note[:50]}",
            score=min(max(sc, -10), 10), weight=2.0, color=_sig_color(sig),
        ))

    return sources


# ── CONFLICT DETECTOR ─────────────────────────────────────────────────────────

def _detect_conflicts(sources: List[SourceSignal]) -> tuple:
    """Returns (conflicting_sources, severity, note)"""
    longs  = [s for s in sources if s.signal == "LONG"  and s.weight >= 1.5]
    shorts = [s for s in sources if s.signal == "SHORT" and s.weight >= 1.5]
    avoids = [s for s in sources if s.signal == "AVOID"]
    caut   = [s for s in sources if s.signal == "CAUTION"]

    conflicts = []
    severity  = "NONE"
    note      = ""

    if avoids:
        severity = "MAJOR"
        note = f"BLOCKER: {', '.join(s.source for s in avoids)} signaling AVOID"
        conflicts = [s.source for s in avoids]
        return conflicts, severity, note

    if longs and shorts:
        severity = "MAJOR"
        l_names  = [s.source for s in longs]
        s_names  = [s.source for s in shorts]
        note     = f"CONFLICT: {', '.join(l_names)} LONG vs {', '.join(s_names)} SHORT"
        conflicts = l_names + s_names
    elif (longs or shorts) and len(caut) >= 2:
        severity = "MINOR"
        note = f"CAUTION: {len(caut)} sources flagging risk"
        conflicts = [s.source for s in caut]

    return conflicts, severity, note


# ── SCALPING SIGNAL BUILDER ───────────────────────────────────────────────────

def _build_scalping_signal(
    direction: str,
    overall_score: int,
    scalp_report,
    nq_report,
    qqq_report,
    current_price: float,
    regime,
    lot_size: float = 0.02,
) -> Optional[ScalpingSignal]:
    """
    Build the intraday scalping signal from the best available setup.
    Priority: CPR Narrow > NQ VWAP > Liquidity Sweep > Gap Fill > Key Level
    """
    if direction == "AVOID" or (regime and regime.state == 2):
        return None

    if current_price <= 0:
        return None

    # Pick best setup
    best_setup   = None
    primary_desc = ""
    key_lvl_str  = ""

    if scalp_report:
        cpr = scalp_report.cpr
        # CPR narrow setup is highest priority
        if cpr and cpr.setup and cpr.cpr_type == "NARROW":
            best_setup   = cpr.setup
            primary_desc = f"CPR NARROW {cpr.setup.direction} at {'TC' if cpr.price_vs_cpr == 'ABOVE_TC' else 'BC'}"
            key_lvl_str  = f"TC: {cpr.tc:,.0f} | BC: {cpr.bc:,.0f} | Pivot: {cpr.pivot:,.0f}"

        # VWAP setup
        elif scalp_report.vwap_setup and not best_setup:
            vs = scalp_report.vwap_setup
            best_setup   = vs
            primary_desc = f"VWAP {vs.direction} ({vs.strength})"
            key_lvl_str  = f"VWAP: {scalp_report.vwap:,.0f}"
            if cpr:
                key_lvl_str += f" | CPR TC: {cpr.tc:,.0f}"

        # Liquidity sweep fade
        elif scalp_report.active_fade_setup and not best_setup:
            afs = scalp_report.active_fade_setup
            primary_desc = f"Liq sweep fade {afs.fade_direction} ({afs.sweep_type})"
            key_lvl_str  = f"Swept: {afs.swept_level:,.0f}"
            # Build a synthetic ScalpSetup-like for price levels
            risk_pts = max(current_price * 0.003, 30)
            if afs.fade_direction == "BUY":
                entry_low  = round(current_price - 10, 0)
                entry_high = round(current_price + 15, 0)
                sl  = round(afs.swept_level - 20, 0)
                tp1 = round(current_price + risk_pts * 1.5, 0)
                tp2 = round(current_price + risk_pts * 2.5, 0)
            else:
                entry_low  = round(current_price - 15, 0)
                entry_high = round(current_price + 10, 0)
                sl  = round(afs.swept_level + 20, 0)
                tp1 = round(current_price - risk_pts * 1.5, 0)
                tp2 = round(current_price - risk_pts * 2.5, 0)
            rk = abs((entry_low + entry_high) / 2 - sl)
            return ScalpingSignal(
                direction=afs.fade_direction, conviction="MODERATE",
                entry_zone_low=entry_low, entry_zone_high=entry_high,
                stop_loss=sl, take_profit_1=tp1, take_profit_2=tp2,
                risk_pts=round(rk, 0),
                rr_1=round(risk_pts * 1.5 / max(rk, 1), 1),
                rr_2=round(risk_pts * 2.5 / max(rk, 1), 1),
                lot_size=lot_size,
                primary_setup=primary_desc,
                key_level=key_lvl_str,
                invalidation_note=f"Close beyond swept level {afs.swept_level:,.0f}",
                scalp_window="Next 30-45 min — sweep reversal",
            )

        # Gap fill
        elif scalp_report.gap_fill_setup and not best_setup:
            best_setup   = scalp_report.gap_fill_setup
            primary_desc = f"Gap fill {best_setup.direction} → {best_setup.target:,.0f}"
            key_lvl_str  = f"Target: {best_setup.target:,.0f}"

    if best_setup:
        mid  = (best_setup.entry_zone_high + best_setup.entry_zone_low) / 2
        risk = abs(mid - best_setup.invalidation)
        rr1  = round(best_setup.pips_to_target / max(best_setup.risk_pips, 1), 1)
        rr2  = round(rr1 * 1.5, 1)
        tp2  = (round(mid + risk * 2.5, 0) if best_setup.direction == "BUY"
                else round(mid - risk * 2.5, 0))

        conv = ("STRONG" if abs(overall_score) >= 50 and "NARROW" in primary_desc
                else "MODERATE" if abs(overall_score) >= 30
                else "WEAK")

        # Lot size adjustment for regime
        lot_adj = lot_size
        if regime and regime.state == 1:
            lot_adj = round(lot_size * 0.5, 2)

        # NQ confirmation adjustment
        if nq_report and nq_report.score:
            if nq_report.score.direction_bias == "NEUTRAL":
                conv = "WEAK" if conv == "MODERATE" else conv

        # Scalp window based on session
        session_map = {
            "US":      "NY session — next 30-60 min",
            "LONDON":  "London session — next 45 min",
            "OVERLAP": "London/NY overlap — high probability window",
            "ASIA":    "Asia session — reduced size, wider stops",
            "AFTER":   "After-hours — avoid unless gap play",
        }
        sess = (nq_report.price.session if nq_report and nq_report.price
                else "US")
        window = session_map.get(sess, "Current session")

        inval_note = (
            f"Close below {best_setup.invalidation:,.0f}"
            if best_setup.direction == "BUY"
            else f"Close above {best_setup.invalidation:,.0f}"
        )

        return ScalpingSignal(
            direction=best_setup.direction,
            conviction=conv,
            entry_zone_low=round(best_setup.entry_zone_low, 0),
            entry_zone_high=round(best_setup.entry_zone_high, 0),
            stop_loss=round(best_setup.invalidation, 0),
            take_profit_1=round(best_setup.target, 0),
            take_profit_2=round(tp2, 0),
            risk_pts=round(risk, 0),
            rr_1=rr1, rr_2=rr2,
            lot_size=lot_adj,
            primary_setup=primary_desc,
            key_level=key_lvl_str,
            invalidation_note=inval_note,
            scalp_window=window,
        )

    return None


# ── GENERAL SIGNAL BUILDER ────────────────────────────────────────────────────

def _build_general_signal(
    direction: str,
    overall_score: int,
    sources: List[SourceSignal],
    scalp_report,
    heatmap,
    nq_report,
    regime,
    macro_snap,
) -> GeneralSignal:
    strength = ("STRONG" if abs(overall_score) >= 50
                else "MODERATE" if abs(overall_score) >= 25
                else "WEAK")

    if direction == "AVOID" or (regime and regime.state == 2):
        return GeneralSignal(
            direction="NEUTRAL", strength="WEAK",
            session_bias="Stay flat — crisis or major blocker active",
            trend_note="All trading suspended until regime clears.",
            key_levels=[],
        )

    # Session bias
    if direction == "LONG":
        sess_bias = ("Favour LONG scalps with VWAP pullbacks"
                     if strength in ("STRONG","MODERATE")
                     else "Cautiously bullish — reduce size, wait for dips")
    elif direction == "SHORT":
        sess_bias = ("Favour SHORT scalps on VWAP bounces"
                     if strength in ("STRONG","MODERATE")
                     else "Cautiously bearish — sell rips, don't chase")
    else:
        sess_bias = "Range trade — buy VAL/S1, sell VAH/R1. Avoid momentum entries."

    # Trend note from strongest sources
    macro_src  = next((s for s in sources if s.source == "Macro"), None)
    nq_src     = next((s for s in sources if s.source == "NQ Futures"), None)
    tech_src   = next((s for s in sources if s.source == "NAS100 Technicals"), None)
    trend_parts = []
    if tech_src:  trend_parts.append(tech_src.detail[:40])
    if nq_src:    trend_parts.append(nq_src.detail[:35])
    if macro_src: trend_parts.append(macro_src.detail[:30])
    trend_note = " | ".join(trend_parts) if trend_parts else "Mixed signals"

    # Key levels to watch
    key_levels = []
    if scalp_report:
        if scalp_report.cpr:
            cpr = scalp_report.cpr
            key_levels.append(f"CPR TC: {cpr.tc:,.0f} | BC: {cpr.bc:,.0f}")
        if scalp_report.vwap:
            key_levels.append(f"VWAP: {scalp_report.vwap:,.0f}")
    if heatmap:
        key_levels.append(f"Call wall: {heatmap.max_call_strike:,.0f}")
        key_levels.append(f"Put wall: {heatmap.max_put_strike:,.0f}")
    if nq_report and nq_report.displacement:
        d = nq_report.displacement
        key_levels.append(f"ON High: {d.overnight_high:,.0f} | Low: {d.overnight_low:,.0f}")

    return GeneralSignal(
        direction=direction,
        strength=strength,
        session_bias=sess_bias,
        trend_note=trend_note,
        key_levels=key_levels[:4],
    )


# ── MASTER COMPUTE ────────────────────────────────────────────────────────────

def compute_unified_signal(
    nas100_ind:      Optional[dict],
    breadth_quality,
    macro_snap,
    nas100_breadth,
    gex,
    heatmap,
    expected_move,
    scalp_report,
    qqq_report,
    nq_report,
    regime,
    vix_value:       Optional[float],
    risk_config=None,
) -> Optional[UnifiedSignal]:
    """
    Aggregate all 10 signal sources into one unified final signal.
    """
    try:
        sources = _extract_sources(
            nas100_ind, breadth_quality, macro_snap, nas100_breadth,
            gex, heatmap, expected_move, scalp_report, qqq_report,
            nq_report, regime, vix_value,
        )

        if not sources:
            return None

        # Weighted score
        total_w = sum(s.weight for s in sources)
        raw_score = sum(s.score * s.weight for s in sources)
        # Scale to -100/+100
        max_possible = sum(10 * s.weight for s in sources)
        overall_score = int(raw_score / max_possible * 100) if max_possible > 0 else 0
        overall_score = max(-100, min(100, overall_score))

        # Direction
        blockers = []
        regime_override = False
        regime_note     = ""

        if regime and regime.state == 2:
            direction = "AVOID"
            regime_override = True
            regime_note = f"CRISIS: {regime.label}. No new entries."
            blockers.append(regime_note)
        elif macro_snap and macro_snap.risk_score >= 70:
            direction = "AVOID"
            blockers.append(f"Macro DANGER ({macro_snap.risk_score}/100)")
        elif overall_score >= 45:   direction = "STRONG LONG"
        elif overall_score >= 20:   direction = "LONG"
        elif overall_score <= -45:  direction = "STRONG SHORT"
        elif overall_score <= -20:  direction = "SHORT"
        else:                        direction = "NEUTRAL"

        # Confidence
        confidence_pct = min(100, int(abs(overall_score)))

        # Conflicts
        conflicts, severity, conflict_note = _detect_conflicts(sources)

        # Current price
        curr_p = 0.0
        if nas100_ind:
            curr_p = float(nas100_ind.get("curr_p", 0))
        if curr_p <= 0 and nq_report and nq_report.price:
            curr_p = nq_report.price.price

        # Base direction for sub-signals
        base_dir = "LONG" if "LONG" in direction else ("SHORT" if "SHORT" in direction else "NEUTRAL")

        lot = risk_config.lot_size if risk_config else 0.02
        scalping = _build_scalping_signal(
            base_dir, overall_score, scalp_report, nq_report,
            qqq_report, curr_p, regime, lot,
        )

        general = _build_general_signal(
            base_dir, overall_score, sources, scalp_report,
            heatmap, nq_report, regime, macro_snap,
        )

        return UnifiedSignal(
            overall_score=overall_score,
            overall_direction=direction,
            confidence_pct=confidence_pct,
            scalping=scalping,
            general=general,
            sources=sources,
            conflicting_sources=conflicts,
            conflict_severity=severity,
            conflict_note=conflict_note,
            regime_override=regime_override,
            regime_note=regime_note,
            blockers=blockers,
            timestamp=pd.Timestamp.now().strftime("%H:%M:%S"),
        )
    except Exception as e:
        print(f"[final_signal] compute error: {e}")
        return None


# ── RENDER ────────────────────────────────────────────────────────────────────

def render_unified_signal(us: Optional[UnifiedSignal], risk_config=None):
    """
    Render the unified harmonized final signal mini-dashboard.
    Replaces ALL individual signal outputs — one compact authoritative view.
    """
    st.subheader("⚡ Unified Harmonized Signal")

    if us is None:
        st.warning("Unified signal unavailable — data loading.")
        return

    # ── OVERALL CONVICTION BAR ─────────────────────────────────────────────────
    dir_cfg = {
        "STRONG LONG":  ("#0d3d0d", "#2d9e2d", "📈📈 STRONG LONG"),
        "LONG":         ("#1a3a1a", "#5cb85c", "📈 LONG"),
        "NEUTRAL":      ("#2a2a2a", "#888888", "⏸  NEUTRAL"),
        "SHORT":        ("#3a1a1a", "#d9534f", "📉 SHORT"),
        "STRONG SHORT": ("#4a0d0d", "#c9302c", "📉📉 STRONG SHORT"),
        "AVOID":        ("#3a0a0a", "#8b0000", "🚫 AVOID"),
    }
    bg, border, label = dir_cfg.get(us.overall_direction, dir_cfg["NEUTRAL"])

    conflict_html = ""
    if us.conflict_severity == "MAJOR":
        conflict_html = (
            f"<div style='background:#5a3a00;padding:6px 10px;border-radius:5px;"
            f"margin-top:8px;font-size:0.88em;color:#e6a817'>"
            f"⚡ CONFLICT: {us.conflict_note}</div>"
        )
    elif us.conflict_severity == "MINOR":
        conflict_html = (
            f"<div style='background:#3a3a00;padding:5px 10px;border-radius:5px;"
            f"margin-top:6px;font-size:0.85em;color:#e6a817'>"
            f"⚠️ {us.conflict_note}</div>"
        )

    st.markdown(
        f"<div style='padding:14px;border-radius:10px;background:{bg};"
        f"border:2px solid {border};margin-bottom:10px'>"
        f"<div style='display:flex;align-items:center;gap:14px;flex-wrap:wrap'>"
        f"<span style='font-size:1.9em;font-weight:900;color:{border}'>{label}</span>"
        f"<span style='color:#ddd;font-size:1em'>Score: {us.overall_score:+d}/100 &nbsp;|&nbsp; "
        f"Confidence: {us.confidence_pct}% &nbsp;|&nbsp; {us.timestamp}</span>"
        f"</div>"
        f"{conflict_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Blockers
    for b in us.blockers:
        st.error(f"🚫 {b}")

    # ── CONVICTION GAUGE ──────────────────────────────────────────────────────
    gauge_pct = (us.overall_score + 100) / 200   # 0-1 range centred at 0.5
    bar_col   = border
    st.markdown(
        f"<div style='background:#333;border-radius:6px;height:14px;margin:6px 0'>"
        f"<div style='background:{bar_col};width:{int(gauge_pct*100)}%;height:14px;"
        f"border-radius:6px'></div></div>"
        f"<div style='display:flex;justify-content:space-between;"
        f"font-size:0.75em;color:#666'>"
        f"<span>STRONG SHORT</span><span>NEUTRAL</span><span>STRONG LONG</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── TWO-COLUMN: SCALPING + GENERAL ────────────────────────────────────────
    scol, gcol = st.columns(2)

    # SCALPING SIGNAL
    with scol:
        st.markdown("### 🎯 Scalping Signal")
        sc = us.scalping
        if sc is None:
            st.info("No scalping setup active. Wait for CPR/VWAP alignment.")
        else:
            sc_dir_col = "#2d9e2d" if sc.direction == "BUY" else (
                         "#c9302c" if sc.direction == "SELL" else "#888")
            conv_cols  = {"STRONG": "#2d9e2d", "MODERATE": "#e6a817", "WEAK": "#888"}
            st.markdown(
                f"<div style='padding:10px;border-radius:8px;"
                f"background:{sc_dir_col}22;border:2px solid {sc_dir_col}'>"
                f"<div style='font-size:1.3em;font-weight:bold;color:{sc_dir_col}'>"
                f"{'📈 BUY' if sc.direction=='BUY' else '📉 SELL'} — {sc.conviction}</div>"
                f"<div style='color:#ccc;font-size:0.88em;margin-top:4px'>"
                f"📋 {sc.primary_setup}</div>"
                f"<div style='color:#aaa;font-size:0.82em'>"
                f"🔑 {sc.key_level}</div>"
                f"<div style='color:#888;font-size:0.8em'>"
                f"⏰ {sc.scalp_window}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            # Price levels
            st.markdown("**Trade Levels**")
            l1, l2, l3, l4 = st.columns(4)
            l1.markdown(
                f"<div style='background:#1a2a1a;padding:6px;border-radius:6px;"
                f"border:1px solid #2d9e2d;text-align:center'>"
                f"<div style='color:#aaa;font-size:0.7em'>ENTRY</div>"
                f"<div style='color:#2d9e2d;font-weight:bold'>"
                f"{sc.entry_zone_low:,.0f}–{sc.entry_zone_high:,.0f}</div></div>",
                unsafe_allow_html=True,
            )
            l2.markdown(
                f"<div style='background:#2a1a1a;padding:6px;border-radius:6px;"
                f"border:1px solid #c9302c;text-align:center'>"
                f"<div style='color:#aaa;font-size:0.7em'>STOP</div>"
                f"<div style='color:#c9302c;font-weight:bold'>{sc.stop_loss:,.0f}</div>"
                f"<div style='color:#888;font-size:0.7em'>{sc.risk_pts:.0f} pts</div></div>",
                unsafe_allow_html=True,
            )
            l3.markdown(
                f"<div style='background:#2a2a1a;padding:6px;border-radius:6px;"
                f"border:1px solid #e6a817;text-align:center'>"
                f"<div style='color:#aaa;font-size:0.7em'>TP1</div>"
                f"<div style='color:#e6a817;font-weight:bold'>{sc.take_profit_1:,.0f}</div>"
                f"<div style='color:#888;font-size:0.7em'>{sc.rr_1:.1f}:1</div></div>",
                unsafe_allow_html=True,
            )
            l4.markdown(
                f"<div style='background:#1a2a1a;padding:6px;border-radius:6px;"
                f"border:1px solid #5cb85c;text-align:center'>"
                f"<div style='color:#aaa;font-size:0.7em'>TP2</div>"
                f"<div style='color:#5cb85c;font-weight:bold'>{sc.take_profit_2:,.0f}</div>"
                f"<div style='color:#888;font-size:0.7em'>{sc.rr_2:.1f}:1</div></div>",
                unsafe_allow_html=True,
            )
            # Lot + invalidation
            lot_base = risk_config.lot_size if risk_config else 0.02
            lot_col  = "#2d9e2d" if sc.lot_size >= lot_base else "#e6a817"
            st.markdown(
                f"**Lot:** <span style='color:{lot_col};font-weight:bold'>"
                f"{sc.lot_size}</span> &nbsp;|&nbsp; "
                f"**❌ Invalidation:** {sc.invalidation_note}",
                unsafe_allow_html=True,
            )

    # GENERAL SIGNAL
    with gcol:
        st.markdown("### 📊 General Signal")
        gs = us.general
        g_col = {"BULLISH": "#2d9e2d", "BEARISH": "#c9302c"}.get(gs.direction, "#888")
        st.markdown(
            f"<div style='padding:10px;border-radius:8px;"
            f"background:{g_col}22;border:2px solid {g_col}'>"
            f"<div style='font-size:1.3em;font-weight:bold;color:{g_col}'>"
            f"{gs.direction} — {gs.strength}</div>"
            f"<div style='color:#ccc;font-size:0.88em;margin-top:4px'>"
            f"🎯 {gs.session_bias}</div>"
            f"<div style='color:#aaa;font-size:0.82em;margin-top:3px'>"
            f"📋 {gs.trend_note[:80]}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if gs.key_levels:
            st.markdown("**Key Levels This Session**")
            for kl in gs.key_levels:
                st.markdown(
                    f"<span style='color:#4a7fb5'>▸</span> {kl}",
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # ── SOURCE SIGNAL TABLE ───────────────────────────────────────────────────
    st.markdown("**📋 All Signal Sources**")
    for src in us.sources:
        score_bar_pct = int((src.score + 10) / 20 * 100)
        is_conflict   = src.source in us.conflicting_sources
        row_bg        = "#2a1a00" if is_conflict else "transparent"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;padding:4px 6px;"
            f"border-radius:5px;background:{row_bg};margin:2px 0'>"
            f"<span style='color:{src.color};font-weight:bold;min-width:140px;"
            f"font-size:0.85em'>{src.source}</span>"
            f"<span style='background:{src.color};color:white;padding:1px 7px;"
            f"border-radius:10px;font-size:0.75em;font-weight:bold;min-width:60px;"
            f"text-align:center'>{src.signal}</span>"
            f"<div style='flex:1;background:#333;border-radius:4px;height:6px'>"
            f"<div style='background:{src.color};width:{score_bar_pct}%;height:6px;"
            f"border-radius:4px'></div></div>"
            f"<span style='color:#aaa;font-size:0.78em;max-width:350px;"
            f"overflow:hidden;white-space:nowrap'>{src.detail[:55]}</span>"
            f"{'<span style=\"color:#e6a817\"> ⚡</span>' if is_conflict else ''}"
            f"</div>",
            unsafe_allow_html=True,
        )
