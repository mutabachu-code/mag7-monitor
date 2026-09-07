"""
options_intelligence.py  — v3
-------------------------------
NAS100 Options Intelligence Engine — fully upgraded per research.

Key upgrades from v2:
  1. Wall Piercing Analysis
  2. Expiration Influence Score per strike (DTE-weighted gamma + vega)
  3. Gamma Wall vs Vega Wall separated
  4. DTE Buckets: 0DTE / 1-3 / 4-7 / 8-30 / 31-90 / 90+
  5. Wall Penetration Probability
  6. Wall Quality Score 0-100 (8 components)
  7. Wall Decay — time-dynamic recalculation
  8. NAS100 Options Reaction Engine panel
  9. CPR + OI confluence
  10. Aggression detection
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone

OI_CACHE_TTL = 300

def _oi_cache_valid(key):
    return (time.time() - st.session_state.get(f"{key}_ts", 0)) < OI_CACHE_TTL

def _store_oi(key, data):
    st.session_state[key] = data
    st.session_state[f"{key}_ts"] = time.time()

def _load_oi(key):
    return st.session_state.get(key)

def _dte_bucket(dte):
    if dte == 0:    return "0DTE",    "Intraday reaction only",        10.0
    elif dte <= 3:  return "1-3DTE",  "Very short-term reaction",      round(1/max(dte**0.5,0.5),2)
    elif dte <= 7:  return "4-7DTE",  "Short-term positioning",        round(1/dte**0.5,2)
    elif dte <= 30: return "8-30DTE", "Swing positioning",             round(1/dte**0.5,2)
    elif dte <= 90: return "31-90DTE","Structural positioning",        round(1/dte**0.5,2)
    else:           return "90+DTE",  "Long-term / vega dominated",    round(1/dte**0.5,2)

@dataclass
class WallLevel:
    strike: float
    wall_type: str
    oi: int
    dte: int
    dte_bucket: str
    dte_interpretation: str
    t_weight: float
    iv: float
    iv_trend: str
    last_price: float
    gamma_est: float
    delta_est: float
    vega_est: float
    is_gamma_wall: bool
    is_vega_wall: bool
    gamma_influence: float
    wall_strength: float
    wall_quality: int
    quality_label: str
    rejection_prob: float
    penetration_prob: float
    penetration_scenario: str
    status: str
    status_color: str
    aggression: str
    aggression_color: str
    distance_pts: float
    distance_pct: float

@dataclass
class OILevel:
    strike: float
    call_oi: int
    put_oi: int
    net_oi: int
    signal: str
    distance_pct: float

@dataclass
class OIHeatmap:
    ticker: str
    current_price: float
    expiry: str
    levels: List[OILevel]
    max_call_strike: float
    max_put_strike: float
    pin_zone: float
    signal_text: str

@dataclass
class GEXData:
    ticker: str
    net_gex: float
    gamma_regime: str
    gamma_flip_price: float
    regime_signal: str
    regime_color: str
    lot_guidance: str

@dataclass
class ExpectedMove:
    ticker: str
    current_price: float
    expected_daily_move_pts: float
    expected_daily_move_pct: float
    upper_bound: float
    lower_bound: float
    actual_move_today_pts: float
    actual_move_today_pct: float
    exhaustion_pct: float
    signal: str
    reversal_warning: bool
    expected_move_remaining_pts: float = 0.0

@dataclass
class OptionsReactionEngine:
    spot: float
    gamma_regime: str
    gamma_flip: float
    vix_trend: str
    yield_10y: float
    put_wall: Optional[WallLevel]
    call_wall: Optional[WallLevel]
    put_gamma_wall: Optional[WallLevel]
    put_vega_wall: Optional[WallLevel]
    call_gamma_wall: Optional[WallLevel]
    call_vega_wall: Optional[WallLevel]
    dte_table: List[Dict]
    market_map: List[Tuple]
    reaction_signal: str
    reaction_color: str
    reaction_confidence: int
    cpr_confluence: Optional[str] = None


def _fetch_all_expiries(ticker_yf="QQQ"):
    cache_key = f"oi_multi_{ticker_yf}"
    if _oi_cache_valid(cache_key):
        return _load_oi(cache_key)
    try:
        ticker = yf.Ticker(ticker_yf)
        exps   = ticker.options
        if not exps:
            return None
        today = pd.Timestamp.now()
        target_dtes = [0, 2, 5, 14, 45, 90]
        selected = []
        for target in target_dtes:
            best, best_diff = None, 9999
            for e in exps:
                diff = abs((pd.Timestamp(e) - today).days - target)
                if diff < best_diff:
                    best_diff, best = diff, e
            if best and best not in selected:
                selected.append(best)
        chains = {}
        for exp in selected[:6]:
            try:
                chain = ticker.option_chain(exp)
                calls = chain.calls[['strike','openInterest','lastPrice',
                                      'impliedVolatility','volume','bid','ask']].copy()
                puts  = chain.puts[['strike','openInterest','lastPrice',
                                     'impliedVolatility','volume','bid','ask']].copy()
                calls.columns = ['strike','call_oi','call_price','call_iv','call_vol','call_bid','call_ask']
                puts.columns  = ['strike','put_oi','put_price','put_iv','put_vol','put_bid','put_ask']
                merged = pd.merge(calls, puts, on='strike', how='outer').fillna(0)
                for col in ['call_oi','put_oi','call_vol','put_vol']:
                    merged[col] = merged[col].astype(int)
                merged['dte']    = int((pd.Timestamp(exp) - today).days)
                merged['expiry'] = exp
                chains[exp] = merged.sort_values('strike').reset_index(drop=True)
            except Exception:
                continue
        if chains:
            _store_oi(cache_key, chains)
        return chains if chains else None
    except Exception as e:
        print(f"[options_intelligence] Multi-expiry fetch error: {e}")
        return None


def _fetch_options_chain(ticker_yf="QQQ"):
    cache_key = f"oi_chain_{ticker_yf}"
    if _oi_cache_valid(cache_key):
        return _load_oi(cache_key)
    try:
        ticker = yf.Ticker(ticker_yf)
        exps   = ticker.options
        if not exps:
            return None
        today = pd.Timestamp.now()
        valid = [e for e in exps if (pd.Timestamp(e) - today).days >= 0]
        exp   = valid[0] if valid else exps[0]
        chain = ticker.option_chain(exp)
        calls = chain.calls[['strike','openInterest','lastPrice',
                              'impliedVolatility','volume','bid','ask']].copy()
        puts  = chain.puts[['strike','openInterest','lastPrice',
                             'impliedVolatility','volume','bid','ask']].copy()
        calls.columns = ['strike','call_oi','call_price','call_iv','call_vol','call_bid','call_ask']
        puts.columns  = ['strike','put_oi','put_price','put_iv','put_vol','put_bid','put_ask']
        merged = pd.merge(calls, puts, on='strike', how='outer').fillna(0)
        for col in ['call_oi','put_oi','call_vol','put_vol']:
            merged[col] = merged[col].astype(int)
        merged['dte']    = int((pd.Timestamp(exp) - today).days)
        merged['expiry'] = exp
        merged = merged.sort_values('strike').reset_index(drop=True)
        _store_oi(cache_key, merged)
        return merged
    except Exception as e:
        print(f"[options_intelligence] Chain fetch error: {e}")
        return None


def _approx_greeks(strike, spot, dte, iv, is_call):
    try:
        from scipy.stats import norm
        T = max(dte, 0.5) / 365.0
        r, sigma = 0.045, max(iv, 0.05)
        S, K = max(spot, 1), max(strike, 1)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*T**0.5)
        gamma = norm.pdf(d1) / (S * sigma * T**0.5)
        vega  = S * norm.pdf(d1) * T**0.5 / 100
        delta = norm.cdf(d1) if is_call else -norm.cdf(-d1)
        return round(gamma, 6), round(abs(delta), 4), round(vega, 4)
    except Exception:
        return 0.001, 0.5, 0.1


def _aggression_signal(vol, oi, bid, ask, last, wall_type):
    if vol == 0 or oi == 0:
        return "LOW", "#888888"
    vol_oi_ratio = vol / max(oi, 1)
    mid  = (bid + ask) / 2 if bid > 0 and ask > 0 else last
    ask_pct = (last - mid) / (ask - mid + 1e-9) if last > 0 and ask > mid else 0.0
    if vol_oi_ratio > 0.5 and ask_pct > 0.5:
        return f"HIGH BUY {'puts' if wall_type=='PUT' else 'calls'}", "#c9302c" if wall_type=="PUT" else "#2d9e2d"
    elif vol_oi_ratio > 0.2 and ask_pct > 0.3:
        return "MODERATE", "#e6a817"
    elif ask_pct < -0.3:
        return "SELLING (closing)", "#5cb85c" if wall_type=="PUT" else "#888"
    return "LOW", "#888888"


def _wall_quality(oi, gamma, dte, iv, vol_ratio, gex_regime,
                   distance_pct, momentum_aligned, spot, max_oi):
    score = 0
    score += int(min(gamma * 10000, 1.0) * 25)
    if max_oi > 0:
        score += int(min(oi / max_oi, 1.0) * 15)
    dte_pts = {0:15, 3:12, 7:9, 30:6, 90:3}
    for d, p in sorted(dte_pts.items()):
        if dte <= d:
            score += p; break
    else:
        score += 1
    score += (10 if iv > 0.35 else 7 if iv > 0.25 else 5 if iv > 0.15 else 2)
    score += (15 if vol_ratio > 2.0 else 10 if vol_ratio > 1.0 else 5 if vol_ratio > 0.5 else 0)
    score += (10 if gex_regime == "POSITIVE" else 3)
    score += (5 if abs(distance_pct) < 0.3 else 3 if abs(distance_pct) < 0.7 else 1 if abs(distance_pct) < 1.5 else 0)
    score += (5 if momentum_aligned else 2)
    return min(max(score, 0), 100)


def _penetration_probability(quality, gex_regime, dte, iv_trend,
                              yield_trend, semi_weak, below_vwap, wall_type):
    base = quality / 100.0
    adj  = 0.0
    adj += (0.10 if dte == 0 else 0.05 if dte <= 3 else -0.05 if dte > 30 else 0)
    adj += (0.08 if gex_regime == "POSITIVE" else -0.08)
    adj += (0.05 if iv_trend == "RISING" else -0.05 if iv_trend == "FALLING" else 0)
    if wall_type == "PUT":
        if yield_trend == "RISING": adj -= 0.05
        if semi_weak:               adj -= 0.05
        if below_vwap:              adj -= 0.08
    else:
        if below_vwap:              adj += 0.05
    rej = min(max(base + adj, 0.10), 0.92)
    pen = 1.0 - rej
    scenario = (
        "Wall fails if: aggressive sell vol + neg delta + QQQ breaks VWAP + 10Y rises → FAILED"
        if wall_type == "PUT" else
        "Wall fails if: heavy call buying + delta surge + breadth improves → BROKEN"
    )
    return round(rej*100, 1), round(pen*100, 1), scenario


def _wall_status(strike, current_price, wall_type, penetration_prob):
    if wall_type == "PUT":
        if current_price > strike * 1.003:   return "INTACT",  "#2d9e2d"
        elif current_price > strike * 0.998: return "TESTING", "#e6a817"
        elif penetration_prob > 60:          return "FAILED",  "#c9302c"
        else:                                 return "BROKEN",  "#8b0000"
    else:
        if current_price < strike * 0.997:   return "INTACT",  "#2d9e2d"
        elif current_price < strike * 1.002: return "TESTING", "#e6a817"
        elif penetration_prob > 60:          return "FAILED",  "#c9302c"
        else:                                 return "BROKEN",  "#8b0000"


def _build_wall(row, wall_type, spot, ratio, gex_regime, max_oi,
                 vix_trend, yield_trend, semi_weak, below_vwap, momentum_toward):
    qqq_s   = float(row['strike'])
    nas_s   = round(qqq_s * ratio, 0)
    dte     = int(row.get('dte', 0))
    bucket, interp, t_wt = _dte_bucket(dte)
    wt      = wall_type.lower()
    oi      = int(row.get(f'{wt}_oi', 0))
    iv      = float(row.get(f'{wt}_iv', 0.20))
    price   = float(row.get(f'{wt}_price', 0))
    vol     = int(row.get(f'{wt}_vol', 0))
    bid     = float(row.get(f'{wt}_bid', 0))
    ask     = float(row.get(f'{wt}_ask', 0))
    is_call = (wall_type == "CALL")
    gamma, delta, vega = _approx_greeks(qqq_s, spot/ratio, dte, iv, is_call)
    gi       = oi * gamma * (spot**2) / 1e8
    strength = oi * gamma * delta * iv * t_wt
    is_gamma = dte <= 7 and gamma > 0.001
    is_vega  = dte >= 30 and vega > 0.1
    vol_ratio= vol / max(oi, 1)
    dist_pct = (nas_s - spot) / spot * 100 if spot > 0 else 0
    quality  = _wall_quality(oi, gamma, dte, iv, vol_ratio, gex_regime,
                              dist_pct, momentum_toward, spot, max_oi)
    q_label  = ("STRONG" if quality >= 80 else "MODERATE" if quality >= 65
                 else "WEAK" if quality >= 50 else "UNRELIABLE")
    rej, pen, pen_s = _penetration_probability(quality, gex_regime, dte, vix_trend,
                                                yield_trend, semi_weak, below_vwap, wall_type)
    status, s_color = _wall_status(nas_s, spot, wall_type, pen)
    aggr, a_color   = _aggression_signal(vol, oi, bid, ask, price, wall_type)
    return WallLevel(
        strike=nas_s, wall_type=wall_type, oi=oi, dte=dte,
        dte_bucket=bucket, dte_interpretation=interp, t_weight=t_wt,
        iv=round(iv*100,1), iv_trend=vix_trend, last_price=price,
        gamma_est=round(gamma,6), delta_est=round(delta,4), vega_est=round(vega,4),
        is_gamma_wall=is_gamma, is_vega_wall=is_vega,
        gamma_influence=round(gi,2), wall_strength=round(strength,4),
        wall_quality=quality, quality_label=q_label,
        rejection_prob=rej, penetration_prob=pen, penetration_scenario=pen_s,
        status=status, status_color=s_color,
        aggression=aggr, aggression_color=a_color,
        distance_pts=round(nas_s-spot,0), distance_pct=round(dist_pct,2),
    )


def get_options_reaction_engine(
    current_price_nas100, qqq_ratio=40.0, gex_regime="NEGATIVE",
    gamma_flip=0.0, vix_value=None, yield_10y=4.5,
    semi_weak=False, below_vwap=False, macro_snap=None,
):
    cache_key = "options_reaction_engine"
    if _oi_cache_valid(cache_key):
        cached = _load_oi(cache_key)
        if cached:
            return cached
    try:
        spot     = current_price_nas100
        qqq_spot = spot / qqq_ratio if qqq_ratio > 0 else spot / 40
        chains   = _fetch_all_expiries("QQQ")
        if not chains:
            return None
        combined = pd.concat(list(chains.values()), ignore_index=True)
        if combined.empty:
            return None
        lo, hi   = qqq_spot * 0.92, qqq_spot * 1.08
        nearby   = combined[(combined['strike'] >= lo) & (combined['strike'] <= hi)].copy()
        if nearby.empty:
            nearby = combined.copy()
        vix_trend   = ("RISING" if vix_value and vix_value > 20 else
                       "FALLING" if vix_value and vix_value < 15 else "FLAT")
        yield_trend = "RISING" if yield_10y > 4.3 else "FLAT"
        max_oi      = max(int(nearby['put_oi'].max()), int(nearby['call_oi'].max()), 1)

        # Put wall candidates (below spot)
        puts_below = nearby[nearby['strike'] * qqq_ratio < spot].copy()
        if puts_below.empty:
            puts_below = nearby.copy()
        put_candidates = []
        for _, row in puts_below.iterrows():
            if row['put_oi'] < 100:
                continue
            toward = abs((row['strike']*qqq_ratio - spot)/spot*100) < 1.0
            put_candidates.append(_build_wall(row, "PUT", spot, qqq_ratio, gex_regime,
                                               max_oi, vix_trend, yield_trend,
                                               semi_weak, below_vwap, toward))
        put_candidates.sort(key=lambda w: w.wall_quality, reverse=True)
        primary_put   = put_candidates[0] if put_candidates else None
        put_gamma_wall= next((w for w in put_candidates if w.is_gamma_wall), None)
        put_vega_wall = next((w for w in put_candidates if w.is_vega_wall), None)

        # Call wall candidates (above spot)
        calls_above = nearby[nearby['strike'] * qqq_ratio > spot].copy()
        if calls_above.empty:
            calls_above = nearby.copy()
        call_candidates = []
        for _, row in calls_above.iterrows():
            if row['call_oi'] < 100:
                continue
            toward = abs((row['strike']*qqq_ratio - spot)/spot*100) < 1.0
            call_candidates.append(_build_wall(row, "CALL", spot, qqq_ratio, gex_regime,
                                                max_oi, vix_trend, yield_trend,
                                                semi_weak, below_vwap, toward))
        call_candidates.sort(key=lambda w: w.wall_quality, reverse=True)
        primary_call   = call_candidates[0] if call_candidates else None
        call_gamma_wall= next((w for w in call_candidates if w.is_gamma_wall), None)
        call_vega_wall = next((w for w in call_candidates if w.is_vega_wall), None)

        # DTE table
        dte_table = []
        key_strikes = set()
        for w in (put_candidates[:3] + call_candidates[:3]):
            key_strikes.add(w.strike)
        for s in sorted(key_strikes):
            rows_at = nearby[abs(nearby['strike']*qqq_ratio - s) < 50]
            if rows_at.empty:
                continue
            row = rows_at.iloc[0]
            dte = int(row.get('dte', 0))
            bucket, interp, _ = _dte_bucket(dte)
            is_put = s < spot
            flames = lambda n: "🔥"*n
            oi_val = int(row.get('put_oi' if is_put else 'call_oi', 0))
            dte_table.append({
                'Strike': f"{s:,.0f} {'PUT' if is_put else 'CALL'}",
                'DTE Bucket': bucket,
                '0DTE Gamma': flames(3) if dte==0 and oi_val>5000 else flames(2) if dte<=3 else flames(1) if dte<=7 else 'Low',
                '7D Gamma':   flames(2) if 4<=dte<=7 else 'Low',
                '30D Vega':   flames(3) if dte>=30 else flames(1) if dte>=14 else 'Low',
                'Interpretation': interp,
            })

        # Market map
        market_map = []
        if primary_put:  market_map.append((primary_put.strike,  "PUT WALL",   "#2d9e2d"))
        market_map.append((spot, "SPOT", "#4a7fb5"))
        if gamma_flip>0: market_map.append((gamma_flip, "GAMMA FLIP", "#aa44ff"))
        if primary_call: market_map.append((primary_call.strike, "CALL WALL",  "#c9302c"))
        market_map.sort(key=lambda x: x[0])

        # Reaction signal
        confidence = 50
        if primary_put and primary_call:
            if primary_put.strike < spot < primary_call.strike:
                if gex_regime == "POSITIVE":
                    sig = (f"RANGE-BOUND — Put ({primary_put.strike:,.0f}) to Call "
                           f"({primary_call.strike:,.0f}). Positive GEX: fade extremes.")
                    col = "#e6a817"; confidence = 65
                else:
                    sig = (f"NEGATIVE GEX momentum. Watch for wall break. "
                           f"Put: {primary_put.strike:,.0f} | Call: {primary_call.strike:,.0f}")
                    col = "#e6a817"; confidence = 55
            elif primary_put.status == "TESTING":
                if primary_put.rejection_prob >= 60:
                    sig = (f"PUT WALL HOLDING ({primary_put.strike:,.0f}) — "
                           f"Rejection {primary_put.rejection_prob:.0f}%. "
                           f"Quality {primary_put.quality_label}. Bounce → long.")
                    col = "#2d9e2d"; confidence = int(primary_put.rejection_prob)
                else:
                    sig = (f"PUT WALL AT RISK ({primary_put.strike:,.0f}) — "
                           f"Penetration {primary_put.penetration_prob:.0f}% likely. "
                           "Wait for retest from below before entry.")
                    col = "#c9302c"; confidence = int(primary_put.penetration_prob)
            elif primary_call.status == "TESTING":
                if primary_call.rejection_prob >= 60:
                    sig = (f"CALL WALL HOLDING ({primary_call.strike:,.0f}) — "
                           f"Rejection {primary_call.rejection_prob:.0f}%. Short scalp.")
                    col = "#c9302c"; confidence = int(primary_call.rejection_prob)
                else:
                    sig = (f"CALL WALL BREAKING ({primary_call.strike:,.0f}) — "
                           f"Momentum long. Pen {primary_call.penetration_prob:.0f}%.")
                    col = "#2d9e2d"; confidence = int(primary_call.penetration_prob)
            else:
                sig = (f"MONITORING — Put: {primary_put.strike:,.0f} ({primary_put.status}) | "
                       f"Call: {primary_call.strike:,.0f} ({primary_call.status})")
                col = "#888888"
        else:
            sig = "Options data loading — walls not yet identified"
            col = "#888888"

        result = OptionsReactionEngine(
            spot=spot, gamma_regime=gex_regime, gamma_flip=gamma_flip,
            vix_trend=vix_trend, yield_10y=yield_10y,
            put_wall=primary_put, call_wall=primary_call,
            put_gamma_wall=put_gamma_wall, put_vega_wall=put_vega_wall,
            call_gamma_wall=call_gamma_wall, call_vega_wall=call_vega_wall,
            dte_table=dte_table, market_map=market_map,
            reaction_signal=sig, reaction_color=col, reaction_confidence=confidence,
        )
        _store_oi(cache_key, result)
        return result
    except Exception as e:
        print(f"[options_intelligence] Reaction engine error: {e}")
        return None


def compute_cpr_oi_confluence(cpr, ore):
    if cpr is None or ore is None:
        return None
    confluences = []
    threshold = ore.spot * 0.005
    if ore.call_wall and abs(cpr.tc - ore.call_wall.strike) < threshold:
        confluences.append(
            f"🔴 CPR TC ({cpr.tc:,.0f}) ≈ Call Wall ({ore.call_wall.strike:,.0f}) "
            f"— dual resistance, {ore.call_wall.rejection_prob:.0f}% rejection"
        )
    if ore.put_wall and abs(cpr.bc - ore.put_wall.strike) < threshold:
        confluences.append(
            f"🟢 CPR BC ({cpr.bc:,.0f}) ≈ Put Wall ({ore.put_wall.strike:,.0f}) "
            f"— dual support, {ore.put_wall.rejection_prob:.0f}% rejection"
        )
    if ore.put_wall and abs(cpr.pivot - ore.put_wall.strike) < threshold:
        confluences.append(
            f"⚡ PIVOT = PUT WALL ({ore.put_wall.strike:,.0f}) — max confluence zone"
        )
    return " | ".join(confluences) if confluences else None


def get_oi_heatmap(current_price_nas100, qqq_ratio=40.0):
    cache_key = "oi_heatmap_nas100"
    if _oi_cache_valid(cache_key):
        return _load_oi(cache_key)
    try:
        qqq_price = current_price_nas100 / qqq_ratio if qqq_ratio > 0 else current_price_nas100 / 40
        chain = _fetch_options_chain("QQQ")
        if chain is None or chain.empty:
            return None
        lo, hi  = qqq_price * 0.95, qqq_price * 1.05
        nearby  = chain[(chain['strike'] >= lo) & (chain['strike'] <= hi)].copy()
        if nearby.empty:
            nearby = chain.copy()
        levels  = []
        for _, row in nearby.iterrows():
            s_nas = round(float(row['strike']) * qqq_ratio, 0)
            c_oi  = int(row.get('call_oi', 0))
            p_oi  = int(row.get('put_oi', 0))
            net   = p_oi - c_oi
            dist  = (s_nas - current_price_nas100) / current_price_nas100 * 100
            if c_oi > p_oi * 2:   sig = "Resistance"
            elif p_oi > c_oi * 2: sig = "Support"
            elif abs(c_oi - p_oi) < max(c_oi, p_oi) * 0.2: sig = "Pin zone"
            else:                  sig = "Neutral"
            levels.append(OILevel(strike=s_nas, call_oi=c_oi, put_oi=p_oi,
                                   net_oi=net, signal=sig, distance_pct=round(dist,2)))
        if not levels:
            return None
        mc_s = round(float(nearby.loc[nearby['call_oi'].idxmax(), 'strike']) * qqq_ratio, 0)
        mp_s = round(float(nearby.loc[nearby['put_oi'].idxmax(),  'strike']) * qqq_ratio, 0)
        nearby['balance'] = abs(nearby['call_oi'] - nearby['put_oi'])
        pin_s = round(float(nearby.loc[nearby['balance'].idxmin(), 'strike']) * qqq_ratio, 0)
        expiry = str(nearby['expiry'].iloc[0]) if 'expiry' in nearby.columns else "nearest"
        if current_price_nas100 > mp_s:
            sig_text = (f"Price above put wall ({mp_s:,.0f}). "
                        f"Dealer positioning favors dip-buying unless {mp_s:,.0f} breaks. "
                        f"Gamma wall resistance at {mc_s:,.0f}.")
        else:
            sig_text = (f"Price below put wall ({mp_s:,.0f}). "
                        f"Dealer hedging may accelerate downside.")
        result = OIHeatmap(ticker="NAS100", current_price=current_price_nas100,
                            expiry=expiry, levels=sorted(levels, key=lambda l: l.strike),
                            max_call_strike=mc_s, max_put_strike=mp_s,
                            pin_zone=pin_s, signal_text=sig_text)
        _store_oi(cache_key, result)
        return result
    except Exception as e:
        print(f"[options_intelligence] OI heatmap error: {e}")
        return None


def get_gex(current_price_nas100, qqq_ratio=40.0):
    cache_key = "gex_nas100"
    if _oi_cache_valid(cache_key):
        return _load_oi(cache_key)
    try:
        qqq_price = current_price_nas100 / qqq_ratio if qqq_ratio > 0 else current_price_nas100 / 40
        chain = _fetch_options_chain("QQQ")
        if chain is None or chain.empty:
            return None
        nearby = chain[(chain['strike'] >= qqq_price*0.97) & (chain['strike'] <= qqq_price*1.03)].copy()
        if nearby.empty:
            nearby = chain[(chain['strike'] >= qqq_price*0.94) & (chain['strike'] <= qqq_price*1.06)].copy()
        if nearby.empty:
            return None
        nearby['dist']   = abs(nearby['strike'] - qqq_price)
        max_dist         = nearby['dist'].max()
        nearby['weight'] = 1 - (nearby['dist'] / (max_dist + 1e-6))
        net_gex = float(((nearby['call_oi'] - nearby['put_oi']) * nearby['weight']).sum())
        chain_s = chain.sort_values('strike').copy()
        chain_s['cum_gex'] = (chain_s['call_oi'] - chain_s['put_oi']).cumsum()
        flip_idx = (chain_s['cum_gex'] * chain_s['cum_gex'].shift(1) < 0)
        flip_qqq = float(chain_s[flip_idx].iloc[0]['strike']) if flip_idx.any() else qqq_price
        flip_nas = round(flip_qqq * qqq_ratio, 0)
        if net_gex >= 0:
            regime, signal, color = "POSITIVE", "Fade extremes — dealers hedge", "#2d9e2d"
            guidance = "Mean-reversion preferred. Tighter TPs."
        else:
            regime, signal, color = "NEGATIVE", "Momentum regime — dealers amplify", "#c9302c"
            guidance = "Momentum/breakout preferred. Wider TPs."
        result = GEXData(ticker="NAS100", net_gex=round(net_gex,0), gamma_regime=regime,
                          gamma_flip_price=flip_nas, regime_signal=signal,
                          regime_color=color, lot_guidance=guidance)
        _store_oi(cache_key, result)
        return result
    except Exception as e:
        print(f"[options_intelligence] GEX error: {e}")
        return None


def _build_em_signal(exhaustion, actual_pts, expected_pts, price):
    if exhaustion >= 100:
        return (f"⚠️ Expected move EXCEEDED ({actual_pts:.0f} vs ±{expected_pts:.0f} pts). High reversal probability.")
    elif exhaustion >= 85:
        return (f"🔴 {exhaustion:.0f}% of expected move used. Reversal risk increasing.")
    elif exhaustion >= 60:
        return (f"🟡 {exhaustion:.0f}% of expected move used. Caution on new entries.")
    elif exhaustion >= 30:
        return (f"🟢 {exhaustion:.0f}% used. Room remains.")
    else:
        return (f"🟢 Only {exhaustion:.0f}% of ±{expected_pts:.0f} pt expected move used.")


def get_expected_move(current_price_nas100, open_price_nas100, qqq_ratio=40.0):
    cache_key = "expected_move_nas100"
    if _oi_cache_valid(cache_key):
        cached = _load_oi(cache_key)
        if cached:
            actual_pts = abs(current_price_nas100 - open_price_nas100)
            actual_pct = actual_pts / open_price_nas100 * 100 if open_price_nas100 > 0 else 0
            exhaustion = actual_pts / cached.expected_daily_move_pts * 100 if cached.expected_daily_move_pts > 0 else 0
            cached.actual_move_today_pts       = round(actual_pts, 0)
            cached.actual_move_today_pct       = round(actual_pct, 2)
            cached.exhaustion_pct              = round(exhaustion, 1)
            cached.reversal_warning            = exhaustion >= 85
            cached.expected_move_remaining_pts = round(max(0.0, cached.expected_daily_move_pts - actual_pts), 0)
            cached.signal = _build_em_signal(exhaustion, actual_pts, cached.expected_daily_move_pts, current_price_nas100)
            return cached
    try:
        qqq_price = current_price_nas100 / qqq_ratio if qqq_ratio > 0 else current_price_nas100 / 40
        chain = _fetch_options_chain("QQQ")
        if chain is None or chain.empty:
            return None
        chain['dist'] = abs(chain['strike'] - qqq_price)
        atm = chain.loc[chain['dist'].idxmin()]
        straddle_nas = (float(atm.get('call_price',0)) + float(atm.get('put_price',0))) * qqq_ratio
        em_pts = round(max(straddle_nas, current_price_nas100 * 0.012), 0)
        em_pct = round(em_pts / current_price_nas100 * 100, 2) if current_price_nas100 > 0 else 0
        upper  = round(current_price_nas100 + em_pts, 0)
        lower  = round(current_price_nas100 - em_pts, 0)
        actual_pts = abs(current_price_nas100 - open_price_nas100)
        actual_pct = actual_pts / open_price_nas100 * 100 if open_price_nas100 > 0 else 0
        exhaustion = actual_pts / em_pts * 100 if em_pts > 0 else 0
        result = ExpectedMove(
            ticker="NAS100", current_price=current_price_nas100,
            expected_daily_move_pts=em_pts, expected_daily_move_pct=em_pct,
            upper_bound=upper, lower_bound=lower,
            actual_move_today_pts=round(actual_pts,0), actual_move_today_pct=round(actual_pct,2),
            exhaustion_pct=round(exhaustion,1),
            signal=_build_em_signal(exhaustion, actual_pts, em_pts, current_price_nas100),
            reversal_warning=(exhaustion >= 85),
            expected_move_remaining_pts=round(max(0.0, em_pts-actual_pts), 0),
        )
        _store_oi(cache_key, result)
        return result
    except Exception as e:
        print(f"[options_intelligence] Expected move error: {e}")
        return None


# ── RENDER ─────────────────────────────────────────────────────────────────────

def render_options_reaction_engine(ore, cpr=None):
    st.subheader("🔥 NAS100 Options Reaction Engine")
    if ore is None:
        st.info("Options Reaction Engine loading — waiting for multi-expiry chain data.")
        return

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Spot", f"{ore.spot:,.0f}")
    gex_col = "#2d9e2d" if ore.gamma_regime == "POSITIVE" else "#c9302c"
    h2.markdown(
        f"<div style='padding:8px;border-radius:6px;background:{gex_col}22'>"
        f"<div style='color:#aaa;font-size:0.75em'>Dealer Gamma</div>"
        f"<div style='color:{gex_col};font-weight:bold'>{ore.gamma_regime}</div></div>",
        unsafe_allow_html=True)
    h3.metric("Gamma Flip", f"{ore.gamma_flip:,.0f}" if ore.gamma_flip else "N/A")
    h4.metric("10Y / VIX",  f"{ore.yield_10y:.2f}% / {ore.vix_trend}")

    st.markdown(
        f"<div style='padding:10px 14px;border-radius:8px;background:{ore.reaction_color}22;"
        f"border:2px solid {ore.reaction_color};margin:8px 0'>"
        f"<span style='color:{ore.reaction_color};font-weight:bold'>{ore.reaction_signal}</span>"
        f"<span style='color:#aaa;margin-left:12px;font-size:0.85em'>"
        f"Confidence: {ore.reaction_confidence}%</span></div>",
        unsafe_allow_html=True)

    if cpr:
        conf = compute_cpr_oi_confluence(cpr, ore)
        if conf:
            st.markdown(
                f"<div style='padding:8px;border-radius:6px;background:#2a1a4a;"
                f"border-left:3px solid #aa44ff;margin-bottom:8px'>"
                f"<span style='color:#aa44ff;font-weight:bold'>⚡ CPR+OI CONFLUENCE: </span>"
                f"<span style='color:#ddd'>{conf}</span></div>",
                unsafe_allow_html=True)

    st.markdown("---")
    wc_put, wc_call = st.columns(2)

    def _wall_card(col, wall, label):
        if wall is None:
            col.caption(f"{label}: No data"); return
        w_col  = "#2d9e2d" if "PUT" in label else "#c9302c"
        flames = "🔥" * (3 if wall.wall_quality >= 80 else 2 if wall.wall_quality >= 65 else 1)
        s_icon = "🟢" if wall.status == "INTACT" else "⚠️" if wall.status == "TESTING" else "🔴"
        col.markdown(
            f"<div style='padding:10px;border-radius:8px;background:{w_col}22;border:2px solid {w_col}'>"
            f"<div style='color:{w_col};font-size:1.1em;font-weight:bold'>{label} — {wall.strike:,.0f}</div>"
            f"<div style='margin:6px 0;color:#aaa;font-size:0.82em'>"
            f"Gamma: {flames} &nbsp; OI: {wall.oi:,} &nbsp; DTE: {wall.dte} ({wall.dte_bucket})<br>"
            f"IV: {wall.iv:.1f}% ({wall.iv_trend}) &nbsp; "
            f"{'⚡ Gamma wall' if wall.is_gamma_wall else '🌊 Vega wall' if wall.is_vega_wall else 'Mixed'}<br>"
            f"Aggression: <span style='color:{wall.aggression_color}'>{wall.aggression}</span></div>"
            f"<div>Quality: <span style='color:{w_col};font-weight:bold'>{wall.wall_quality}/100 — {wall.quality_label}</span></div>"
            f"<div style='margin-top:4px'>"
            f"<span style='color:#2d9e2d'>Rejection: {wall.rejection_prob:.0f}%</span> &nbsp; "
            f"<span style='color:#c9302c'>Penetration: {wall.penetration_prob:.0f}%</span></div>"
            f"<div style='margin-top:4px'>"
            f"<span style='color:{wall.status_color};font-weight:bold'>{s_icon} {wall.status}</span>"
            f"</div></div>",
            unsafe_allow_html=True)

    _wall_card(wc_put,  ore.put_wall,  "PUT WALL")
    _wall_card(wc_call, ore.call_wall, "CALL WALL")

    st.markdown("---")
    st.markdown("**⚡ Gamma Wall vs Vega Wall**")
    gv1, gv2 = st.columns(2)
    with gv1:
        st.caption("Short DTE → Price-reaction (intraday)")
        for w in [ore.put_gamma_wall, ore.call_gamma_wall]:
            if w:
                col = "#2d9e2d" if w.wall_type == "PUT" else "#c9302c"
                st.markdown(
                    f"<span style='color:{col}'>{w.wall_type} {w.strike:,.0f}</span> "
                    f"<span style='color:#888'>{w.dte_bucket} | γ={w.gamma_est:.5f} | T_wt={w.t_weight:.1f}×</span>",
                    unsafe_allow_html=True)
    with gv2:
        st.caption("Long DTE → Structural/volatility")
        for w in [ore.put_vega_wall, ore.call_vega_wall]:
            if w:
                col = "#2d9e2d" if w.wall_type == "PUT" else "#c9302c"
                st.markdown(
                    f"<span style='color:{col}'>{w.wall_type} {w.strike:,.0f}</span> "
                    f"<span style='color:#888'>{w.dte_bucket} | vega={w.vega_est:.3f}</span>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**🗺️ Market Map**")
    for price, label, color in sorted(ore.market_map, reverse=True):
        spot_marker = " ◄" if label == "SPOT" else ""
        bg = "background:#1a2a2a;" if label == "SPOT" else ""
        st.markdown(
            f"<div style='padding:3px 8px;border-left:3px solid {color};{bg}'>"
            f"<span style='color:{color};font-weight:bold'>{price:,.0f}</span>"
            f" — <span style='color:#ccc'>{label}</span>"
            f"<span style='color:#666'>{spot_marker}</span></div>",
            unsafe_allow_html=True)

    if ore.dte_table:
        with st.expander("📊 DTE Breakdown Table", expanded=False):
            st.dataframe(pd.DataFrame(ore.dte_table), use_container_width=True, hide_index=True)

    for wall in [ore.put_wall, ore.call_wall]:
        if wall and wall.penetration_prob >= 50:
            st.warning(
                f"⚠️ {wall.wall_type} WALL {wall.strike:,.0f}: "
                f"{wall.penetration_prob:.0f}% penetration risk. "
                f"{wall.penetration_scenario[:120]}")


def render_oi_heatmap(heatmap):
    st.markdown("**📊 Options OI Heatmap (QQQ → NAS100)**")
    st.caption(f"Expiry: {heatmap.expiry} | Call wall: {heatmap.max_call_strike:,.0f} | "
               f"Put wall: {heatmap.max_put_strike:,.0f} | Pin: {heatmap.pin_zone:,.0f}")
    sig_color = "#2d9e2d" if "above" in heatmap.signal_text else "#c9302c"
    st.markdown(
        f"<div style='padding:8px;border-radius:6px;background:{sig_color}22;"
        f"border-left:3px solid {sig_color};margin:6px 0'>"
        f"<span style='color:{sig_color}'>{heatmap.signal_text}</span></div>",
        unsafe_allow_html=True)
    nearest = sorted(heatmap.levels, key=lambda l: abs(l.distance_pct))[:8]
    nearest = sorted(nearest, key=lambda l: l.strike, reverse=True)
    rows = [{"Strike": f"{lv.strike:,.0f}{'  ◄' if abs(lv.distance_pct)<0.3 else ''}",
             "Call OI": f"{lv.call_oi:,}" if lv.call_oi > 0 else "—",
             "Put OI":  f"{lv.put_oi:,}"  if lv.put_oi  > 0 else "—",
             "Signal":  lv.signal,
             "Dist":    f"{lv.distance_pct:+.1f}%"} for lv in nearest]
    def cs(val):
        if val == "Resistance": return "color:#c9302c;font-weight:bold"
        if val == "Support":    return "color:#2d9e2d;font-weight:bold"
        if val == "Pin zone":   return "color:#e6a817;font-weight:bold"
        return ""
    st.dataframe(pd.DataFrame(rows).style.map(cs, subset=["Signal"]),
                 use_container_width=True, hide_index=True, height=280)


def render_gex_panel(gex):
    st.markdown("**⚡ Gamma Exposure (GEX)**")
    st.markdown(
        f"<div style='padding:8px;border-radius:6px;background:{gex.regime_color}22;"
        f"border-left:3px solid {gex.regime_color}'>"
        f"<span style='color:{gex.regime_color};font-weight:bold'>"
        f"{'📈' if gex.gamma_regime == 'POSITIVE' else '📉'} {gex.gamma_regime} GAMMA</span><br>"
        f"<span style='font-size:0.85em'>{gex.regime_signal}</span></div>",
        unsafe_allow_html=True)
    st.caption(f"Flip zone: {gex.gamma_flip_price:,.0f} | {gex.lot_guidance}")


def render_expected_move_panel(em):
    st.markdown("**🎯 Expected Daily Move**")
    st.markdown(
        f"±<span style='font-size:1.3em;font-weight:bold'>{em.expected_daily_move_pts:.0f} pts</span> "
        f"<span style='color:#aaa'>({em.expected_daily_move_pct:.1f}%)</span>",
        unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Upper bound", f"{em.upper_bound:,.0f}")
    c2.metric("Lower bound", f"{em.lower_bound:,.0f}")
    st.progress(min(em.exhaustion_pct/100, 1.0),
                text=f"Today: {em.actual_move_today_pts:.0f} pts ({em.exhaustion_pct:.0f}% of expected)")
    if em.reversal_warning:       st.error(em.signal)
    elif em.exhaustion_pct >= 60: st.warning(em.signal)
    else:                          st.success(em.signal)
