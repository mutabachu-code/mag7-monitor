"""
options_intelligence.py
-----------------------
Three institutional-grade options engines added per upgrade plan:

  1. Options Open Interest Heatmap
     - QQQ/NAS100 OI by strike (call OI = resistance, put OI = support)
     - Identifies dealer hedging zones and gamma walls
     - Generates signal: "Price above gamma support. Dip-buying favored unless X breaks."

  2. Gamma Exposure Engine (GEX)
     - Net dealer gamma: long gamma (mean-reverting) vs short gamma (trending)
     - Gamma flip zone: strike where dealer positioning flips sign
     - Regime signal: Positive Gamma → fade extremes | Negative Gamma → momentum

  3. Expected Move Engine
     - Calculates implied daily move from nearest ATM options
     - Compares current day's move to expected → reversal probability signal
     - Extremely useful for NAS100 scalping

All data sourced from yfinance options chain (single fetch, cached 10 min).
Zero interference with existing data_fetcher / regime_detector logic.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
from dataclasses import dataclass, field
from typing import Optional, List


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class OILevel:
    strike: float
    call_oi: int
    put_oi: int
    net_oi: int          # put_oi - call_oi  (positive = net support)
    signal: str          # "Resistance" | "Support" | "Pin zone" | "Neutral"
    distance_pct: float  # % distance from current price


@dataclass
class OIHeatmap:
    ticker: str
    current_price: float
    expiry: str
    levels: List[OILevel]
    max_call_strike: float   # largest call OI strike = gamma wall (resistance)
    max_put_strike: float    # largest put OI strike = put wall (support)
    pin_zone: float          # strike where call_oi ≈ put_oi = max pain / pin
    signal_text: str         # institutional-grade narrative signal


@dataclass
class GEXData:
    ticker: str
    net_gex: float           # positive = dealers long gamma, negative = short gamma
    gamma_regime: str        # "POSITIVE" | "NEGATIVE"
    gamma_flip_price: float  # approximate price where GEX flips sign
    regime_signal: str       # "Fade extremes" | "Momentum breakout regime"
    regime_color: str
    lot_guidance: str        # position sizing note


@dataclass
class ExpectedMove:
    ticker: str
    current_price: float
    expected_daily_move_pts: float   # ± points
    expected_daily_move_pct: float   # ± %
    upper_bound: float
    lower_bound: float
    actual_move_today_pts: float     # how far price has moved today
    actual_move_today_pct: float
    exhaustion_pct: float            # actual / expected * 100
    signal: str                      # e.g. "60% of expected move used — room to run"
    reversal_warning: bool
    expected_move_remaining_pts: float = 0.0  # expected_daily_move_pts - actual_move_today_pts


# ── CACHE ─────────────────────────────────────────────────────────────────────

OI_CACHE_TTL = 600   # 10 minutes — options data doesn't change rapidly


def _oi_cache_valid(key: str) -> bool:
    return (time.time() - st.session_state.get(f"{key}_ts", 0)) < OI_CACHE_TTL


def _store_oi(key: str, data):
    st.session_state[key] = data
    st.session_state[f"{key}_ts"] = time.time()


def _load_oi(key: str):
    return st.session_state.get(key)


# ── OPTIONS CHAIN FETCH ───────────────────────────────────────────────────────

def _fetch_options_chain(ticker_yf: str = "QQQ") -> Optional[pd.DataFrame]:
    """
    Fetch the nearest-expiry options chain for QQQ.
    Returns combined calls + puts DataFrame with: strike, callOI, putOI.
    Cached 10 minutes — one fetch serves all three engines.
    """
    cache_key = f"oi_chain_{ticker_yf}"
    if _oi_cache_valid(cache_key):
        return _load_oi(cache_key)

    try:
        ticker = yf.Ticker(ticker_yf)
        exps   = ticker.options
        if not exps:
            return None

        # Pick nearest expiry >= 2 days out (avoid 0DTE noise)
        today = pd.Timestamp.now()
        valid = [e for e in exps if (pd.Timestamp(e) - today).days >= 2]
        exp   = valid[0] if valid else exps[0]

        chain  = ticker.option_chain(exp)
        calls  = chain.calls[['strike', 'openInterest', 'lastPrice']].copy()
        puts   = chain.puts[['strike',  'openInterest', 'lastPrice']].copy()

        calls.columns = ['strike', 'call_oi', 'call_price']
        puts.columns  = ['strike', 'put_oi',  'put_price']

        merged = pd.merge(calls, puts, on='strike', how='outer').fillna(0)
        merged['call_oi'] = merged['call_oi'].astype(int)
        merged['put_oi']  = merged['put_oi'].astype(int)
        merged = merged.sort_values('strike').reset_index(drop=True)
        merged['expiry'] = exp

        _store_oi(cache_key, merged)
        print(f"[options_intelligence] Fetched {len(merged)} strikes for {ticker_yf} exp {exp}")
        return merged

    except Exception as e:
        print(f"[options_intelligence] Chain fetch error: {e}")
        return None


# ── ENGINE 1: OI HEATMAP ──────────────────────────────────────────────────────

def get_oi_heatmap(current_price_nas100: float,
                   qqq_ratio: float = 40.0) -> Optional[OIHeatmap]:
    """
    Build OI heatmap from QQQ options, scaled to NAS100 index prices.
    current_price_nas100: NAS100 index price (~19000).
    qqq_ratio: QQQ-to-NAS100 multiplier from data_fetcher.
    """
    cache_key = "oi_heatmap_nas100"
    if _oi_cache_valid(cache_key):
        return _load_oi(cache_key)

    # QQQ price = NAS100 / ratio
    qqq_price = current_price_nas100 / qqq_ratio if qqq_ratio > 0 else current_price_nas100 / 40

    chain = _fetch_options_chain("QQQ")
    if chain is None or chain.empty:
        return None

    try:
        # Filter to strikes within ±5% of current QQQ price
        lo = qqq_price * 0.95
        hi = qqq_price * 1.05
        nearby = chain[(chain['strike'] >= lo) & (chain['strike'] <= hi)].copy()
        if nearby.empty:
            nearby = chain.copy()

        # Build OI levels scaled to NAS100 index points
        levels = []
        for _, row in nearby.iterrows():
            strike_nas = round(float(row['strike']) * qqq_ratio, 0)
            call_oi    = int(row['call_oi'])
            put_oi     = int(row['put_oi'])
            net_oi     = put_oi - call_oi
            dist_pct   = (strike_nas - current_price_nas100) / current_price_nas100 * 100

            # Classify signal
            if call_oi > put_oi * 2:
                sig = "Resistance"
            elif put_oi > call_oi * 2:
                sig = "Support"
            elif abs(call_oi - put_oi) < max(call_oi, put_oi) * 0.2:
                sig = "Pin zone"
            else:
                sig = "Neutral"

            levels.append(OILevel(
                strike=strike_nas,
                call_oi=call_oi,
                put_oi=put_oi,
                net_oi=net_oi,
                signal=sig,
                distance_pct=round(dist_pct, 2),
            ))

        if not levels:
            return None

        # Max pain / gamma walls
        max_call_row = nearby.loc[nearby['call_oi'].idxmax()]
        max_put_row  = nearby.loc[nearby['put_oi'].idxmax()]
        max_call_strike = round(float(max_call_row['strike']) * qqq_ratio, 0)
        max_put_strike  = round(float(max_put_row['strike'])  * qqq_ratio, 0)

        # Pin zone = strike where |call_oi - put_oi| is minimised
        nearby['balance'] = abs(nearby['call_oi'] - nearby['put_oi'])
        pin_row    = nearby.loc[nearby['balance'].idxmin()]
        pin_strike = round(float(pin_row['strike']) * qqq_ratio, 0)
        expiry     = str(nearby['expiry'].iloc[0]) if 'expiry' in nearby.columns else "nearest"

        # Narrative signal
        if current_price_nas100 > max_put_strike:
            support_str = f"{max_put_strike:,.0f}"
            resist_str  = f"{max_call_strike:,.0f}"
            signal_text = (
                f"Price trading above major gamma support ({support_str}). "
                f"Dealer positioning favors dip-buying unless {support_str} breaks. "
                f"Gamma wall resistance at {resist_str}."
            )
        else:
            signal_text = (
                f"Price below gamma support ({max_put_strike:,.0f}). "
                f"Dealer put hedging may accelerate downside. "
                f"Watch for recovery above {max_put_strike:,.0f}."
            )

        result = OIHeatmap(
            ticker="NAS100",
            current_price=current_price_nas100,
            expiry=expiry,
            levels=sorted(levels, key=lambda l: l.strike),
            max_call_strike=max_call_strike,
            max_put_strike=max_put_strike,
            pin_zone=pin_strike,
            signal_text=signal_text,
        )

        _store_oi(cache_key, result)
        return result

    except Exception as e:
        print(f"[options_intelligence] OI heatmap error: {e}")
        return None


# ── ENGINE 2: GAMMA EXPOSURE (GEX) ────────────────────────────────────────────

def get_gex(current_price_nas100: float,
            qqq_ratio: float = 40.0) -> Optional[GEXData]:
    """
    Approximate dealer GEX from QQQ options open interest.
    GEX formula: (call_OI - put_OI) × delta × gamma × spot² × contract_size
    Simplified: use call_OI - put_OI weighted by proximity to spot as proxy.
    Positive GEX → dealers long gamma → they sell rallies, buy dips → mean reversion.
    Negative GEX → dealers short gamma → they buy rallies, sell dips → trends.
    """
    cache_key = "gex_nas100"
    if _oi_cache_valid(cache_key):
        return _load_oi(cache_key)

    qqq_price = current_price_nas100 / qqq_ratio if qqq_ratio > 0 else current_price_nas100 / 40
    chain = _fetch_options_chain("QQQ")
    if chain is None or chain.empty:
        return None

    try:
        # Weight OI by moneyness proximity (closer strikes matter more)
        nearby = chain[
            (chain['strike'] >= qqq_price * 0.97) &
            (chain['strike'] <= qqq_price * 1.03)
        ].copy()

        if nearby.empty:
            nearby = chain[
                (chain['strike'] >= qqq_price * 0.94) &
                (chain['strike'] <= qqq_price * 1.06)
            ].copy()

        if nearby.empty:
            return None

        # Proximity weight: inverse distance from spot
        nearby['dist'] = abs(nearby['strike'] - qqq_price)
        max_dist = nearby['dist'].max()
        nearby['weight'] = 1 - (nearby['dist'] / (max_dist + 1e-6))

        # Net GEX: positive = net call OI (dealers long gamma), negative = net put OI
        net_gex = float(
            ((nearby['call_oi'] - nearby['put_oi']) * nearby['weight']).sum()
        )

        # Gamma flip zone: find strike where cumulative GEX crosses zero
        chain_sorted = chain.sort_values('strike').copy()
        chain_sorted['cum_gex'] = (chain_sorted['call_oi'] - chain_sorted['put_oi']).cumsum()

        flip_idx = (chain_sorted['cum_gex'] * chain_sorted['cum_gex'].shift(1) < 0)
        if flip_idx.any():
            flip_row  = chain_sorted[flip_idx].iloc[0]
            flip_qqq  = float(flip_row['strike'])
        else:
            flip_qqq = qqq_price  # default to spot if no flip found
        flip_nas = round(flip_qqq * qqq_ratio, 0)

        if net_gex >= 0:
            regime   = "POSITIVE"
            signal   = "Fade extremes — Dealers hedge by selling rallies & buying dips"
            color    = "#2d9e2d"
            guidance = "Mean-reversion signals preferred. Tighter TP targets."
        else:
            regime   = "NEGATIVE"
            signal   = "Momentum breakout regime — Dealers amplify moves"
            color    = "#c9302c"
            guidance = "Momentum/breakout signals preferred. Wider TP targets."

        result = GEXData(
            ticker="NAS100",
            net_gex=round(net_gex, 0),
            gamma_regime=regime,
            gamma_flip_price=flip_nas,
            regime_signal=signal,
            regime_color=color,
            lot_guidance=guidance,
        )

        _store_oi(cache_key, result)
        return result

    except Exception as e:
        print(f"[options_intelligence] GEX error: {e}")
        return None


# ── ENGINE 3: EXPECTED MOVE ───────────────────────────────────────────────────

def get_expected_move(current_price_nas100: float,
                      open_price_nas100: float,
                      qqq_ratio: float = 40.0) -> Optional[ExpectedMove]:
    """
    Calculate today's expected daily move from ATM options IV.
    Formula: Expected Move = Price × IV × √(1/252)
    Then compare actual intraday move to expected → reversal probability.
    """
    cache_key = "expected_move_nas100"
    if _oi_cache_valid(cache_key):
        cached = _load_oi(cache_key)
        if cached:
            # Update actual move even on cache hit (price changes)
            actual_pts = abs(current_price_nas100 - open_price_nas100)
            actual_pct = actual_pts / open_price_nas100 * 100 if open_price_nas100 > 0 else 0
            exhaustion = (actual_pts / cached.expected_daily_move_pts * 100
                          if cached.expected_daily_move_pts > 0 else 0)
            cached.actual_move_today_pts = round(actual_pts, 0)
            cached.actual_move_today_pct = round(actual_pct, 2)
            cached.exhaustion_pct        = round(exhaustion, 1)
            cached.reversal_warning      = exhaustion >= 85
            cached.expected_move_remaining_pts = round(
                max(0.0, cached.expected_daily_move_pts - actual_pts), 0
            )
            cached.signal = _build_em_signal(
                exhaustion, actual_pts, cached.expected_daily_move_pts, current_price_nas100
            )
            return cached

    qqq_price = current_price_nas100 / qqq_ratio if qqq_ratio > 0 else current_price_nas100 / 40
    chain = _fetch_options_chain("QQQ")
    if chain is None or chain.empty:
        return None

    try:
        # Find ATM strike
        chain['dist'] = abs(chain['strike'] - qqq_price)
        atm = chain.loc[chain['dist'].idxmin()]

        # Use average of ATM call + put price as straddle price (≈ implied move)
        straddle_price_qqq = float(atm.get('call_price', 0)) + float(atm.get('put_price', 0))

        # Convert straddle price to NAS100 points
        straddle_nas = straddle_price_qqq * qqq_ratio
        # Straddle price ≈ 68% confidence interval (1σ expected move)
        expected_1sd = straddle_nas if straddle_nas > 10 else current_price_nas100 * 0.012

        # For display, use ±1SD as expected daily range
        expected_move_pts = round(expected_1sd, 0)
        expected_move_pct = round(expected_move_pts / current_price_nas100 * 100, 2)

        upper = round(current_price_nas100 + expected_move_pts, 0)
        lower = round(current_price_nas100 - expected_move_pts, 0)

        actual_pts = abs(current_price_nas100 - open_price_nas100)
        actual_pct = actual_pts / open_price_nas100 * 100 if open_price_nas100 > 0 else 0
        exhaustion = actual_pts / expected_move_pts * 100 if expected_move_pts > 0 else 0

        signal = _build_em_signal(exhaustion, actual_pts, expected_move_pts, current_price_nas100)

        result = ExpectedMove(
            ticker="NAS100",
            current_price=current_price_nas100,
            expected_daily_move_pts=expected_move_pts,
            expected_daily_move_pct=expected_move_pct,
            upper_bound=upper,
            lower_bound=lower,
            actual_move_today_pts=round(actual_pts, 0),
            actual_move_today_pct=round(actual_pct, 2),
            exhaustion_pct=round(exhaustion, 1),
            signal=signal,
            reversal_warning=(exhaustion >= 85),
            expected_move_remaining_pts=round(
                max(0.0, expected_move_pts - actual_pts), 0
            ),
        )

        _store_oi(cache_key, result)
        return result

    except Exception as e:
        print(f"[options_intelligence] Expected move error: {e}")
        return None


def _build_em_signal(exhaustion: float, actual_pts: float,
                     expected_pts: float, price: float) -> str:
    if exhaustion >= 100:
        return (f"⚠️ Expected move EXCEEDED ({actual_pts:.0f} vs ±{expected_pts:.0f} pts). "
                f"High reversal probability — counter-trend entries favored.")
    elif exhaustion >= 85:
        return (f"🔴 {exhaustion:.0f}% of expected move used ({actual_pts:.0f}/{expected_pts:.0f} pts). "
                f"Reversal probability increasing — tighten stops or avoid new entries.")
    elif exhaustion >= 60:
        return (f"🟡 {exhaustion:.0f}% of expected move used. "
                f"Room remains but approach with caution. Prefer short-duration scalps.")
    elif exhaustion >= 30:
        return (f"🟢 {exhaustion:.0f}% of expected move used ({actual_pts:.0f}/{expected_pts:.0f} pts). "
                f"Plenty of range remaining — momentum trades valid.")
    else:
        return (f"🟢 Only {exhaustion:.0f}% of ±{expected_pts:.0f} pt expected move used. "
                f"Full range available — high-probability momentum entries.")


# ── RENDER FUNCTIONS ──────────────────────────────────────────────────────────

def render_oi_heatmap(heatmap: OIHeatmap):
    """Render OI heatmap table for dashboard."""
    st.markdown("**📊 Options OI Heatmap (QQQ → NAS100)**")
    st.caption(f"Expiry: {heatmap.expiry} | Gamma wall: {heatmap.max_call_strike:,.0f} | "
               f"Put wall: {heatmap.max_put_strike:,.0f} | Pin zone: {heatmap.pin_zone:,.0f}")

    # Institutional signal
    sig_color = "#2d9e2d" if "above" in heatmap.signal_text else "#c9302c"
    st.markdown(
        f"<div style='padding:8px;border-radius:6px;background:{sig_color}22;"
        f"border-left:3px solid {sig_color};margin:6px 0'>"
        f"<span style='color:{sig_color}'>{heatmap.signal_text}</span></div>",
        unsafe_allow_html=True
    )

    # Table — show top 8 nearest strikes
    nearest = sorted(heatmap.levels, key=lambda l: abs(l.distance_pct))[:8]
    nearest = sorted(nearest, key=lambda l: l.strike, reverse=True)

    rows = []
    for lv in nearest:
        dist_str = f"{lv.distance_pct:+.1f}%"
        marker   = " ◄" if abs(lv.distance_pct) < 0.3 else ""
        call_str = f"{lv.call_oi:,}" if lv.call_oi > 0 else "—"
        put_str  = f"{lv.put_oi:,}"  if lv.put_oi  > 0 else "—"
        rows.append({
            "Strike": f"{lv.strike:,.0f}{marker}",
            "Call OI": call_str,
            "Put OI":  put_str,
            "Signal":  lv.signal,
            "Dist":    dist_str,
        })

    df = pd.DataFrame(rows)

    def color_signal(val):
        if val == "Resistance":   return "color:#c9302c;font-weight:bold"
        if val == "Support":      return "color:#2d9e2d;font-weight:bold"
        if val == "Pin zone":     return "color:#e6a817;font-weight:bold"
        return ""

    styled = df.style.map(color_signal, subset=["Signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=280)


def render_gex_panel(gex: GEXData):
    """Render GEX regime card."""
    st.markdown("**⚡ Gamma Exposure (GEX)**")
    st.markdown(
        f"<div style='padding:8px;border-radius:6px;background:{gex.regime_color}22;"
        f"border-left:3px solid {gex.regime_color}'>"
        f"<span style='color:{gex.regime_color};font-weight:bold'>"
        f"{'📈' if gex.gamma_regime == 'POSITIVE' else '📉'} "
        f"{gex.gamma_regime} GAMMA</span><br>"
        f"<span style='font-size:0.85em'>{gex.regime_signal}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    st.caption(f"Gamma flip zone: {gex.gamma_flip_price:,.0f} | {gex.lot_guidance}")


def render_expected_move_panel(em: ExpectedMove):
    """Render expected move gauge."""
    st.markdown("**🎯 Expected Daily Move**")
    bar_color = "#8b0000" if em.reversal_warning else (
        "#e6a817" if em.exhaustion_pct >= 60 else "#2d9e2d"
    )
    st.markdown(
        f"±<span style='font-size:1.3em;font-weight:bold'>{em.expected_daily_move_pts:.0f} pts</span> "
        f"<span style='color:#aaa'>({em.expected_daily_move_pct:.1f}%)</span>",
        unsafe_allow_html=True
    )
    c1, c2 = st.columns(2)
    c1.metric("Upper bound", f"{em.upper_bound:,.0f}")
    c2.metric("Lower bound", f"{em.lower_bound:,.0f}")

    st.progress(
        min(em.exhaustion_pct / 100, 1.0),
        text=f"Today's move: {em.actual_move_today_pts:.0f} pts "
             f"({em.exhaustion_pct:.0f}% of expected)"
    )
    if em.reversal_warning:
        st.error(em.signal)
    elif em.exhaustion_pct >= 60:
        st.warning(em.signal)
    else:
        st.success(em.signal)
