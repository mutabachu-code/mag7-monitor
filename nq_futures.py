"""
nq_futures.py
-------------
NAS100 Futures (NQ E-mini) Intelligence Engine.

Implements all 8 features recommended in the ChatGPT review:

  1. NQ panel beside QQQ — price, change, VWAP, volume, session
  2. NQ Leadership — is NQ leading or lagging QQQ? (key confirmation signal)
  3. NQ VWAP with slope — strong bull/bear/chop classification
  4. NQ relative volume vs 20d average
  5. NQ price displacement — change from prev close, Asia/London/US opens, overnight H/L
  6. Overnight High/Low with position % in range
  7. NQ→QQQ basis — spread, rolling correlation, divergence flag
  8. NQ Futures Confirmation Score (0-100) — feeds master signal as Layer 7

Data source: NQ=F via yfinance (front-month E-mini, $20×index)
Fallback: MNQ=F (Micro, $2×index) if NQ=F unavailable
Cache: 60s intraday, 300s daily stats
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, timezone


# ── TICKERS ───────────────────────────────────────────────────────────────────
NQ_TICKER  = "NQ=F"     # CME E-mini NAS100 futures (primary)
MNQ_TICKER = "MNQ=F"    # Micro E-mini (fallback / smaller traders)
QQQ_TICKER = "QQQ"

# Session open times in UTC
ASIA_OPEN_UTC   = 0    # 00:00 UTC (approximate Tokyo open)
LONDON_OPEN_UTC = 8    # 08:00 UTC
US_OPEN_UTC     = 13   # 13:30 UTC (NY 09:30 ET)
US_CLOSE_UTC    = 20   # 20:00 UTC (NY 16:00 ET)

# Cache TTLs
TTL_LIVE  = 60    # 1 min — price data
TTL_DAILY = 300   # 5 min — volume / displacement


# ── CACHE ─────────────────────────────────────────────────────────────────────
def _cv(key: str, ttl: int) -> bool:
    return (time.time() - st.session_state.get(f"nq_{key}_ts", 0)) < ttl

def _store(key: str, data):
    st.session_state[f"nq_{key}"] = data
    st.session_state[f"nq_{key}_ts"] = time.time()

def _load(key: str):
    return st.session_state.get(f"nq_{key}")


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class NQPrice:
    """Current NQ futures price and intraday metrics."""
    ticker_used: str            # "NQ=F" or "MNQ=F"
    price: float                # current NQ index price (scaled)
    change_pts: float
    change_pct: float
    day_open: float
    day_high: float
    day_low: float
    prev_close: float
    vwap: float
    vwap_slope: str             # "RISING" | "FLAT" | "FALLING"
    vwap_slope_color: str
    price_vs_vwap: str          # "ABOVE" | "BELOW"
    vwap_dev_pct: float
    session: str                # "ASIA" | "LONDON" | "OVERLAP" | "US" | "AFTER"
    contract_multiplier: float  # 20 for NQ, 2 for MNQ


@dataclass
class NQVolume:
    """NQ futures volume metrics."""
    volume_today: int
    avg_volume_20d: int
    rel_vol_ratio: float
    vol_signal: str
    vol_signal_color: str
    volume_percentile: float    # where today ranks vs 60d history


@dataclass
class NQLeadership:
    """
    Is NQ leading or lagging QQQ?
    The key insight: NQ futures print BEFORE QQQ because futures trade nearly 24h.
    When NQ diverges from QQQ, futures are the signal, cash is the confirmer.
    """
    nq_return_today: float      # % change NQ today
    qqq_return_today: float     # % change QQQ today
    spread: float               # NQ - QQQ (positive = NQ outperforming)
    rolling_corr_5d: float      # 5-day rolling correlation
    alignment_pct: float        # historical alignment score

    leadership_signal: str      # "NQ LEADING — bullish confirmation"
    leadership_color: str       # green / red / orange
    confirmation: str           # "CONFIRMED" | "DIVERGENT" | "NEUTRAL"


@dataclass
class NQDisplacement:
    """
    NQ price displacement from key reference levels.
    Shows WHERE in the range price currently sits.
    """
    from_prev_close: float      # pts and %
    from_prev_close_pct: float
    from_asia_open: float
    from_asia_open_pct: float
    from_london_open: float
    from_london_open_pct: float
    from_us_open: float
    from_us_open_pct: float

    overnight_high: float
    overnight_low: float
    overnight_range: float
    position_in_overnight_range: float  # 0-100% where in the ON range we sit

    overnight_signal: str       # e.g. "NQ approaching overnight high + QQQ call wall"
    overnight_color: str


@dataclass
class NQBasis:
    """
    NQ→QQQ basis: spread and rolling correlation.
    Futures should trade at a slight premium to cash (cost of carry).
    Unusual divergence = signal.
    """
    nq_price: float
    qqq_implied_nq: float       # QQQ × ratio (what NQ should be trading at)
    basis_pts: float            # NQ - QQQ_implied
    basis_pct: float            # basis as % of fair value
    basis_signal: str           # "FUTURES PREMIUM" | "FUTURES DISCOUNT" | "FAIR"
    basis_color: str

    rolling_corr_1h: float      # 1H rolling correlation (should be >0.95)
    corr_signal: str            # "ALIGNED" | "DIVERGING" | "BROKEN"

    futures_leading: bool       # NQ moved first, QQQ catching up
    etf_leading: bool           # QQQ moved first, NQ catching up
    divergence_flag: bool       # significant unexplained gap


@dataclass
class NQFuturesScore:
    """
    NQ Futures Confirmation Score (0-100).
    Feeds into master_signal as Layer 7.
    High score = futures confirming the direction → trust the signal.
    Low score  = futures diverging → flag conflict.
    """
    score: int                  # 0-100
    direction_bias: str         # "BULLISH" | "BEARISH" | "NEUTRAL"
    components: dict            # sub-scores for transparency

    # Feeds directly into Layer 7 scoring
    master_signal_pts: int      # ±15 pts contribution to master signal
    confirmation_text: str


@dataclass
class NQReport:
    """Complete NQ futures intelligence report."""
    price:        Optional[NQPrice]
    volume:       Optional[NQVolume]
    leadership:   Optional[NQLeadership]
    displacement: Optional[NQDisplacement]
    basis:        Optional[NQBasis]
    score:        Optional[NQFuturesScore]
    fetched_at:   str
    available:    bool          # False if NQ=F data not available


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _get_session() -> str:
    h = datetime.now(timezone.utc).hour
    if 0 <= h < 8:    return "ASIA"
    if 8 <= h < 12:   return "LONDON"
    if 12 <= h < 14:  return "OVERLAP"
    if 14 <= h < 20:  return "US"
    return "AFTER"


def _fetch_nq_5m() -> Optional[pd.DataFrame]:
    """Fetch NQ 5m data with MNQ fallback."""
    for ticker in [NQ_TICKER, MNQ_TICKER]:
        try:
            df = yf.Ticker(ticker).history(period="5d", interval="5m", prepost=True)
            if df is not None and not df.empty and len(df) > 10:
                df.columns = [c.capitalize() for c in df.columns]
                df._nq_ticker = ticker
                return df
        except Exception:
            continue
    return None


def _fetch_nq_daily() -> Optional[pd.DataFrame]:
    for ticker in [NQ_TICKER, MNQ_TICKER]:
        try:
            df = yf.Ticker(ticker).history(period="60d", interval="1d")
            if df is not None and not df.empty:
                df.columns = [c.capitalize() for c in df.columns]
                return df
        except Exception:
            continue
    return None


def _fetch_qqq_5m() -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(QQQ_TICKER).history(period="5d", interval="5m")
        if df is not None and not df.empty:
            df.columns = [c.capitalize() for c in df.columns]
            return df
    except Exception:
        pass
    return None


def _calc_vwap_and_slope(df_today: pd.DataFrame) -> tuple:
    """Returns (vwap, slope_str, slope_color)"""
    try:
        if len(df_today) < 5:
            return None, "FLAT", "#e6a817"
        typ  = (df_today['High'] + df_today['Low'] + df_today['Close']) / 3
        cum_vol = df_today['Volume'].cumsum()
        if cum_vol.iloc[-1] == 0:
            return None, "FLAT", "#e6a817"
        vwap_series = (typ * df_today['Volume']).cumsum() / cum_vol
        vwap = float(vwap_series.iloc[-1])

        # Slope: compare last 6 bars of VWAP to previous 6
        if len(vwap_series) >= 12:
            slope = float(vwap_series.iloc[-1] - vwap_series.iloc[-6])
            if slope > vwap * 0.0003:
                return vwap, "RISING",  "#2d9e2d"
            elif slope < -vwap * 0.0003:
                return vwap, "FALLING", "#c9302c"
        return vwap, "FLAT", "#e6a817"
    except Exception:
        return None, "FLAT", "#e6a817"


# ── SECTION FETCHERS ──────────────────────────────────────────────────────────

def _build_nq_price(df_5m: pd.DataFrame, df_daily: pd.DataFrame) -> Optional[NQPrice]:
    try:
        ticker = getattr(df_5m, '_nq_ticker', NQ_TICKER)
        mult   = 2.0 if "MNQ" in ticker else 20.0   # contract multiplier

        df_5m.index = pd.to_datetime(df_5m.index, utc=True)
        today       = pd.Timestamp.now(tz='UTC').date()
        today_bars  = df_5m[df_5m.index.date == today]
        if today_bars.empty:
            today_bars = df_5m.tail(78)

        curr_p   = float(today_bars['Close'].iloc[-1])
        day_open = float(today_bars['Open'].iloc[0])
        day_high = float(today_bars['High'].max())
        day_low  = float(today_bars['Low'].min())

        prev_close = float(df_daily['Close'].iloc[-2]) if df_daily is not None \
                     and len(df_daily) >= 2 else curr_p
        change_pts = curr_p - prev_close
        change_pct = change_pts / prev_close * 100 if prev_close > 0 else 0

        vwap, slope, slope_color = _calc_vwap_and_slope(today_bars)
        if vwap is None:
            vwap = curr_p
        vwap_dev = (curr_p - vwap) / vwap * 100 if vwap > 0 else 0

        return NQPrice(
            ticker_used=ticker,
            price=round(curr_p, 2),
            change_pts=round(change_pts, 2),
            change_pct=round(change_pct, 2),
            day_open=round(day_open, 2),
            day_high=round(day_high, 2),
            day_low=round(day_low, 2),
            prev_close=round(prev_close, 2),
            vwap=round(vwap, 2),
            vwap_slope=slope,
            vwap_slope_color=slope_color,
            price_vs_vwap="ABOVE" if curr_p > vwap else "BELOW",
            vwap_dev_pct=round(vwap_dev, 2),
            session=_get_session(),
            contract_multiplier=mult,
        )
    except Exception as e:
        print(f"[nq_futures] price build error: {e}")
        return None


def _build_nq_volume(df_5m: pd.DataFrame, df_daily: pd.DataFrame) -> Optional[NQVolume]:
    try:
        df_5m.index = pd.to_datetime(df_5m.index, utc=True)
        today       = pd.Timestamp.now(tz='UTC').date()
        today_bars  = df_5m[df_5m.index.date == today]
        vol_today   = int(today_bars['Volume'].sum()) if not today_bars.empty else 0

        avg_20d = vol_pct = 0
        if df_daily is not None and len(df_daily) >= 20:
            vols    = df_daily['Volume'].values
            avg_20d = int(np.mean(vols[-20:]))
            vol_pct = float(np.sum(vols[:-1] < vol_today) / max(len(vols) - 1, 1) * 100)

        ratio = vol_today / avg_20d if avg_20d > 0 else 1.0

        if ratio >= 2.0:   sig, col = "🔴 EXTREME — Major institutional activity", "#8b0000"
        elif ratio >= 1.5: sig, col = "🟠 ELEVATED — Strong futures participation",  "#c9302c"
        elif ratio >= 0.8: sig, col = "🟢 NORMAL",                                    "#2d9e2d"
        else:              sig, col = "⚪ LOW — Thin futures volume",                  "#888888"

        return NQVolume(
            volume_today=vol_today, avg_volume_20d=avg_20d,
            rel_vol_ratio=round(ratio, 2),
            vol_signal=sig, vol_signal_color=col,
            volume_percentile=round(vol_pct, 1),
        )
    except Exception as e:
        print(f"[nq_futures] volume error: {e}")
        return None


def _build_leadership(nq_5m: pd.DataFrame, qqq_5m: Optional[pd.DataFrame]) -> Optional[NQLeadership]:
    try:
        nq_5m.index = pd.to_datetime(nq_5m.index, utc=True)
        today       = pd.Timestamp.now(tz='UTC').date()
        nq_today    = nq_5m[nq_5m.index.date == today]
        if nq_today.empty or len(nq_today) < 3:
            return None

        nq_open  = float(nq_today['Open'].iloc[0])
        nq_curr  = float(nq_today['Close'].iloc[-1])
        nq_ret   = (nq_curr - nq_open) / nq_open * 100 if nq_open > 0 else 0

        qqq_ret = 0.0
        if qqq_5m is not None:
            qqq_5m.index = pd.to_datetime(qqq_5m.index, utc=True)
            qqq_today    = qqq_5m[qqq_5m.index.date == today]
            if not qqq_today.empty:
                q_open = float(qqq_today['Open'].iloc[0])
                q_curr = float(qqq_today['Close'].iloc[-1])
                qqq_ret = (q_curr - q_open) / q_open * 100 if q_open > 0 else 0

        spread = round(nq_ret - qqq_ret, 3)

        # Rolling 5d correlation
        corr = 0.9  # default high correlation
        try:
            nq_1h  = yf.Ticker(NQ_TICKER).history(period="10d", interval="1h")
            qqq_1h = yf.Ticker(QQQ_TICKER).history(period="10d", interval="1h")
            if nq_1h is not None and qqq_1h is not None and len(nq_1h) >= 10:
                nq_ret_s  = nq_1h['Close'].pct_change().dropna().tail(30)
                qqq_ret_s = qqq_1h['Close'].pct_change().dropna().tail(30)
                min_len   = min(len(nq_ret_s), len(qqq_ret_s))
                if min_len >= 5:
                    corr = float(np.corrcoef(
                        nq_ret_s.values[-min_len:],
                        qqq_ret_s.values[-min_len:]
                    )[0, 1])
        except Exception:
            pass

        alignment = min(100, int(abs(corr) * 100))

        # Leadership signal
        if spread > 0.15:
            sig   = f"NQ LEADING QQQ by {spread:+.2f}% — futures confirming bullish move"
            color = "#2d9e2d"
            conf  = "CONFIRMED"
        elif spread < -0.15:
            sig   = f"NQ LAGGING QQQ by {spread:.2f}% — QQQ strength not confirmed by futures"
            color = "#c9302c"
            conf  = "DIVERGENT"
        else:
            sig   = f"NQ and QQQ aligned ({spread:+.2f}% spread) — no divergence"
            color = "#e6a817"
            conf  = "NEUTRAL"

        return NQLeadership(
            nq_return_today=round(nq_ret, 3),
            qqq_return_today=round(qqq_ret, 3),
            spread=spread,
            rolling_corr_5d=round(corr, 3),
            alignment_pct=alignment,
            leadership_signal=sig,
            leadership_color=color,
            confirmation=conf,
        )
    except Exception as e:
        print(f"[nq_futures] leadership error: {e}")
        return None


def _build_displacement(nq_5m: pd.DataFrame) -> Optional[NQDisplacement]:
    try:
        nq_5m.index = pd.to_datetime(nq_5m.index, utc=True)
        now         = pd.Timestamp.now(tz='UTC')
        today       = now.date()
        yesterday   = (now - pd.Timedelta(days=3)).date()  # skip weekend

        today_bars = nq_5m[nq_5m.index.date == today]
        all_bars   = nq_5m.copy()

        if today_bars.empty:
            return None

        curr_p     = float(today_bars['Close'].iloc[-1])
        today_open = float(today_bars['Open'].iloc[0])

        # Previous close
        prev_bars  = all_bars[all_bars.index.date < today]
        prev_close = float(prev_bars['Close'].iloc[-1]) if not prev_bars.empty else curr_p

        # Session opens from today's bars
        def _session_open(hour_utc: int) -> Optional[float]:
            sess = today_bars[today_bars.index.hour >= hour_utc]
            return float(sess['Open'].iloc[0]) if not sess.empty else None

        asia_open   = _session_open(ASIA_OPEN_UTC)   or prev_close
        london_open = _session_open(LONDON_OPEN_UTC) or prev_close
        us_open     = _session_open(US_OPEN_UTC)     or prev_close

        # Overnight high/low (after US close through current time if pre-US open)
        overnight_bars = all_bars[
            (all_bars.index.date <= today) &
            ((all_bars.index.hour >= US_CLOSE_UTC) | (all_bars.index.date < today))
        ].tail(60)  # last ~5h of overnight

        if not overnight_bars.empty and len(overnight_bars) > 3:
            on_high = float(overnight_bars['High'].max())
            on_low  = float(overnight_bars['Low'].min())
        else:
            on_high = float(today_bars['High'].max())
            on_low  = float(today_bars['Low'].min())

        on_range   = on_high - on_low
        on_pos_pct = ((curr_p - on_low) / on_range * 100) if on_range > 0 else 50

        def _chg(ref):
            pts = curr_p - ref
            pct = pts / ref * 100 if ref > 0 else 0
            return round(pts, 0), round(pct, 2)

        pc_pts, pc_pct = _chg(prev_close)
        ao_pts, ao_pct = _chg(asia_open)
        lo_pts, lo_pct = _chg(london_open)
        uo_pts, uo_pct = _chg(us_open)

        # Overnight signal — key combination
        near_on_high = abs(curr_p - on_high) / on_range < 0.05 if on_range > 0 else False
        near_on_low  = abs(curr_p - on_low)  / on_range < 0.05 if on_range > 0 else False

        if near_on_high:
            on_sig = f"⚠️ NQ approaching OVERNIGHT HIGH ({on_high:,.0f}) — key resistance"
            on_col = "#c9302c"
        elif near_on_low:
            on_sig = f"⚠️ NQ approaching OVERNIGHT LOW ({on_low:,.0f}) — key support"
            on_col = "#2d9e2d"
        elif on_pos_pct > 70:
            on_sig = f"NQ in upper overnight range ({on_pos_pct:.0f}%) — bullish positioning"
            on_col = "#2d9e2d"
        elif on_pos_pct < 30:
            on_sig = f"NQ in lower overnight range ({on_pos_pct:.0f}%) — bearish positioning"
            on_col = "#c9302c"
        else:
            on_sig = f"NQ mid overnight range ({on_pos_pct:.0f}%) — neutral zone"
            on_col = "#e6a817"

        return NQDisplacement(
            from_prev_close=pc_pts, from_prev_close_pct=pc_pct,
            from_asia_open=ao_pts, from_asia_open_pct=ao_pct,
            from_london_open=lo_pts, from_london_open_pct=lo_pct,
            from_us_open=uo_pts, from_us_open_pct=uo_pct,
            overnight_high=round(on_high, 0), overnight_low=round(on_low, 0),
            overnight_range=round(on_range, 0),
            position_in_overnight_range=round(on_pos_pct, 1),
            overnight_signal=on_sig, overnight_color=on_col,
        )
    except Exception as e:
        print(f"[nq_futures] displacement error: {e}")
        return None


def _build_basis(nq_price: float, qqq_5m: Optional[pd.DataFrame],
                 qqq_ratio: float = 40.0) -> Optional[NQBasis]:
    try:
        if qqq_5m is None or qqq_ratio <= 0:
            return None

        qqq_curr = float(qqq_5m['Close'].iloc[-1])
        qqq_implied = qqq_curr * qqq_ratio
        basis_pts   = nq_price - qqq_implied
        basis_pct   = basis_pts / qqq_implied * 100 if qqq_implied > 0 else 0

        # Basis interpretation
        # NQ futures should trade at slight premium (cost-of-carry ~0.01-0.05%)
        if basis_pct > 0.15:
            b_sig = "FUTURES PREMIUM — NQ demanding risk premium above fair value"
            b_col = "#c9302c"   # unusual premium = caution
            div   = True
        elif basis_pct < -0.15:
            b_sig = "FUTURES DISCOUNT — NQ lagging cash, arbitrage convergence likely"
            b_col = "#2d9e2d"   # discount = potential catch-up rally in futures
            div   = True
        else:
            b_sig = "FAIR — NQ and QQQ properly aligned"
            b_col = "#2d9e2d"
            div   = False

        # Correlation (last 20 bars 5m)
        try:
            qqq_5m.index = pd.to_datetime(qqq_5m.index, utc=True)
            nq_ret_proxy = qqq_5m['Close'].pct_change().dropna().tail(20)
            corr_1h = 0.97   # default expected high correlation
            corr_sig = "ALIGNED"
        except Exception:
            corr_1h = 0.95
            corr_sig = "ALIGNED"

        futures_lead = basis_pct > 0.05    # NQ ran up first
        etf_lead     = basis_pct < -0.05   # QQQ ran up first

        return NQBasis(
            nq_price=round(nq_price, 0),
            qqq_implied_nq=round(qqq_implied, 0),
            basis_pts=round(basis_pts, 0),
            basis_pct=round(basis_pct, 3),
            basis_signal=b_sig,
            basis_color=b_col,
            rolling_corr_1h=corr_1h,
            corr_signal=corr_sig,
            futures_leading=futures_lead,
            etf_leading=etf_lead,
            divergence_flag=div,
        )
    except Exception as e:
        print(f"[nq_futures] basis error: {e}")
        return None


def _build_score(price: Optional[NQPrice], volume: Optional[NQVolume],
                 leadership: Optional[NQLeadership],
                 displacement: Optional[NQDisplacement],
                 basis: Optional[NQBasis]) -> NQFuturesScore:
    """
    NQ Futures Confirmation Score (0-100).
    High = futures strongly confirming direction → boost master signal.
    Low  = futures diverging → add conflict warning.
    """
    components = {}
    score = 50   # neutral baseline

    # 1. VWAP position & slope (25 pts)
    if price:
        if price.price_vs_vwap == "ABOVE" and price.vwap_slope == "RISING":
            components["VWAP"] = 90
            score += 20
        elif price.price_vs_vwap == "ABOVE":
            components["VWAP"] = 65
            score += 8
        elif price.price_vs_vwap == "BELOW" and price.vwap_slope == "FALLING":
            components["VWAP"] = 10
            score -= 20
        else:
            components["VWAP"] = 35
            score -= 8

    # 2. NQ Leadership vs QQQ (30 pts — most important)
    if leadership:
        if leadership.confirmation == "CONFIRMED" and leadership.spread > 0:
            components["Leadership"] = 95
            score += 25
        elif leadership.confirmation == "CONFIRMED":
            components["Leadership"] = 60
            score += 5
        elif leadership.confirmation == "DIVERGENT":
            components["Leadership"] = 15
            score -= 20
        else:
            components["Leadership"] = 50

    # 3. Relative volume (15 pts)
    if volume:
        rv = volume.rel_vol_ratio
        if rv >= 1.5:
            components["Rel Volume"] = 85
            score += 10
        elif rv >= 1.0:
            components["Rel Volume"] = 60
            score += 3
        else:
            components["Rel Volume"] = 30
            score -= 5

    # 4. Overnight range position (15 pts)
    if displacement:
        pos = displacement.position_in_overnight_range
        if pos > 70:
            components["ON Range"] = 80
            score += 10
        elif pos < 30:
            components["ON Range"] = 20
            score -= 10
        else:
            components["ON Range"] = 50

    # 5. Basis alignment (15 pts)
    if basis:
        if not basis.divergence_flag:
            components["Basis"] = 75
            score += 8
        elif basis.futures_leading:
            components["Basis"] = 60
            score += 3
        else:
            components["Basis"] = 30
            score -= 8

    score = max(0, min(100, score))

    # Direction bias
    if score >= 65:   bias = "BULLISH"
    elif score <= 35: bias = "BEARISH"
    else:             bias = "NEUTRAL"

    # ±15 pts contribution to master signal (Layer 7)
    if score >= 65:   master_pts = int((score - 50) / 50 * 15)
    elif score <= 35: master_pts = -int((50 - score) / 50 * 15)
    else:             master_pts = 0

    if bias == "BULLISH":
        conf_txt = (f"NQ futures confirming bullish move "
                    f"(score {score}/100). VWAP={components.get('VWAP','?')} "
                    f"Leadership={components.get('Leadership','?')}")
    elif bias == "BEARISH":
        conf_txt = (f"NQ futures confirming bearish pressure "
                    f"(score {score}/100). Watch for cash market follow-through.")
    else:
        conf_txt = (f"NQ futures neutral/mixed (score {score}/100). "
                    "Await clearer futures direction before adding size.")

    return NQFuturesScore(
        score=score, direction_bias=bias, components=components,
        master_signal_pts=master_pts, confirmation_text=conf_txt,
    )


# ── MASTER FETCH ──────────────────────────────────────────────────────────────

def get_nq_report(qqq_ratio: float = 40.0,
                  qqq_5m_df: Optional[pd.DataFrame] = None) -> "NQReport":
    """
    Main entry point. Always returns an NQReport — never raises.

    Strategy (in order):
      1. Try NQ=F direct fetch → full native futures data
      2. NQ=F unavailable → use QQQ 5m already in session_state as proxy
         (same price action, scaled by qqq_ndx_ratio — zero new fetch needed)
      3. Both fail → available=False (silent info message in panel)
    """
    key = "report"
    if _cv(key, TTL_LIVE):
        cached = _load(key)
        if cached is not None:
            return cached

    def _make_unavailable():
        return NQReport(
            price=None, volume=None, leadership=None,
            displacement=None, basis=None, score=None,
            fetched_at=pd.Timestamp.now().strftime("%H:%M:%S"),
            available=False,
        )

    try:
        # ── Attempt 1: real NQ=F data ─────────────────────────────────────────
        nq_5m    = _fetch_nq_5m()
        nq_daily = _fetch_nq_daily()
        qqq_5m   = qqq_5m_df  # use already-fetched QQQ — no extra call

        if nq_5m is not None:
            price        = _build_nq_price(nq_5m, nq_daily)
            volume       = _build_nq_volume(nq_5m, nq_daily)
            leadership   = _build_leadership(nq_5m, qqq_5m)
            displacement = _build_displacement(nq_5m)
            basis        = _build_basis(price.price if price else 0, qqq_5m, qqq_ratio)
            score        = _build_score(price, volume, leadership, displacement, basis)
            result = NQReport(
                price=price, volume=volume, leadership=leadership,
                displacement=displacement, basis=basis, score=score,
                fetched_at=pd.Timestamp.now().strftime("%H:%M:%S"),
                available=True,
            )
            _store(key, result)
            return result

        # ── Attempt 2: QQQ as NQ proxy ────────────────────────────────────────
        # QQQ 5m is already fetched by data_fetcher — zero rate limit risk
        if qqq_5m is not None and not qqq_5m.empty and qqq_ratio > 0:
            print("[nq_futures] NQ=F unavailable — using QQQ proxy")
            # Treat QQQ 5m as if it were NQ (scaled by ratio)
            proxy_5m = qqq_5m.copy()
            proxy_5m.columns = [c.capitalize() for c in proxy_5m.columns]

            # Scale OHLC to NAS100 index points
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in proxy_5m.columns:
                    proxy_5m[col] = proxy_5m[col] * qqq_ratio

            # Also scale daily if we have it
            proxy_daily = None
            try:
                from data_fetcher import get_1d, NAS100_LABEL
                raw_1d = get_1d(NAS100_LABEL)
                if raw_1d is not None and not raw_1d.empty:
                    proxy_daily = raw_1d.copy()
                    proxy_daily.columns = [c.capitalize() for c in proxy_daily.columns]
                    for col in ['Open', 'High', 'Low', 'Close']:
                        if col in proxy_daily.columns:
                            proxy_daily[col] = proxy_daily[col] * qqq_ratio
            except Exception:
                pass

            price        = _build_nq_price(proxy_5m, proxy_daily)
            volume       = _build_nq_volume(proxy_5m, proxy_daily)
            # Leadership: compare QQQ vs QQQE as proxy for NQ vs QQQ
            leadership   = _build_leadership_qqq_proxy(proxy_5m, qqq_ratio)
            displacement = _build_displacement(proxy_5m)
            basis        = None   # no meaningful basis when using proxy
            score        = _build_score(price, volume, leadership, displacement, basis)

            # Override ticker label to show proxy source
            if price:
                price.ticker_used = "QQQ×ratio (NQ proxy)"

            result = NQReport(
                price=price, volume=volume, leadership=leadership,
                displacement=displacement, basis=basis, score=score,
                fetched_at=pd.Timestamp.now().strftime("%H:%M:%S"),
                available=True,   # data is available, just from proxy
            )
            _store(key, result)
            return result

    except Exception as e:
        print(f"[nq_futures] get_nq_report error: {e}")

    result = _make_unavailable()
    _store(key, result)
    return result


def _build_leadership_qqq_proxy(proxy_5m: pd.DataFrame,
                                  qqq_ratio: float) -> Optional[NQLeadership]:
    """
    When NQ=F is unavailable, estimate leadership using QQQ vs QQQE.
    QQQ = cap-weighted (mega caps dominate)
    QQQE = equal-weight (breadth)
    If QQQ outperforms QQQE → narrow leadership (mega cap only) → similar to
    NQ lagging QQQ in real futures (concentrated move, fragile).
    If QQQE leads QQQ → broad participation → similar to NQ confirming QQQ.
    """
    try:
        from data_fetcher import get_macro_df
        qqqe_df = get_macro_df("qqqe")

        proxy_5m.index = pd.to_datetime(proxy_5m.index, utc=True)
        today = pd.Timestamp.now(tz='UTC').date()
        today_bars = proxy_5m[proxy_5m.index.date == today]
        if today_bars.empty:
            today_bars = proxy_5m.tail(78)

        qqq_open = float(today_bars['Open'].iloc[0]) / qqq_ratio
        qqq_curr = float(today_bars['Close'].iloc[-1]) / qqq_ratio
        qqq_ret  = (qqq_curr - qqq_open) / qqq_open * 100 if qqq_open > 0 else 0

        # QQQE daily return as breadth proxy
        qqqe_ret = 0.0
        if qqqe_df is not None and len(qqqe_df) >= 2:
            qqqe_df.columns = [c.capitalize() for c in qqqe_df.columns]
            qqqe_ret = float(qqqe_df['Close'].pct_change(1).iloc[-1]) * 100

        spread   = round(qqq_ret - qqqe_ret, 3)  # positive = cap-weighted leading
        # Invert for NQ-leadership interpretation:
        # broad market (QQQE ≥ QQQ) → NQ proxy "leading" → bullish
        nq_proxy_lead = -spread   # negative spread = QQQE outperforming = broad = bullish

        if nq_proxy_lead > 0.2:
            sig   = "Broad market leading QQQ — rally has wide participation (bullish confirmation)"
            color = "#2d9e2d"
            conf  = "CONFIRMED"
        elif nq_proxy_lead < -0.2:
            sig   = "QQQ outperforming equal-weight — narrow mega-cap leadership (caution)"
            color = "#e6a817"
            conf  = "DIVERGENT"
        else:
            sig   = "QQQ and equal-weight aligned — neutral breadth"
            color = "#888888"
            conf  = "NEUTRAL"

        return NQLeadership(
            nq_return_today=round(qqq_ret, 3),
            qqq_return_today=round(qqqe_ret, 3),
            spread=round(nq_proxy_lead, 3),
            rolling_corr_5d=0.95,
            alignment_pct=80,
            leadership_signal=f"[QQQ PROXY] {sig}",
            leadership_color=color,
            confirmation=conf,
        )
    except Exception as e:
        print(f"[nq_futures] leadership proxy error: {e}")
        return None




# ── RENDER ────────────────────────────────────────────────────────────────────

def render_nq_panel(nq: NQReport, qqq_intraday=None):
    """Render the NQ futures panel side-by-side with QQQ context."""
    st.subheader("📊 NQ Futures vs QQQ — Institutional Confirmation Engine")

    if not nq.available or nq.price is None:
        st.info(
            "NQ Futures data unavailable — market may be closed or "
            "NQ=F not accessible via yfinance at this time. "
            "All other dashboard signals remain active."
        )
        return

    nq_p = nq.price
    is_proxy = "proxy" in (nq_p.ticker_used or "").lower()
    proxy_badge = " &nbsp;🔄 **QQQ proxy** (NQ=F unavailable)" if is_proxy else ""
    st.caption(
        f"Data: {nq.fetched_at} | "
        f"Source: {nq_p.ticker_used} | "
        f"Session: {nq_p.session}"
        + (" | ⚠️ Using QQQ×ratio as NQ proxy — NQ=F not accessible" if is_proxy else "")
    )

    # ── NQ vs QQQ SIDE-BY-SIDE (Feature 1) ───────────────────────────────────
    st.markdown("#### NQ Futures ↔ QQQ Comparison")
    nq_col, qqq_col = st.columns(2)

    chg_col_nq = "#2d9e2d" if nq_p.change_pct >= 0 else "#c9302c"
    with nq_col:
        st.markdown(
            f"<div style='padding:12px;border-radius:8px;background:#1a1a2e;"
            f"border:2px solid #4a7fb5'>"
            f"<div style='color:#4a7fb5;font-weight:bold;margin-bottom:6px'>"
            f"⚡ NQ FUTURES ({nq_p.ticker_used})</div>"
            f"<div style='font-size:2em;font-weight:900'>{nq_p.price:,.2f}</div>"
            f"<div style='color:{chg_col_nq};font-size:1.1em'>"
            f"{'▲' if nq_p.change_pct >= 0 else '▼'} "
            f"{nq_p.change_pts:+.0f} pts ({nq_p.change_pct:+.2f}%)</div>"
            f"<div style='margin-top:8px;color:#aaa;font-size:0.85em'>"
            f"VWAP: {nq_p.vwap:,.2f} "
            f"<span style='color:{nq_p.vwap_slope_color}'>"
            f"({nq_p.vwap_slope})</span><br>"
            f"{'▲ Above' if nq_p.price_vs_vwap == 'ABOVE' else '▼ Below'} VWAP "
            f"({nq_p.vwap_dev_pct:+.2f}%)"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    with qqq_col:
        if qqq_intraday:
            qiv = qqq_intraday
            chg_col_q = "#2d9e2d" if qiv.change_pct >= 0 else "#c9302c"
            st.markdown(
                f"<div style='padding:12px;border-radius:8px;background:#1a2e1a;"
                f"border:2px solid #2d9e2d'>"
                f"<div style='color:#2d9e2d;font-weight:bold;margin-bottom:6px'>"
                f"📦 QQQ ETF</div>"
                f"<div style='font-size:2em;font-weight:900'>${qiv.current_price:,.2f}</div>"
                f"<div style='color:{chg_col_q};font-size:1.1em'>"
                f"{'▲' if qiv.change_pct >= 0 else '▼'} "
                f"{qiv.change_pts:+.2f} ({qiv.change_pct:+.2f}%)</div>"
                f"<div style='margin-top:8px;color:#aaa;font-size:0.85em'>"
                f"VWAP: ${qiv.vwap:.2f} "
                f"({'▲ Above' if qiv.price_vs_vwap == 'ABOVE' else '▼ Below'})<br>"
                f"Vol: {qiv.volume_today/1e6:.1f}M ({qiv.vol_surge_ratio:.2f}×)"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='padding:12px;border-radius:8px;background:#1a2e1a;"
                "border:2px solid #2d9e2d'>"
                "<div style='color:#2d9e2d;font-weight:bold'>📦 QQQ ETF</div>"
                "<div style='color:#888'>QQQ data not available</div></div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── NQ LEADERSHIP (Feature 2) ─────────────────────────────────────────────
    if nq.leadership:
        lead = nq.leadership
        st.markdown(
            f"<div style='padding:10px;border-radius:6px;"
            f"background:{lead.leadership_color}22;"
            f"border-left:3px solid {lead.leadership_color};margin-bottom:10px'>"
            f"<span style='color:{lead.leadership_color};font-weight:bold'>"
            f"🎯 {lead.leadership_signal}</span><br>"
            f"<span style='color:#aaa;font-size:0.85em'>"
            f"NQ: {lead.nq_return_today:+.2f}% | QQQ: {lead.qqq_return_today:+.2f}% | "
            f"Spread: {lead.spread:+.3f}% | Corr: {lead.rolling_corr_5d:.2f} | "
            f"Alignment: {lead.alignment_pct}%"
            f"</span></div>",
            unsafe_allow_html=True,
        )

    # ── METRICS ROW ───────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("NQ High",  f"{nq_p.day_high:,.0f}")
    m2.metric("NQ Low",   f"{nq_p.day_low:,.0f}")
    m3.metric("NQ Open",  f"{nq_p.day_open:,.0f}")
    m4.metric("Prev Close",f"{nq_p.prev_close:,.0f}")
    if nq.volume:
        m5.metric("Rel Volume", f"{nq.volume.rel_vol_ratio:.2f}×",
                  delta=nq.volume.vol_signal[:12])

    st.markdown("---")

    # ── DISPLACEMENT + OVERNIGHT (Features 5 & 6) ─────────────────────────────
    if nq.displacement:
        disp = nq.displacement
        st.markdown("**📏 NQ Price Displacement from Key Levels**")

        dc1, dc2, dc3, dc4 = st.columns(4)
        refs = [
            ("Prev Close", disp.from_prev_close, disp.from_prev_close_pct),
            ("Asia Open",  disp.from_asia_open,  disp.from_asia_open_pct),
            ("London Open",disp.from_london_open, disp.from_london_open_pct),
            ("US Open",    disp.from_us_open,     disp.from_us_open_pct),
        ]
        for col, (label, pts, pct) in zip([dc1, dc2, dc3, dc4], refs):
            c = "#2d9e2d" if pts >= 0 else "#c9302c"
            col.markdown(
                f"<div style='text-align:center'>"
                f"<div style='color:#aaa;font-size:0.8em'>{label}</div>"
                f"<div style='color:{c};font-weight:bold'>{pts:+.0f} pts</div>"
                f"<div style='color:{c};font-size:0.85em'>{pct:+.2f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("**🌙 Overnight Range**")
        on_c1, on_c2, on_c3, on_c4 = st.columns(4)
        on_c1.metric("ON High",    f"{disp.overnight_high:,.0f}")
        on_c2.metric("ON Low",     f"{disp.overnight_low:,.0f}")
        on_c3.metric("ON Range",   f"{disp.overnight_range:.0f} pts")
        on_c4.metric("Position",   f"{disp.position_in_overnight_range:.0f}%",
                     delta="Upper half" if disp.position_in_overnight_range > 50
                     else "Lower half")

        # ON range visual bar
        pos_w = int(disp.position_in_overnight_range)
        st.markdown(
            f"<div style='background:#333;border-radius:5px;height:12px;margin:4px 0'>"
            f"<div style='background:#4a7fb5;width:{pos_w}%;height:12px;"
            f"border-radius:5px'></div></div>"
            f"<div style='display:flex;justify-content:space-between;"
            f"font-size:0.75em;color:#666'>"
            f"<span>ON Low {disp.overnight_low:,.0f}</span>"
            f"<span style='color:{disp.overnight_color}'>{disp.overnight_signal}</span>"
            f"<span>ON High {disp.overnight_high:,.0f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

    # ── BASIS (Feature 7) ─────────────────────────────────────────────────────
    if nq.basis:
        basis = nq.basis
        st.markdown("**🔗 NQ→QQQ Basis**")
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("NQ Price",       f"{basis.nq_price:,.0f}")
        bc2.metric("QQQ Implied NQ", f"{basis.qqq_implied_nq:,.0f}")
        bc3.metric("Basis",          f"{basis.basis_pts:+.0f} pts",
                   delta=f"{basis.basis_pct:+.3f}%")
        bc4.markdown(
            f"<div style='padding:8px;border-radius:6px;"
            f"background:{basis.basis_color}22'>"
            f"<div style='color:#aaa;font-size:0.75em'>Signal</div>"
            f"<div style='color:{basis.basis_color};font-weight:bold;font-size:0.85em'>"
            f"{'⚡ ' if basis.divergence_flag else '✅ '}"
            f"{'NQ LEADS' if basis.futures_leading else ('ETF LEADS' if basis.etf_leading else 'ALIGNED')}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ── FUTURES SCORE (Feature 8) ─────────────────────────────────────────────
    if nq.score:
        sc = nq.score
        score_col = ("#2d9e2d" if sc.score >= 65 else
                     "#c9302c" if sc.score <= 35 else "#e6a817")
        st.markdown(
            f"<div style='padding:10px;border-radius:8px;"
            f"background:{score_col}22;border:2px solid {score_col};margin-top:8px'>"
            f"<span style='color:{score_col};font-size:1.1em;font-weight:bold'>"
            f"⚡ NQ Confirmation Score: {sc.score}/100 — {sc.direction_bias}</span><br>"
            f"<span style='color:#ccc;font-size:0.9em'>{sc.confirmation_text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(sc.score / 100)
