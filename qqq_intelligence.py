"""
qqq_intelligence.py
-------------------
QQQ ETF Intelligence Module for the Mag7 + NAS100 Dashboard.

Covers 7 data categories:
  1. Price & Volume (intraday 5m + live metrics)
  2. Daily Volume Analysis (unusual volume detection)
  3. NAV & Fund Information
  4. Top Holdings (live weights where available)
  5. Sector Weighting
  6. Historical Performance (1d to 5y)
  7. Options Chain (calls/puts, PCR, max pain, IV skew)

Rate-limit strategy:
  - Each section has its own TTL-based cache
  - Intraday data: 60s cache (syncs with dashboard refresh)
  - Fund info / performance: 900s (15 min)
  - Options chain: 300s (5 min)
  - Holdings / sectors: 86400s (24h — rarely changes)
  - All fetches wrapped in try/except — failures show warnings, never crash
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
QQQ_TICKER = "QQQ"
NDX_TICKER = "^NDX"

# Static top-10 QQQ holdings (weights updated periodically by Invesco)
# These are used as fallback when live fund data is unavailable
QQQ_TOP_HOLDINGS = [
    ("AAPL",  "Apple Inc",              20.81),
    ("MSFT",  "Microsoft Corp",         19.19),
    ("NVDA",  "NVIDIA Corp",             8.52),
    ("AMZN",  "Amazon.com Inc",          5.23),
    ("META",  "Meta Platforms",           4.84),
    ("TSLA",  "Tesla Inc",               3.52),
    ("GOOGL", "Alphabet Inc Class A",     2.83),
    ("GOOG",  "Alphabet Inc Class C",     2.65),
    ("AVGO",  "Broadcom Inc",             2.43),
    ("COST",  "Costco Wholesale",         2.11),
]

# QQQ sector allocation (Invesco published, approximate)
QQQ_SECTORS = {
    "Information Technology": 52.4,
    "Consumer Discretionary": 13.8,
    "Communication Services":  8.6,
    "Health Care":             5.9,
    "Consumer Staples":        4.6,
    "Industrials":             4.4,
    "Financials":              1.8,
    "Energy":                  0.8,
    "Other":                   7.7,
}

SECTOR_COLORS = {
    "Information Technology": "#4e79a7",
    "Consumer Discretionary": "#f28e2b",
    "Communication Services": "#e15759",
    "Health Care":            "#76b7b2",
    "Consumer Staples":       "#59a14f",
    "Industrials":            "#edc948",
    "Financials":             "#b07aa1",
    "Energy":                 "#ff9da7",
    "Other":                  "#9c755f",
}

# Cache TTLs
TTL_INTRADAY    = 60      # 1 min  — price/volume
TTL_DAILY       = 300     # 5 min  — volume analysis
TTL_FUND        = 900     # 15 min — NAV/info/performance
TTL_OPTIONS     = 300     # 5 min  — options chain
TTL_STATIC      = 86400   # 24 h   — holdings/sectors


# ── CACHE HELPERS ─────────────────────────────────────────────────────────────
def _cache_valid(key: str, ttl: int) -> bool:
    return (time.time() - st.session_state.get(f"{key}_ts", 0)) < ttl

def _store(key: str, data):
    st.session_state[key] = data
    st.session_state[f"{key}_ts"] = time.time()

def _load(key: str):
    return st.session_state.get(key)


# ── DATA CLASSES ──────────────────────────────────────────────────────────────

@dataclass
class QQQIntraday:
    current_price: float
    prev_close: float
    change_pts: float
    change_pct: float
    day_open: float
    day_high: float
    day_low: float
    intraday_range_pct: float   # (high-low)/open * 100
    vwap: float
    price_vs_vwap: str          # "ABOVE" | "BELOW"
    vwap_dev_pct: float
    volume_today: int
    avg_volume_20d: int
    vol_surge_ratio: float      # session-adjusted pace ratio (primary)
    vol_ratio_raw: float        # raw today_total / 20d_avg
    vol_pace_ratio: float       # today_vol / expected_by_now (time-adjusted)
    vol_accel: float            # late bars / early bars acceleration
    vol_signal: str
    vol_signal_color: str
    dollar_volume_today: float  # price × volume (billions)
    session_frac: float         # 0-1, how far through NY session
    bars_5m: Optional[pd.DataFrame] = None   # raw 5m data for charting


@dataclass
class VolumeAnalysis:
    volumes_20d: List[float]
    avg_20d: float
    avg_5d: float
    today_vol: float
    vol_ratio_vs_20d: float
    vol_percentile: float       # today vol vs 60d history
    unusual_threshold: float    # 20d avg × 1.5
    is_unusual: bool
    trend_5d: str               # "INCREASING" | "DECREASING" | "FLAT"
    biggest_vol_day: str        # date of highest vol in 60d
    dollar_vol_avg: float       # avg daily dollar volume (millions)
    signal: str
    signal_color: str


@dataclass
class FundInfo:
    total_assets_bn: float      # AUM in billions
    nav_price: float
    expense_ratio: float        # %
    inception_date: str
    fund_family: str
    category: str
    ytd_return: float
    one_year_return: float
    three_year_return: float
    five_year_return: float
    beta: float
    pe_ratio: float
    dividend_yield: float
    shares_outstanding: int
    market_cap_bn: float
    premium_discount: float     # (price - NAV) / NAV * 100


@dataclass
class HoldingData:
    ticker: str
    name: str
    weight: float
    price: Optional[float]
    change_pct_1d: Optional[float]
    change_pct_5d: Optional[float]
    above_sma20: Optional[bool]
    contribution_signal: str    # "LIFTING" | "NEUTRAL" | "DRAGGING"


@dataclass
class PerformanceData:
    ret_1d: float
    ret_5d: float
    ret_1m: float
    ret_3m: float
    ret_ytd: float
    ret_1y: float
    ret_3y: float
    ret_5y: float
    high_52w: float
    low_52w: float
    pct_from_52w_high: float
    pct_from_52w_low: float
    volatility_30d: float       # annualized
    max_drawdown_1y: float
    sharpe_1y: float            # proxy (return / vol)
    vs_spy_1y: float            # QQQ vs SPY relative return


@dataclass
class OptionStrike:
    strike: float
    call_last: float
    call_bid: float
    call_ask: float
    call_oi: int
    call_vol: int
    call_iv: float
    put_last: float
    put_bid: float
    put_ask: float
    put_oi: int
    put_vol: int
    put_iv: float
    is_atm: bool
    distance_pct: float


@dataclass
class OptionsData:
    expiry: str
    available_expiries: List[str]
    atm_strike: float
    atm_call_iv: float
    atm_put_iv: float
    iv_skew: float              # put_iv - call_iv (positive = put premium = fear)
    put_call_ratio_oi: float    # total put OI / total call OI
    put_call_ratio_vol: float   # today's put vol / call vol
    max_pain: float             # strike where total OI loss is minimised
    gamma_wall_call: float      # largest call OI strike
    gamma_wall_put: float       # largest put OI strike
    strikes: List[OptionStrike]
    unusual_calls: List[Dict]   # strikes with vol >> OI (fresh positioning)
    unusual_puts: List[Dict]
    pcr_signal: str             # "BEARISH FEAR" | "NEUTRAL" | "BULLISH COMPLACENCY"
    pcr_signal_color: str
    expected_move_1sd: float    # straddle price ≈ 1σ expected move
    expected_move_pct: float


@dataclass
class QQQReport:
    """Complete QQQ intelligence report — all 7 sections."""
    intraday:    Optional[QQQIntraday]
    volume:      Optional[VolumeAnalysis]
    fund:        Optional[FundInfo]
    holdings:    List[HoldingData]
    performance: Optional[PerformanceData]
    options:     Optional[OptionsData]
    fetched_at:  str
    errors:      List[str]


# ── SECTION 1: INTRADAY PRICE & VOLUME ────────────────────────────────────────

def _fetch_intraday() -> Optional[QQQIntraday]:
    key = "qqq_intraday"
    if _cache_valid(key, TTL_INTRADAY):
        return _load(key)

    try:
        q      = yf.Ticker(QQQ_TICKER)
        df_5m  = q.history(period="2d", interval="5m", prepost=True)
        df_1d  = q.history(period="60d", interval="1d")

        if df_5m is None or df_5m.empty:
            return _load(key)  # return stale

        df_5m.columns = [c.capitalize() for c in df_5m.columns]
        df_1d.columns = [c.capitalize() for c in df_1d.columns] if df_1d is not None else []

        # Today's bars only
        df_5m.index = pd.to_datetime(df_5m.index, utc=True)
        today       = pd.Timestamp.now(tz='UTC').date()
        today_bars  = df_5m[df_5m.index.date == today]

        if today_bars.empty:
            today_bars = df_5m.tail(78)  # fallback: last session

        curr_p   = float(today_bars['Close'].iloc[-1])
        day_open = float(today_bars['Open'].iloc[0])
        day_high = float(today_bars['High'].max())
        day_low  = float(today_bars['Low'].min())

        prev_close = float(df_1d['Close'].iloc[-2]) if df_1d is not None and len(df_1d) >= 2 else curr_p
        change_pts = curr_p - prev_close
        change_pct = change_pts / prev_close * 100 if prev_close > 0 else 0
        intraday_range_pct = (day_high - day_low) / day_open * 100 if day_open > 0 else 0

        # VWAP
        typ  = (today_bars['High'] + today_bars['Low'] + today_bars['Close']) / 3
        vwap = float((typ * today_bars['Volume']).cumsum().iloc[-1] /
                     today_bars['Volume'].cumsum().iloc[-1]) if today_bars['Volume'].sum() > 0 else curr_p
        vwap_dev = (curr_p - vwap) / vwap * 100 if vwap > 0 else 0

        # Volume
        vol_today = int(today_bars['Volume'].sum())
        avg_20d   = int(df_1d['Volume'].tail(20).mean()) if df_1d is not None and len(df_1d) >= 20 else vol_today
        vol_ratio_raw = vol_today / avg_20d if avg_20d > 0 else 1.0
        dollar_vol = curr_p * vol_today / 1e9  # billions

        # ── FIX 1: Session-time-adjusted benchmark ────────────────────────────
        # Compare vol to what's EXPECTED by this point in the session
        # NY session = 09:30-16:00 ET = 13:30-20:00 UTC = 6.5h = 78 bars
        utc_now      = pd.Timestamp.now(tz='UTC')
        ny_open_utc  = utc_now.replace(hour=13, minute=30, second=0, microsecond=0)
        ny_close_utc = utc_now.replace(hour=20, minute=0,  second=0, microsecond=0)
        session_elapsed = max(0.0, (utc_now - ny_open_utc).total_seconds() / 3600)
        session_total   = 6.5  # hours
        session_frac    = min(session_elapsed / session_total, 1.0) if session_elapsed > 0 else 1.0

        # Expected volume by this time of day
        expected_by_now = avg_20d * session_frac if session_frac > 0.05 else avg_20d
        pace_ratio      = vol_today / expected_by_now if expected_by_now > 0 else vol_ratio_raw

        # ── FIX 2: Volume acceleration (last 6 bars vs previous 6) ────────────
        vol_accel = 1.0
        if len(today_bars) >= 12:
            early_bars = today_bars['Volume'].iloc[-12:-6].mean()
            late_bars  = today_bars['Volume'].iloc[-6:].mean()
            vol_accel  = late_bars / early_bars if early_bars > 0 else 1.0

        # ── FIX 3: Volume × Price direction matrix ────────────────────────────
        price_rising = change_pct > 0.1   # up on the day
        price_flat   = abs(change_pct) <= 0.1
        price_falling= change_pct < -0.1
        vol_building = pace_ratio >= 1.15 or vol_accel >= 1.3  # accelerating

        # Classification — pace_ratio is the primary measure
        if pace_ratio >= 2.0:
            if price_rising:
                vol_sig   = "🟢 ACCUMULATION — Unusual volume + rising price. Institutional buying. Chase."
                vol_color = "#1a7a1a"
            else:
                vol_sig   = "🔴 DISTRIBUTION — Unusual volume + falling price. Institutional selling."
                vol_color = "#8b0000"
        elif pace_ratio >= 1.4:
            if price_rising:
                vol_sig   = "🟢 ELEVATED + Rising — Volume building into rally. Conviction confirmed."
                vol_color = "#2d9e2d"
            elif price_falling:
                vol_sig   = "🟠 ELEVATED + Falling — Volume building into decline. Distribution."
                vol_color = "#c9302c"
            else:
                vol_sig   = "🟠 ELEVATED — Above-pace volume. Watch for directional break."
                vol_color = "#e6a817"
        elif pace_ratio >= 0.85:
            if vol_accel >= 1.4 and price_rising:
                vol_sig   = "🟡 ACCELERATING — Volume pace picking up into rally. Momentum building."
                vol_color = "#e6a817"
            elif vol_accel >= 1.4 and price_falling:
                vol_sig   = "🟡 ACCELERATING — Volume picking up into decline. Watch for continuation."
                vol_color = "#e6a817"
            else:
                vol_sig   = "🟢 ON PACE — Normal session volume."
                vol_color = "#2d9e2d"
        else:
            # Below pace — but is price moving anyway?
            if price_rising:
                vol_sig   = "⚪ LOW CONVICTION RALLY — Price rising on below-pace volume. Fade at resistance."
                vol_color = "#888888"
            elif price_falling:
                vol_sig   = "🟡 LOW VOLUME DECLINE — Sellers not committing. Possible exhaustion / bounce."
                vol_color = "#e6a817"
            else:
                vol_sig   = "⚪ THIN — Below-pace volume. Wait for catalyst."
                vol_color = "#888888"

        # Use pace_ratio as the primary surge ratio (more accurate than raw ratio)
        vol_surge_display = pace_ratio

        result = QQQIntraday(
            current_price=round(curr_p, 2),
            prev_close=round(prev_close, 2),
            change_pts=round(change_pts, 2),
            change_pct=round(change_pct, 2),
            day_open=round(day_open, 2),
            day_high=round(day_high, 2),
            day_low=round(day_low, 2),
            intraday_range_pct=round(intraday_range_pct, 2),
            vwap=round(vwap, 2),
            price_vs_vwap="ABOVE" if curr_p > vwap else "BELOW",
            vwap_dev_pct=round(vwap_dev, 2),
            volume_today=vol_today,
            avg_volume_20d=avg_20d,
            vol_surge_ratio=round(vol_surge_display, 2),
            vol_ratio_raw=round(vol_ratio_raw, 2),
            vol_pace_ratio=round(pace_ratio, 2),
            vol_accel=round(vol_accel, 2),
            vol_signal=vol_sig,
            vol_signal_color=vol_color,
            dollar_volume_today=round(dollar_vol, 2),
            session_frac=round(session_frac, 2),
            bars_5m=today_bars,
        )
        _store(key, result)
        return result
    except Exception as e:
        print(f"[qqq] Intraday error: {e}")
        return _load(key)


# ── SECTION 2: DAILY VOLUME ANALYSIS ─────────────────────────────────────────

def _fetch_volume_analysis() -> Optional[VolumeAnalysis]:
    key = "qqq_volume"
    if _cache_valid(key, TTL_DAILY):
        return _load(key)

    try:
        q    = yf.Ticker(QQQ_TICKER)
        df   = q.history(period="60d", interval="1d")
        if df is None or df.empty or len(df) < 10:
            return _load(key)

        df.columns = [c.capitalize() for c in df.columns]
        vols  = df['Volume'].values
        close = df['Close'].values

        avg_20d   = float(np.mean(vols[-20:]))
        avg_5d    = float(np.mean(vols[-5:]))
        today_vol = float(vols[-1])
        ratio_20d = today_vol / avg_20d if avg_20d > 0 else 1.0

        # Volume percentile (where does today rank in 60d?)
        vol_pct = float(np.sum(vols[:-1] < today_vol) / len(vols[:-1]) * 100)

        # Trend: is volume increasing or decreasing over 5d?
        vol_5d_chg = (avg_5d - float(np.mean(vols[-10:-5]))) / float(np.mean(vols[-10:-5])) * 100 \
                     if len(vols) >= 10 else 0
        trend = "INCREASING" if vol_5d_chg > 5 else ("DECREASING" if vol_5d_chg < -5 else "FLAT")

        # Biggest vol day
        biggest_idx  = int(np.argmax(vols))
        biggest_date = str(df.index[biggest_idx])[:10]

        avg_dollar_vol = float(np.mean(close[-20:] * vols[-20:])) / 1e6  # millions

        unusual_thresh = avg_20d * 1.5
        is_unusual     = today_vol >= unusual_thresh

        if ratio_20d >= 2.5:
            sig   = "🔴 EXTREME VOLUME — Major institutional event"
            color = "#8b0000"
        elif ratio_20d >= 1.8:
            sig   = "🟠 VERY HIGH — Strong directional conviction"
            color = "#c9302c"
        elif ratio_20d >= 1.4:
            sig   = "🟡 ELEVATED — Above-normal interest"
            color = "#e6a817"
        elif ratio_20d >= 0.7:
            sig   = "🟢 NORMAL — Typical daily volume"
            color = "#2d9e2d"
        else:
            sig   = "⚪ BELOW AVERAGE — Low conviction move"
            color = "#888888"

        result = VolumeAnalysis(
            volumes_20d=list(vols[-20:].astype(int)),
            avg_20d=round(avg_20d),
            avg_5d=round(avg_5d),
            today_vol=round(today_vol),
            vol_ratio_vs_20d=round(ratio_20d, 2),
            vol_percentile=round(vol_pct, 1),
            unusual_threshold=round(unusual_thresh),
            is_unusual=is_unusual,
            trend_5d=trend,
            biggest_vol_day=biggest_date,
            dollar_vol_avg=round(avg_dollar_vol, 1),
            signal=sig,
            signal_color=color,
        )
        _store(key, result)
        return result
    except Exception as e:
        print(f"[qqq] Volume error: {e}")
        return _load(key)


# ── SECTION 3: NAV & FUND INFO ────────────────────────────────────────────────

def _fetch_fund_info() -> Optional[FundInfo]:
    key = "qqq_fund"
    if _cache_valid(key, TTL_FUND):
        return _load(key)

    try:
        q    = yf.Ticker(QQQ_TICKER)
        info = q.info or {}
        fi   = q.fast_info

        def _g(d, *keys, default=0.0):
            for k in keys:
                v = d.get(k)
                if v is not None:
                    try: return float(v)
                    except: pass
            return default

        def _gs(d, *keys, default="N/A"):
            for k in keys:
                v = d.get(k)
                if v: return str(v)
            return default

        curr_price = _g(info, "regularMarketPrice", "previousClose",
                        default=getattr(fi, "last_price", 0) or 0)
        nav        = _g(info, "navPrice", default=curr_price)
        prem_disc  = (curr_price - nav) / nav * 100 if nav > 0 else 0

        total_assets = _g(info, "totalAssets", default=0) / 1e9

        result = FundInfo(
            total_assets_bn=round(total_assets, 2),
            nav_price=round(nav, 2),
            expense_ratio=round(_g(info, "annualReportExpenseRatio") * 100
                                if info.get("annualReportExpenseRatio") else 0.20, 3),
            inception_date="Mar 10, 1999",   # QQQ inception — static
            fund_family=_gs(info, "fundFamily", default="Invesco"),
            category=_gs(info, "category", default="Large Growth"),
            ytd_return=round(_g(info, "ytdReturn") * 100, 2),
            one_year_return=round(_g(info, "oneYearReturn",
                                     "trailingAnnualReturnRate") * 100, 2),
            three_year_return=round(_g(info, "threeYearAverageReturn") * 100, 2),
            five_year_return=round(_g(info, "fiveYearAverageReturn") * 100, 2),
            beta=round(_g(info, "beta3Year", "beta", default=1.0), 2),
            pe_ratio=round(_g(info, "trailingPE", "forwardPE", default=0), 1),
            dividend_yield=round(_g(info, "yield", "dividendYield") * 100
                                  if info.get("yield") else 0, 3),
            shares_outstanding=int(_g(info, "sharesOutstanding", default=0)),
            market_cap_bn=round(_g(info, "marketCap", default=0) / 1e9, 2),
            premium_discount=round(prem_disc, 3),
        )
        _store(key, result)
        return result
    except Exception as e:
        print(f"[qqq] Fund info error: {e}")
        return _load(key)


# ── SECTION 4: TOP HOLDINGS ───────────────────────────────────────────────────

def _fetch_holdings() -> List[HoldingData]:
    key = "qqq_holdings"
    if _cache_valid(key, TTL_STATIC):
        cached = _load(key)
        if cached:
            return cached

    try:
        # Fetch live 1d data for top holdings to show contribution
        tickers = [h[0] for h in QQQ_TOP_HOLDINGS]
        df = yf.download(tickers, period="30d", interval="1d",
                         auto_adjust=True, progress=False, group_by="ticker", threads=True)

        holdings = []
        for ticker, name, weight in QQQ_TOP_HOLDINGS:
            price, chg_1d, chg_5d, above_sma20 = None, None, None, None

            try:
                if ticker in df.columns or (ticker, "Close") in df.columns:
                    try:
                        sub = df[ticker].dropna()
                    except Exception:
                        sub = df.xs(ticker, axis=1, level=0).dropna()

                    if len(sub) >= 6:
                        closes   = sub["Close"].dropna()
                        price    = round(float(closes.iloc[-1]), 2)
                        chg_1d   = round(float(closes.pct_change(1).iloc[-1]) * 100, 2)
                        chg_5d   = round(float(closes.pct_change(5).iloc[-1]) * 100, 2) \
                                   if len(closes) >= 6 else None
                        sma20    = float(closes.rolling(min(20, len(closes))).mean().iloc[-1])
                        above_sma20 = price > sma20
            except Exception:
                pass

            # Contribution signal based on weight and performance
            if chg_1d is not None:
                weighted_contrib = (weight / 100) * chg_1d if chg_1d else 0
                if weighted_contrib > 0.05:
                    contrib = "LIFTING"
                elif weighted_contrib < -0.05:
                    contrib = "DRAGGING"
                else:
                    contrib = "NEUTRAL"
            else:
                contrib = "NEUTRAL"

            holdings.append(HoldingData(
                ticker=ticker, name=name, weight=weight,
                price=price, change_pct_1d=chg_1d, change_pct_5d=chg_5d,
                above_sma20=above_sma20, contribution_signal=contrib,
            ))

        _store(key, holdings)
        return holdings
    except Exception as e:
        print(f"[qqq] Holdings error: {e}")
        # Return static holdings with no live data
        return [
            HoldingData(ticker=t, name=n, weight=w,
                        price=None, change_pct_1d=None, change_pct_5d=None,
                        above_sma20=None, contribution_signal="NEUTRAL")
            for t, n, w in QQQ_TOP_HOLDINGS
        ]


# ── SECTION 5: SECTOR WEIGHTING (static + visual) ─────────────────────────────
# No fetch needed — uses QQQ_SECTORS constant


# ── SECTION 6: HISTORICAL PERFORMANCE ────────────────────────────────────────

def _fetch_performance() -> Optional[PerformanceData]:
    key = "qqq_performance"
    if _cache_valid(key, TTL_FUND):
        return _load(key)

    try:
        q    = yf.Ticker(QQQ_TICKER)
        spy  = yf.Ticker("SPY")
        df   = q.history(period="5y", interval="1d")
        df_s = spy.history(period="1y", interval="1d")

        if df is None or df.empty or len(df) < 5:
            return _load(key)

        df.columns = [c.capitalize() for c in df.columns]
        c = df['Close']

        def pct(n):
            return round(float(c.pct_change(n).iloc[-1]) * 100, 2) if len(c) > n else 0.0

        # YTD
        df.index = pd.to_datetime(df.index, utc=True)
        year_start = c[c.index.year == pd.Timestamp.now().year]
        ytd = round((float(c.iloc[-1]) - float(year_start.iloc[0])) /
                    float(year_start.iloc[0]) * 100, 2) if len(year_start) > 0 else 0.0

        # Multi-period returns
        ret_1y  = pct(252)
        ret_3y  = pct(756)
        ret_5y  = pct(1260)

        # 52-week range
        high_52w = float(c.tail(252).max())
        low_52w  = float(c.tail(252).min())
        curr_p   = float(c.iloc[-1])
        pct_from_high = (curr_p - high_52w) / high_52w * 100
        pct_from_low  = (curr_p - low_52w)  / low_52w  * 100

        # 30-day realized volatility (annualized)
        log_ret  = np.log(c / c.shift(1)).dropna()
        vol_30d  = float(log_ret.tail(30).std() * np.sqrt(252) * 100)

        # Max drawdown 1y
        c1y      = c.tail(252)
        peak     = c1y.cummax()
        dd       = (c1y - peak) / peak
        max_dd   = float(dd.min() * 100)

        # Sharpe proxy (1y return / 30d vol)
        sharpe   = round(ret_1y / vol_30d, 2) if vol_30d > 0 else 0

        # vs SPY 1y
        vs_spy = 0.0
        if df_s is not None and len(df_s) >= 5:
            df_s.columns = [c2.capitalize() for c2 in df_s.columns]
            spy_ret = round(float(df_s['Close'].pct_change(252).iloc[-1]) * 100, 2) \
                      if len(df_s) >= 252 else 0
            vs_spy  = round(ret_1y - spy_ret, 2)

        result = PerformanceData(
            ret_1d=pct(1), ret_5d=pct(5), ret_1m=pct(21), ret_3m=pct(63),
            ret_ytd=ytd, ret_1y=ret_1y, ret_3y=ret_3y, ret_5y=ret_5y,
            high_52w=round(high_52w, 2), low_52w=round(low_52w, 2),
            pct_from_52w_high=round(pct_from_high, 2),
            pct_from_52w_low=round(pct_from_low, 2),
            volatility_30d=round(vol_30d, 2),
            max_drawdown_1y=round(max_dd, 2),
            sharpe_1y=sharpe,
            vs_spy_1y=vs_spy,
        )
        _store(key, result)
        return result
    except Exception as e:
        print(f"[qqq] Performance error: {e}")
        return _load(key)


# ── SECTION 7: OPTIONS CHAIN ──────────────────────────────────────────────────

def _fetch_options(expiry_idx: int = 0) -> Optional[OptionsData]:
    key = f"qqq_options_{expiry_idx}"
    if _cache_valid(key, TTL_OPTIONS):
        return _load(key)

    try:
        q    = yf.Ticker(QQQ_TICKER)
        exps = q.options
        if not exps:
            return _load(key)

        avail = list(exps)
        exp   = avail[min(expiry_idx, len(avail) - 1)]
        chain = q.option_chain(exp)
        calls = chain.calls.copy()
        puts  = chain.puts.copy()

        # Current price
        fi    = q.fast_info
        try:
            curr_p = float(fi.last_price or fi.previous_close or 0)
        except Exception:
            curr_p = float(calls['strike'].median())

        if curr_p <= 0:
            curr_p = float(calls['strike'].median())

        # Filter to ±8% of current price
        lo = curr_p * 0.92
        hi = curr_p * 1.08
        calls_f = calls[(calls['strike'] >= lo) & (calls['strike'] <= hi)].copy()
        puts_f  = puts[ (puts['strike']  >= lo) & (puts['strike']  <= hi)].copy()

        # ATM strike
        calls_f['dist'] = abs(calls_f['strike'] - curr_p)
        atm_idx  = calls_f['dist'].idxmin() if not calls_f.empty else None
        atm_call = calls_f.loc[atm_idx] if atm_idx is not None else None
        atm_strike = float(atm_call['strike']) if atm_call is not None else curr_p

        atm_put  = puts_f[abs(puts_f['strike'] - atm_strike) < 1].iloc[0] \
                   if not puts_f[abs(puts_f['strike'] - atm_strike) < 1].empty else None

        atm_call_iv = round(float(atm_call['impliedVolatility']) * 100, 1) \
                      if atm_call is not None else 0
        atm_put_iv  = round(float(atm_put['impliedVolatility']) * 100, 1) \
                      if atm_put is not None else 0
        iv_skew     = round(atm_put_iv - atm_call_iv, 2)

        # PCR
        total_call_oi  = int(calls['openInterest'].fillna(0).sum())
        total_put_oi   = int(puts['openInterest'].fillna(0).sum())
        total_call_vol = int(calls['volume'].fillna(0).sum())
        total_put_vol  = int(puts['volume'].fillna(0).sum())
        pcr_oi  = round(total_put_oi / total_call_oi, 2) if total_call_oi > 0 else 1.0
        pcr_vol = round(total_put_vol / total_call_vol, 2) if total_call_vol > 0 else 1.0

        if pcr_oi > 1.3:
            pcr_sig   = "🐻 BEARISH FEAR — Heavy put buying"
            pcr_color = "#c9302c"
        elif pcr_oi < 0.7:
            pcr_sig   = "🐂 BULLISH COMPLACENCY — Call heavy"
            pcr_color = "#e6a817"
        else:
            pcr_sig   = "⚖️ NEUTRAL"
            pcr_color = "#2d9e2d"

        # Max pain (sum of all OI loss at each strike)
        all_strikes = sorted(set(list(calls['strike']) + list(puts['strike'])))
        min_pain    = float('inf')
        max_pain    = float(atm_strike)
        for s in all_strikes:
            c_loss = calls[calls['strike'] <= s]['openInterest'].fillna(0).sum() * 0
            p_loss = puts[puts['strike'] >= s]['openInterest'].fillna(0).sum()
            c_loss2= calls[calls['strike'] > s].apply(
                lambda r: r['openInterest'] * max(0, s - r['strike']), axis=1
            ).sum()
            p_loss2 = puts[puts['strike'] < s].apply(
                lambda r: r['openInterest'] * max(0, r['strike'] - s), axis=1
            ).sum()
            total_pain = c_loss2 + p_loss2
            if total_pain < min_pain:
                min_pain = total_pain
                max_pain = s

        # Gamma walls
        gamma_call = float(calls.loc[calls['openInterest'].idxmax(), 'strike']) \
                     if not calls.empty else atm_strike
        gamma_put  = float(puts.loc[puts['openInterest'].idxmax(), 'strike']) \
                     if not puts.empty else atm_strike

        # Expected move from straddle
        straddle_price = (float(atm_call.get('lastPrice', 0)) +
                          float(atm_put['lastPrice']) if atm_put is not None else 0) \
                         if atm_call is not None else 0
        em_pts = round(straddle_price, 2)
        em_pct = round(em_pts / curr_p * 100, 2) if curr_p > 0 else 0

        # Build strike list
        merged = pd.merge(
            calls_f[['strike','lastPrice','bid','ask','volume','openInterest','impliedVolatility']],
            puts_f[['strike', 'lastPrice','bid','ask','volume','openInterest','impliedVolatility']],
            on='strike', suffixes=('_c','_p'), how='outer'
        ).fillna(0).sort_values('strike', ascending=False)

        strikes = []
        for _, row in merged.iterrows():
            s = float(row['strike'])
            strikes.append(OptionStrike(
                strike=s,
                call_last=round(float(row.get('lastPrice_c', 0)), 2),
                call_bid=round(float(row.get('bid_c', 0)), 2),
                call_ask=round(float(row.get('ask_c', 0)), 2),
                call_oi=int(row.get('openInterest_c', 0)),
                call_vol=int(row.get('volume_c', 0)),
                call_iv=round(float(row.get('impliedVolatility_c', 0)) * 100, 1),
                put_last=round(float(row.get('lastPrice_p', 0)), 2),
                put_bid=round(float(row.get('bid_p', 0)), 2),
                put_ask=round(float(row.get('ask_p', 0)), 2),
                put_oi=int(row.get('openInterest_p', 0)),
                put_vol=int(row.get('volume_p', 0)),
                put_iv=round(float(row.get('impliedVolatility_p', 0)) * 100, 1),
                is_atm=abs(s - atm_strike) < 1,
                distance_pct=round((s - curr_p) / curr_p * 100, 2),
            ))

        # Unusual options activity (volume >> open interest = fresh positioning)
        def _unusual(df_opt, side):
            out = []
            for _, r in df_opt.iterrows():
                vol = float(r.get('volume', 0) or 0)
                oi  = float(r.get('openInterest', 0) or 0)
                if vol >= 500 and (oi == 0 or vol / max(oi, 1) > 3.0):
                    out.append({
                        'strike': float(r['strike']),
                        'volume': int(vol), 'oi': int(oi),
                        'iv': round(float(r.get('impliedVolatility', 0)) * 100, 1),
                        'last': round(float(r.get('lastPrice', 0)), 2),
                        'type': side,
                    })
            return sorted(out, key=lambda x: x['volume'], reverse=True)[:5]

        result = OptionsData(
            expiry=exp,
            available_expiries=avail[:8],
            atm_strike=atm_strike,
            atm_call_iv=atm_call_iv,
            atm_put_iv=atm_put_iv,
            iv_skew=iv_skew,
            put_call_ratio_oi=pcr_oi,
            put_call_ratio_vol=pcr_vol,
            max_pain=round(max_pain, 0),
            gamma_wall_call=round(gamma_call, 0),
            gamma_wall_put=round(gamma_put, 0),
            strikes=strikes,
            unusual_calls=_unusual(calls, 'CALL'),
            unusual_puts=_unusual(puts, 'PUT'),
            pcr_signal=pcr_sig,
            pcr_signal_color=pcr_color,
            expected_move_1sd=em_pts,
            expected_move_pct=em_pct,
        )
        _store(key, result)
        return result
    except Exception as e:
        print(f"[qqq] Options error: {e}")
        return _load(key)


# ── MASTER FETCH ──────────────────────────────────────────────────────────────

def get_qqq_report(selected_expiry_idx: int = 0) -> QQQReport:
    """Fetch all sections (each independently cached) and return full report."""
    errors = []

    def _safe(fn, *args, label=""):
        try:
            return fn(*args)
        except Exception as e:
            errors.append(f"{label}: {e}")
            return None

    intraday    = _safe(_fetch_intraday,        label="Intraday")
    volume      = _safe(_fetch_volume_analysis, label="Volume")
    fund        = _safe(_fetch_fund_info,        label="Fund")
    holdings    = _safe(_fetch_holdings,         label="Holdings") or []
    performance = _safe(_fetch_performance,      label="Performance")
    options     = _safe(_fetch_options, selected_expiry_idx, label="Options")

    return QQQReport(
        intraday=intraday, volume=volume, fund=fund,
        holdings=holdings, performance=performance,
        options=options,
        fetched_at=pd.Timestamp.now().strftime("%H:%M:%S"),
        errors=errors,
    )


# ── RENDER ────────────────────────────────────────────────────────────────────

def render_qqq_intelligence(report: QQQReport, selected_expiry: int = 0):
    """Full QQQ intelligence panel — 7 tabs."""
    import streamlit as st

    st.subheader("📦 QQQ ETF Intelligence")
    st.caption(f"Data as of {report.fetched_at} | Invesco QQQ Trust (NASDAQ: QQQ)")

    if report.errors:
        with st.expander(f"⚠️ {len(report.errors)} fetch warning(s)"):
            for e in report.errors:
                st.caption(e)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Price & Volume",
        "📊 Volume Analysis",
        "🏦 Fund Info",
        "🏆 Holdings",
        "🥧 Sectors",
        "📉 Performance",
        "⚙️ Options Chain",
    ])

    # ── TAB 1: PRICE & VOLUME ─────────────────────────────────────────────────
    with tab1:
        iv = report.intraday
        if iv is None:
            st.warning("Intraday data unavailable.")
        else:
            chg_col = "#2d9e2d" if iv.change_pct >= 0 else "#c9302c"
            chg_icon = "▲" if iv.change_pct >= 0 else "▼"

            st.markdown(
                f"<div style='display:flex;align-items:baseline;gap:16px;margin-bottom:12px'>"
                f"<span style='font-size:2.5em;font-weight:900'>${iv.current_price:,.2f}</span>"
                f"<span style='color:{chg_col};font-size:1.3em;font-weight:bold'>"
                f"{chg_icon} {iv.change_pts:+.2f} ({iv.change_pct:+.2f}%)</span>"
                f"<span style='color:#888;font-size:0.9em'>vs prev close ${iv.prev_close:.2f}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Open",      f"${iv.day_open:.2f}")
            c2.metric("Day High",  f"${iv.day_high:.2f}")
            c3.metric("Day Low",   f"${iv.day_low:.2f}")
            c4.metric("Range",     f"{iv.intraday_range_pct:.2f}%")
            c5.metric("VWAP",      f"${iv.vwap:.2f}",
                      delta=f"{iv.vwap_dev_pct:+.2f}% {'above' if iv.price_vs_vwap == 'ABOVE' else 'below'}",
                      delta_color="normal" if iv.price_vs_vwap == "ABOVE" else "inverse")

            st.markdown("---")
            v1, v2, v3, v4, v5 = st.columns(5)
            v1.metric("Volume Today",    f"{iv.volume_today/1e6:.1f}M")
            v2.metric("20d Avg Full Day",f"{iv.avg_volume_20d/1e6:.1f}M")

            # Session-adjusted pace — the key fix
            pace_col = ("normal" if iv.vol_pace_ratio >= 0.85 else "inverse")
            v3.metric("Pace Ratio",
                      f"{iv.vol_pace_ratio:.2f}×",
                      delta=f"{'On pace' if iv.vol_pace_ratio >= 0.85 else 'Below pace'} "
                            f"({iv.session_frac*100:.0f}% thru session)",
                      delta_color=pace_col)

            # Acceleration
            accel_delta = f"{'↑ Accelerating' if iv.vol_accel >= 1.3 else ('↓ Decelerating' if iv.vol_accel < 0.8 else 'Steady')}"
            v4.metric("Vol Acceleration", f"{iv.vol_accel:.2f}×", delta=accel_delta)
            v5.metric("Dollar Volume",   f"${iv.dollar_volume_today:.1f}B")

            # Volume signal banner
            st.markdown(
                f"<div style='padding:8px 12px;border-radius:6px;"
                f"background:{iv.vol_signal_color}22;"
                f"border-left:3px solid {iv.vol_signal_color};margin-top:8px'>"
                f"<span style='color:{iv.vol_signal_color};font-weight:bold'>"
                f"{iv.vol_signal}</span>"
                f"<span style='color:#888;font-size:0.82em;margin-left:12px'>"
                f"Raw ratio: {iv.vol_ratio_raw:.2f}× 20d avg</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # 5m intraday chart using st.line_chart
            if iv.bars_5m is not None and not iv.bars_5m.empty:
                st.markdown("**Intraday 5m Price**")
                chart_df = iv.bars_5m[['Close']].copy()
                chart_df.columns = ["QQQ Price"]
                chart_df.index = chart_df.index.strftime("%H:%M")
                st.line_chart(chart_df, height=200)

    # ── TAB 2: VOLUME ANALYSIS ────────────────────────────────────────────────
    with tab2:
        va = report.volume
        if va is None:
            st.warning("Volume data unavailable.")
        else:
            st.markdown(
                f"<div style='padding:10px;border-radius:8px;"
                f"background:{va.signal_color}22;border:2px solid {va.signal_color};"
                f"margin-bottom:12px'>"
                f"<span style='color:{va.signal_color};font-size:1.1em;font-weight:bold'>"
                f"{va.signal}</span></div>",
                unsafe_allow_html=True,
            )

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Today Volume",     f"{va.today_vol/1e6:.1f}M")
            a2.metric("20d Avg",          f"{va.avg_20d/1e6:.1f}M")
            a3.metric("Ratio vs 20d",     f"{va.vol_ratio_vs_20d:.2f}x",
                      delta="Unusual" if va.is_unusual else "Normal",
                      delta_color="off")
            a4.metric("Volume Percentile",f"{va.vol_percentile:.0f}th",
                      delta=f"5d trend: {va.trend_5d}")

            b1, b2, b3 = st.columns(3)
            b1.metric("Unusual Threshold",  f"{va.unusual_threshold/1e6:.1f}M (1.5× avg)")
            b2.metric("Avg Dollar Volume",  f"${va.dollar_vol_avg:.0f}M/day")
            b3.metric("Biggest Vol Day",    va.biggest_vol_day)

            # Volume bar chart (20 days)
            st.markdown("**20-Day Volume History**")
            vol_df = pd.DataFrame({
                "Volume (M)": [v / 1e6 for v in va.volumes_20d]
            })
            colors_list = [
                "#c9302c" if v / 1e6 >= va.unusual_threshold / 1e6 else "#4a7fb5"
                for v in va.volumes_20d
            ]
            st.bar_chart(vol_df, height=200)
            st.caption("🔴 Red bars = unusual volume (≥1.5× 20d avg)")

    # ── TAB 3: FUND INFO ──────────────────────────────────────────────────────
    with tab3:
        fi = report.fund
        if fi is None:
            st.warning("Fund info unavailable.")
            # Show static known facts
            st.markdown("""
            **QQQ — Invesco QQQ Trust**
            - Tracks: NASDAQ-100 Index
            - Expense Ratio: 0.20%
            - Inception: March 10, 1999
            - Exchange: NASDAQ
            - Fund Family: Invesco
            """)
        else:
            f1, f2, f3 = st.columns(3)
            with f1:
                st.markdown("**Fund Basics**")
                st.metric("AUM",            f"${fi.total_assets_bn:.1f}B" if fi.total_assets_bn else "~$300B")
                st.metric("NAV Price",      f"${fi.nav_price:.2f}")
                st.metric("Expense Ratio",  f"{fi.expense_ratio:.3f}%")
                st.metric("Dividend Yield", f"{fi.dividend_yield:.3f}%")
                st.caption(f"Family: {fi.fund_family} | Category: {fi.category}")
                st.caption(f"Inception: {fi.inception_date}")

            with f2:
                st.markdown("**Returns**")
                for label, val in [
                    ("YTD",    fi.ytd_return),
                    ("1 Year", fi.one_year_return),
                    ("3 Year", fi.three_year_return),
                    ("5 Year", fi.five_year_return),
                ]:
                    color = "#2d9e2d" if val >= 0 else "#c9302c"
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between'>"
                        f"<span style='color:#aaa'>{label}</span>"
                        f"<span style='color:{color};font-weight:bold'>{val:+.2f}%</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            with f3:
                st.markdown("**Risk Metrics**")
                st.metric("Beta (3Y)",    f"{fi.beta:.2f}")
                st.metric("P/E Ratio",   f"{fi.pe_ratio:.1f}" if fi.pe_ratio else "N/A")
                pd_color = "#2d9e2d" if abs(fi.premium_discount) < 0.1 else "#e6a817"
                st.markdown(
                    f"**Premium/Discount to NAV:** "
                    f"<span style='color:{pd_color}'>{fi.premium_discount:+.3f}%</span>",
                    unsafe_allow_html=True,
                )
                if fi.shares_outstanding > 0:
                    st.metric("Shares Out", f"{fi.shares_outstanding/1e6:.0f}M")

    # ── TAB 4: TOP HOLDINGS ───────────────────────────────────────────────────
    with tab4:
        holdings = report.holdings
        if not holdings:
            st.warning("Holdings data unavailable.")
        else:
            st.caption("Top 10 holdings by weight | Live prices where available")

            # Weight bar visual + table
            for h in holdings:
                c1, c2, c3, c4, c5 = st.columns([1, 3, 2, 2, 2])
                sig_colors = {"LIFTING": "#2d9e2d", "DRAGGING": "#c9302c", "NEUTRAL": "#888"}
                sig_icons  = {"LIFTING": "▲", "DRAGGING": "▼", "NEUTRAL": "—"}
                sc = sig_colors.get(h.contribution_signal, "#888")
                si = sig_icons.get(h.contribution_signal, "—")

                with c1:
                    st.markdown(
                        f"<span style='color:{sc};font-weight:bold'>{si}</span>",
                        unsafe_allow_html=True,
                    )
                with c2:
                    bar_w = int(h.weight / 21 * 100)  # scale to max holding
                    st.markdown(
                        f"<div><span style='font-weight:bold'>{h.ticker}</span> "
                        f"<span style='color:#888;font-size:0.85em'>{h.name}</span></div>"
                        f"<div style='background:#333;border-radius:3px;height:5px;margin-top:2px'>"
                        f"<div style='background:#4a7fb5;width:{bar_w}%;height:5px;"
                        f"border-radius:3px'></div></div>",
                        unsafe_allow_html=True,
                    )
                with c3:
                    st.markdown(
                        f"<span style='color:#e6a817;font-weight:bold'>{h.weight:.2f}%</span>",
                        unsafe_allow_html=True,
                    )
                with c4:
                    if h.price:
                        st.markdown(f"${h.price:.2f}", unsafe_allow_html=True)
                    else:
                        st.markdown("—")
                with c5:
                    if h.change_pct_1d is not None:
                        col = "#2d9e2d" if h.change_pct_1d >= 0 else "#c9302c"
                        sma_badge = (
                            " 🟢" if h.above_sma20 else " 🔴"
                        ) if h.above_sma20 is not None else ""
                        st.markdown(
                            f"<span style='color:{col}'>{h.change_pct_1d:+.2f}%</span>"
                            f"<span style='font-size:0.8em'>{sma_badge}</span>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown("—")

            st.caption("🟢/🔴 = above/below 20-day SMA | ▲/▼ = lifting/dragging index today")

    # ── TAB 5: SECTOR WEIGHTING ───────────────────────────────────────────────
    with tab5:
        st.markdown("**QQQ Sector Allocation (Invesco published)**")
        st.caption("Source: Invesco fund facts — updated periodically")

        total = sum(QQQ_SECTORS.values())
        for sector, weight in sorted(QQQ_SECTORS.items(), key=lambda x: x[1], reverse=True):
            color   = SECTOR_COLORS.get(sector, "#888")
            bar_w   = int(weight / total * 100)
            st.markdown(
                f"<div style='margin:5px 0'>"
                f"<div style='display:flex;justify-content:space-between;margin-bottom:2px'>"
                f"<span style='color:#ddd'>{sector}</span>"
                f"<span style='color:{color};font-weight:bold'>{weight:.1f}%</span>"
                f"</div>"
                f"<div style='background:#333;border-radius:4px;height:8px'>"
                f"<div style='background:{color};width:{int(weight/max(QQQ_SECTORS.values())*100)}%;"
                f"height:8px;border-radius:4px'></div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption(
            "**Note:** QQQ is market-cap weighted. Technology alone accounts for >50% "
            "of the index — this concentration means NVDA, AAPL, MSFT moves have "
            "outsized impact on QQQ price."
        )

    # ── TAB 6: HISTORICAL PERFORMANCE ─────────────────────────────────────────
    with tab6:
        perf = report.performance
        if perf is None:
            st.warning("Performance data unavailable.")
        else:
            st.markdown("**Return Summary**")
            periods = [
                ("1 Day",   perf.ret_1d),
                ("5 Day",   perf.ret_5d),
                ("1 Month", perf.ret_1m),
                ("3 Month", perf.ret_3m),
                ("YTD",     perf.ret_ytd),
                ("1 Year",  perf.ret_1y),
                ("3 Year",  perf.ret_3y),
                ("5 Year",  perf.ret_5y),
            ]
            p_cols = st.columns(4)
            for i, (label, val) in enumerate(periods):
                col = "#2d9e2d" if val >= 0 else "#c9302c"
                p_cols[i % 4].markdown(
                    f"<div style='padding:8px;border-radius:6px;"
                    f"background:{col}22;text-align:center;margin:3px'>"
                    f"<div style='color:#aaa;font-size:0.8em'>{label}</div>"
                    f"<div style='color:{col};font-size:1.2em;font-weight:bold'>"
                    f"{val:+.2f}%</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("52W High", f"${perf.high_52w:.2f}",
                      delta=f"{perf.pct_from_52w_high:.1f}% from high")
            r2.metric("52W Low",  f"${perf.low_52w:.2f}",
                      delta=f"+{perf.pct_from_52w_low:.1f}% from low",
                      delta_color="normal")
            r3.metric("30d Volatility (Ann.)", f"{perf.volatility_30d:.1f}%")
            r4.metric("Max Drawdown 1Y", f"{perf.max_drawdown_1y:.1f}%",
                      delta_color="inverse")

            s1, s2 = st.columns(2)
            s1.metric("Sharpe Ratio (1Y proxy)", f"{perf.sharpe_1y:.2f}")
            vs_col = "#2d9e2d" if perf.vs_spy_1y >= 0 else "#c9302c"
            s2.markdown(
                f"**vs SPY (1Y):** "
                f"<span style='color:{vs_col};font-weight:bold'>"
                f"{perf.vs_spy_1y:+.2f}%</span>",
                unsafe_allow_html=True,
            )

    # ── TAB 7: OPTIONS CHAIN ──────────────────────────────────────────────────
    with tab7:
        opt = report.options
        if opt is None:
            st.warning("Options data unavailable.")
        else:
            # Expiry selector
            if opt.available_expiries:
                sel_exp = st.selectbox(
                    "Expiry Date",
                    options=opt.available_expiries,
                    index=min(selected_expiry, len(opt.available_expiries) - 1),
                    key="qqq_expiry_select",
                )
                if sel_exp != opt.expiry:
                    new_idx = opt.available_expiries.index(sel_exp)
                    new_opt = _fetch_options(new_idx)
                    if new_opt:
                        opt = new_opt

            # Summary metrics
            st.markdown(
                f"<div style='padding:8px 12px;border-radius:6px;"
                f"background:{opt.pcr_signal_color}22;"
                f"border-left:3px solid {opt.pcr_signal_color};margin-bottom:10px'>"
                f"<span style='color:{opt.pcr_signal_color};font-weight:bold'>"
                f"{opt.pcr_signal}</span>"
                f"<span style='color:#aaa;margin-left:16px;font-size:0.9em'>"
                f"PCR (OI): {opt.put_call_ratio_oi:.2f} | "
                f"PCR (Vol): {opt.put_call_ratio_vol:.2f}"
                f"</span></div>",
                unsafe_allow_html=True,
            )

            o1, o2, o3, o4, o5 = st.columns(5)
            o1.metric("ATM Strike",     f"${opt.atm_strike:.0f}")
            o2.metric("ATM Call IV",    f"{opt.atm_call_iv:.1f}%")
            o3.metric("ATM Put IV",     f"{opt.atm_put_iv:.1f}%")
            iv_sk_col = "#c9302c" if opt.iv_skew > 3 else (
                        "#e6a817" if opt.iv_skew > 1 else "#2d9e2d")
            o4.markdown(
                f"<div style='padding:8px;border-radius:6px;background:{iv_sk_col}22'>"
                f"<div style='color:#aaa;font-size:0.75em'>IV Skew (put-call)</div>"
                f"<div style='color:{iv_sk_col};font-weight:bold;font-size:1.1em'>"
                f"{opt.iv_skew:+.1f}%</div></div>",
                unsafe_allow_html=True,
            )
            o5.metric("Expected Move", f"±${opt.expected_move_1sd:.2f} ({opt.expected_move_pct:.1f}%)")

            st.markdown(
                f"**Key Levels:** Max Pain: `{opt.max_pain:,.0f}` | "
                f"Call Wall (γ): `{opt.gamma_wall_call:,.0f}` | "
                f"Put Wall (γ): `{opt.gamma_wall_put:,.0f}`"
            )

            # Options chain table
            st.markdown("**Options Chain (±8% of current price)**")
            if opt.strikes:
                rows = []
                for s in opt.strikes:
                    atm_marker = " ◄ ATM" if s.is_atm else ""
                    rows.append({
                        "Strike":     f"{s.strike:.0f}{atm_marker}",
                        "Dist%":      f"{s.distance_pct:+.1f}%",
                        "Call Last":  f"${s.call_last:.2f}" if s.call_last else "—",
                        "Call OI":    f"{s.call_oi:,}"  if s.call_oi  else "—",
                        "Call Vol":   f"{s.call_vol:,}" if s.call_vol else "—",
                        "Call IV%":   f"{s.call_iv:.1f}%" if s.call_iv else "—",
                        "Put Last":   f"${s.put_last:.2f}"  if s.put_last  else "—",
                        "Put OI":     f"{s.put_oi:,}"   if s.put_oi   else "—",
                        "Put Vol":    f"{s.put_vol:,}"  if s.put_vol  else "—",
                        "Put IV%":    f"{s.put_iv:.1f}%" if s.put_iv  else "—",
                    })

                df_chain = pd.DataFrame(rows)

                def _highlight_atm(row):
                    if "ATM" in str(row.get("Strike", "")):
                        return ["background-color:#2a2a4a"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    df_chain.style.apply(_highlight_atm, axis=1),
                    use_container_width=True,
                    height=400,
                    hide_index=True,
                )

            # Unusual activity
            ua_c, ua_p = st.columns(2)
            with ua_c:
                st.markdown("**⚡ Unusual Call Activity**")
                if opt.unusual_calls:
                    for u in opt.unusual_calls:
                        st.markdown(
                            f"Strike `{u['strike']:.0f}` — "
                            f"Vol: **{u['volume']:,}** OI: {u['oi']:,} "
                            f"IV: {u['iv']:.1f}%",
                        )
                else:
                    st.caption("No unusual call activity detected")

            with ua_p:
                st.markdown("**⚡ Unusual Put Activity**")
                if opt.unusual_puts:
                    for u in opt.unusual_puts:
                        st.markdown(
                            f"Strike `{u['strike']:.0f}` — "
                            f"Vol: **{u['volume']:,}** OI: {u['oi']:,} "
                            f"IV: {u['iv']:.1f}%",
                        )
                else:
                    st.caption("No unusual put activity detected")
