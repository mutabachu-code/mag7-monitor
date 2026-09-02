"""
data_fetcher.py  —  v4
-----------------------
Single data layer for the entire Mag7 + NAS100 dashboard.
Fetches ALL instruments in one parallel batch with hard timeouts.

Instruments covered:
  - Mag 7 stocks:     AAPL MSFT GOOGL AMZN TSLA META NVDA
  - NAS100 proxy:     QQQ  (^NDX for scaling ratio)
  - Volatility:       ^VIX
  - Macro:            ^TNX (10Y yield), BZ=F (Brent oil), QQQE (equal-weight Nasdaq)

Before v4: 34 individual yfinance calls per refresh
After  v4: 1 parallel batch, all results in session_state cache (65s TTL)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import streamlit as st
import threading
from typing import Optional, Tuple

# ── INSTRUMENT REGISTRY ───────────────────────────────────────────────────────
MAG7         = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
NAS100_LABEL = 'NAS100'
NAS100_YF    = 'QQQ'        # reliable proxy; ^NDX often blocked on Linux
VIX_YF       = '^VIX'
NDX_YF       = '^NDX'       # only used for QQQ→NAS100 scaling ratio
TNX_YF       = '^TNX'       # US 10Y Treasury yield
TNX_FALLBACK = 'IEF'        # 7-10Y Treasury ETF as fallback
OIL_YF       = 'BZ=F'       # Brent crude futures
OIL_FALLBACK = 'USO'        # Oil ETF as fallback (more reliable)
QQQE_YF      = 'QQQE'       # Equal-weight Nasdaq-100 (breadth indicator)

ALL_LABELS   = [NAS100_LABEL] + MAG7   # price card tickers

# Macro instruments fetched separately (daily data only)
GOLD_YF      = 'GLD'       # Gold ETF — reliable proxy for XAU/USD
MACRO_YF     = [TNX_YF, OIL_YF, QQQE_YF, NDX_YF, VIX_YF, GOLD_YF]

CACHE_TTL    = 65    # seconds — slightly longer than 60s refresh
FETCH_TIMEOUT = 10   # seconds per ticker before abandoning


# ── TIMEOUT WRAPPER ───────────────────────────────────────────────────────────

def _fetch_with_timeout(func, timeout=FETCH_TIMEOUT):
    result = [None]
    def target():
        try:
            result[0] = func()
        except Exception as e:
            print(f"[data_fetcher] {e}")
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result[0]


# ── CACHE HELPERS ─────────────────────────────────────────────────────────────

def _cache_valid() -> bool:
    return (time.time() - st.session_state.get("data_fetch_ts", 0)) < CACHE_TTL

def _store(key, data):
    st.session_state[key] = data

def _load(key):
    return st.session_state.get(key)


# ── PRICE TICKER FETCH (5m + 1h + 1d) ────────────────────────────────────────

def _fetch_price_ticker(label: str) -> Tuple[Optional[pd.DataFrame],
                                              Optional[pd.DataFrame],
                                              Optional[pd.DataFrame]]:
    yfticker = NAS100_YF if label == NAS100_LABEL else label

    def get_5m():
        df = yf.Ticker(yfticker).history(period="5d", interval="5m", prepost=True).ffill().bfill()
        return df if not df.empty else None

    def get_1h():
        df = yf.Ticker(yfticker).history(period="60d", interval="1h").ffill().bfill()
        return df if not df.empty else None

    def get_1d():
        df = yf.Ticker(yfticker).history(period="365d", interval="1d").ffill().bfill()
        return df if not df.empty else None

    return (
        _fetch_with_timeout(get_5m, FETCH_TIMEOUT),
        _fetch_with_timeout(get_1h, FETCH_TIMEOUT),
        _fetch_with_timeout(get_1d, FETCH_TIMEOUT),
    )


# ── MACRO INSTRUMENT FETCH (daily only) ───────────────────────────────────────

def _fetch_macro_instrument(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetch daily data for a macro instrument with fallback.
    BZ=F (Brent) often returns stale contract data — falls back to USO.
    ^TNX sometimes returns empty — falls back to IEF yield proxy.
    """
    fallbacks = {OIL_YF: OIL_FALLBACK, TNX_YF: TNX_FALLBACK}

    def get():
        period = "30d" if symbol in [TNX_YF, QQQE_YF, NDX_YF] else "5d"
        df = yf.Ticker(symbol).history(period=period, interval="1d").ffill().bfill()
        if not df.empty:
            # Validate data is recent (within 3 trading days)
            last_date = pd.Timestamp(df.index[-1]).date()
            today     = pd.Timestamp.now().date()
            days_old  = (today - last_date).days
            if days_old <= 4:   # allow for weekends
                return df
            print(f"[data_fetcher] {symbol} data is {days_old} days old — trying fallback")

        # Try fallback symbol if available
        fallback = fallbacks.get(symbol)
        if fallback:
            print(f"[data_fetcher] Falling back {symbol} → {fallback}")
            df2 = yf.Ticker(fallback).history(period="5d", interval="1d").ffill().bfill()
            if not df2.empty:
                return df2
        return None
    return _fetch_with_timeout(get, FETCH_TIMEOUT)


# ── MASTER FETCH ─────────────────────────────────────────────────────────────

def fetch_all_data() -> bool:
    """
    Fetch ALL instruments in a single parallel batch.
    Stores everything in session_state. Returns True if at least some data loaded.

    Call once per 60s refresh cycle — subsequent reads use cache.
    """
    if _cache_valid():
        return True

    print(f"[data_fetcher] Parallel fetch: {len(ALL_LABELS)} price tickers + {len(MACRO_YF)} macro instruments")
    start = time.time()

    price_results = {}
    macro_results = {}
    threads = []

    # Price tickers (5m + 1h + 1d)
    def fetch_price(label):
        price_results[label] = _fetch_price_ticker(label)

    for label in ALL_LABELS:
        t = threading.Thread(target=fetch_price, args=(label,), daemon=True)
        threads.append(t)
        t.start()

    # Macro instruments (1d only)
    def fetch_macro(sym):
        macro_results[sym] = _fetch_macro_instrument(sym)

    for sym in MACRO_YF:
        t = threading.Thread(target=fetch_macro, args=(sym,), daemon=True)
        threads.append(t)
        t.start()

    # Wait for all threads
    for t in threads:
        t.join(timeout=FETCH_TIMEOUT + 2)

    # Store price data
    any_success = False
    for label, (df_5m, df_1h, df_1d) in price_results.items():
        if df_5m is not None or df_1h is not None:
            any_success = True
        _store(f"df_5m_{label}", df_5m)
        _store(f"df_1h_{label}", df_1h)
        _store(f"df_1d_{label}", df_1d)

    # Store macro data
    for sym, df in macro_results.items():
        key = {
            TNX_YF:  "macro_tnx",
            OIL_YF:  "macro_oil",
            QQQE_YF: "macro_qqqe",
            NDX_YF:  "macro_ndx",
            VIX_YF:  "macro_vix",
            GOLD_YF: "macro_gold",
        }.get(sym, f"macro_{sym}")
        _store(key, df)

    # Compute QQQ→NAS100 scaling ratio from macro data
    ndx_df = macro_results.get(NDX_YF)
    qqq_5m = price_results.get(NAS100_LABEL, (None, None, None))[0]
    ratio  = 40.0   # fallback
    if ndx_df is not None and not ndx_df.empty:
        ndx_close = float(ndx_df['Close'].iloc[-1])
        # Get QQQ 1d close
        qqq_1d = price_results.get(NAS100_LABEL, (None, None, None))[2]
        if qqq_1d is not None and not qqq_1d.empty:
            qqq_close = float(qqq_1d['Close'].iloc[-1])
            if qqq_close > 0:
                ratio = ndx_close / qqq_close
    _store("qqq_ndx_ratio", ratio)

    if any_success:
        st.session_state["data_fetch_ts"] = time.time()
        print(f"[data_fetcher] Complete in {time.time()-start:.1f}s | ratio={ratio:.1f}")

    return any_success


# ── PUBLIC ACCESSORS — price data ─────────────────────────────────────────────

def get_5m(label: str) -> Optional[pd.DataFrame]:
    return _load(f"df_5m_{label}")

def get_1h(label: str) -> Optional[pd.DataFrame]:
    return _load(f"df_1h_{label}")

def get_1d(label: str) -> Optional[pd.DataFrame]:
    return _load(f"df_1d_{label}")

def get_qqq_ndx_ratio() -> float:
    return _load("qqq_ndx_ratio") or 40.0

def get_heatmap_data(label: str) -> Optional[pd.DataFrame]:
    """Returns 1h data for heatmap (intraday resolution). Falls back to 1d."""
    df = _load(f"df_1h_{label}")
    if df is not None and not df.empty:
        return df
    return _load(f"df_1d_{label}")

def get_vix() -> Optional[float]:
    """VIX latest close — used by iv_calculator."""
    df = _load("macro_vix")
    if df is not None and not df.empty:
        return float(df['Close'].iloc[-1])
    return None


# ── PUBLIC ACCESSORS — macro data ─────────────────────────────────────────────

def get_macro_df(instrument: str) -> Optional[pd.DataFrame]:
    """
    Returns daily DataFrame for a macro instrument.
    instrument: 'tnx' | 'oil' | 'qqqe' | 'ndx' | 'vix'
    """
    return _load(f"macro_{instrument}")

def get_yield_10y() -> Optional[float]:
    """
    US 10Y Treasury yield latest close (%).
    ^TNX returns yield directly (e.g. 4.38).
    IEF fallback returns price (~$95) — we skip yield calculation in that case.
    """
    df = get_macro_df("tnx")
    if df is not None and not df.empty:
        val = float(df['Close'].iloc[-1])
        # ^TNX yield is 3-6%, IEF price is 80-110 — easy to distinguish
        if val < 15:
            return val   # genuine yield %
        # IEF price — approximate yield (IEF ~$95 ≈ 4% yield, inverse relationship)
        return round(max(0, 10 - (val / 11)), 2)
    return None

def get_oil_price() -> Optional[float]:
    """Brent crude latest close (USD)."""
    df = get_macro_df("oil")
    if df is not None and not df.empty:
        return float(df['Close'].iloc[-1])
    return None

def get_qqqe_df() -> Optional[pd.DataFrame]:
    """Equal-weight Nasdaq-100 daily data."""
    return get_macro_df("qqqe")

def get_gold_df() -> Optional[pd.DataFrame]:
    """Gold ETF (GLD) daily data — for cross-asset risk-off detection."""
    return _load("macro_gold")

def get_qqq_1d() -> Optional[pd.DataFrame]:
    """QQQ daily data (stored under NAS100 label)."""
    return _load(f"df_1d_{NAS100_LABEL}")
