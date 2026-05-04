"""
forex_data_fetcher.py  —  v2
------------------------------
Fetches ALL forex + macro instruments in one parallel batch.
Stores everything in session_state cache (65s TTL).

Instruments:
  Forex pairs (1h + 4h resampled + 1d):
    EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD

  Macro (1d — shared with Mag7 dashboard context):
    ^TNX  — US 10Y Yield
    BZ=F  — Brent Crude Oil
    QQQ   — Nasdaq proxy (for yield/equity context)
    ^VIX  — Volatility index (risk-on/risk-off)
    DXY   — US Dollar Index (UUP ETF as proxy)

Before v2: 7 pairs × 3 timeframes = 21 calls (sequential per pair)
After  v2: all 7 pairs + 5 macro = 12 parallel threads, zero sequential waits
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import streamlit as st
import threading
from typing import Optional, Tuple

# ── FOREX PAIRS ───────────────────────────────────────────────────────────────
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD', 'NZDUSD']

YF_MAP = {
    'EURUSD': 'EURUSD=X', 'GBPUSD': 'GBPUSD=X', 'USDJPY': 'USDJPY=X',
    'USDCHF': 'USDCHF=X', 'AUDUSD': 'AUDUSD=X', 'USDCAD': 'USDCAD=X',
    'NZDUSD': 'NZDUSD=X',
}

PIP_MAP = {
    'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01,
    'USDCHF': 0.0001, 'AUDUSD': 0.0001, 'USDCAD': 0.0001, 'NZDUSD': 0.0001,
}

# ── MACRO INSTRUMENTS (daily) ─────────────────────────────────────────────────
MACRO_INSTRUMENTS = {
    'tnx':  '^TNX',    # US 10Y yield (fallback: IEF ETF)
    'oil':  'BZ=F',    # Brent crude (fallback: USO ETF)
    'qqq':  'QQQ',     # Nasdaq proxy
    'vix':  '^VIX',    # Volatility
    'dxy':  'UUP',     # USD Index proxy
    'gold': 'GLD',     # Gold ETF
}

# Fallback symbols for unreliable futures/index tickers
MACRO_FALLBACKS = {
    '^TNX': 'IEF',    # 7-10Y Treasury ETF
    'BZ=F': 'USO',    # Oil ETF
}

CACHE_TTL     = 65
FETCH_TIMEOUT = 10


# ── TIMEOUT WRAPPER ───────────────────────────────────────────────────────────

def _fetch_with_timeout(func, timeout=FETCH_TIMEOUT):
    result = [None]
    def target():
        try:
            result[0] = func()
        except Exception as e:
            print(f"[forex_fetcher] {e}")
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result[0]


# ── CACHE HELPERS ─────────────────────────────────────────────────────────────

def _cache_valid() -> bool:
    return (time.time() - st.session_state.get("fx_fetch_ts", 0)) < CACHE_TTL

def _store(key, data):
    st.session_state[key] = data

def _load(key):
    return st.session_state.get(key)


# ── FOREX PAIR FETCH (1h + 4h + 1d) ──────────────────────────────────────────

def _fetch_pair(pair: str) -> Tuple[Optional[pd.DataFrame],
                                     Optional[pd.DataFrame],
                                     Optional[pd.DataFrame]]:
    sym = YF_MAP[pair]

    def get_1h():
        df = yf.Ticker(sym).history(period="30d", interval="1h").ffill().bfill()
        return df if not df.empty else None

    def get_4h():
        # Resample 1h → 4h (avoids a separate API call)
        df = yf.Ticker(sym).history(period="60d", interval="1h").ffill().bfill()
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        df4 = df.resample('4h').agg({
            'Open': 'first', 'High': 'max',
            'Low': 'min',    'Close': 'last', 'Volume': 'sum'
        }).dropna()
        return df4 if not df4.empty else None

    def get_1d():
        df = yf.Ticker(sym).history(period="365d", interval="1d").ffill().bfill()
        return df if not df.empty else None

    # Run all three with individual timeouts
    df_1h = _fetch_with_timeout(get_1h, FETCH_TIMEOUT)
    df_4h = _fetch_with_timeout(get_4h, FETCH_TIMEOUT)
    df_1d = _fetch_with_timeout(get_1d, FETCH_TIMEOUT)
    return df_1h, df_4h, df_1d


# ── MACRO FETCH (daily) ───────────────────────────────────────────────────────

def _fetch_macro(key: str, symbol: str) -> Optional[pd.DataFrame]:
    def get():
        period = "30d" if key in ('tnx', 'qqq', 'dxy') else "5d"
        df = yf.Ticker(symbol).history(period=period, interval="1d").ffill().bfill()
        if not df.empty:
            last_date = pd.Timestamp(df.index[-1]).date()
            days_old  = (pd.Timestamp.now().date() - last_date).days
            if days_old <= 4:
                return df
            print(f"[forex_fetcher] {symbol} stale ({days_old}d) — trying fallback")
        fallback = MACRO_FALLBACKS.get(symbol)
        if fallback:
            df2 = yf.Ticker(fallback).history(period="5d", interval="1d").ffill().bfill()
            if not df2.empty:
                return df2
        return None
    return _fetch_with_timeout(get, FETCH_TIMEOUT)


# ── MASTER FETCH ──────────────────────────────────────────────────────────────

def fetch_all_pairs() -> bool:
    """
    Fetch all forex pairs + macro instruments in a single parallel batch.
    Returns True if at least some data loaded.
    """
    if _cache_valid():
        return True

    print(f"[forex_fetcher] Parallel fetch: {len(PAIRS)} pairs + {len(MACRO_INSTRUMENTS)} macro")
    start   = time.time()
    results = {}
    macro_r = {}
    threads = []

    # Forex pair threads
    def fetch_and_store(pair):
        results[pair] = _fetch_pair(pair)

    for pair in PAIRS:
        t = threading.Thread(target=fetch_and_store, args=(pair,), daemon=True)
        threads.append(t)
        t.start()

    # Macro threads
    def fetch_macro_store(key, sym):
        macro_r[key] = _fetch_macro(key, sym)

    for key, sym in MACRO_INSTRUMENTS.items():
        t = threading.Thread(target=fetch_macro_store, args=(key, sym), daemon=True)
        threads.append(t)
        t.start()

    # Wait for all
    for t in threads:
        t.join(timeout=FETCH_TIMEOUT + 2)

    # Store forex pair data
    any_ok = False
    for pair, (df_1h, df_4h, df_1d) in results.items():
        if df_1h is not None:
            any_ok = True
        _store(f"fx_1h_{pair}", df_1h)
        _store(f"fx_4h_{pair}", df_4h)
        _store(f"fx_1d_{pair}", df_1d)

    # Store macro data
    for key, df in macro_r.items():
        _store(f"fx_macro_{key}", df)

    if any_ok:
        st.session_state["fx_fetch_ts"] = time.time()
        print(f"[forex_fetcher] Done in {time.time()-start:.1f}s")
    return any_ok


# ── PUBLIC ACCESSORS — forex pairs ────────────────────────────────────────────

def get_1h(pair: str) -> Optional[pd.DataFrame]:
    return _load(f"fx_1h_{pair}")

def get_4h(pair: str) -> Optional[pd.DataFrame]:
    return _load(f"fx_4h_{pair}")

def get_1d(pair: str) -> Optional[pd.DataFrame]:
    return _load(f"fx_1d_{pair}")

def get_pip(pair: str) -> float:
    return PIP_MAP.get(pair, 0.0001)


# ── PUBLIC ACCESSORS — macro ──────────────────────────────────────────────────

def get_fx_macro(key: str) -> Optional[pd.DataFrame]:
    """
    Returns daily macro DataFrame.
    key: 'tnx' | 'oil' | 'qqq' | 'vix' | 'dxy'
    """
    return _load(f"fx_macro_{key}")

def get_fx_yield_10y() -> Optional[float]:
    """US 10Y yield latest close (%)."""
    df = get_fx_macro("tnx")
    if df is not None and not df.empty:
        return float(df['Close'].iloc[-1])
    return None

def get_fx_oil() -> Optional[float]:
    """Brent crude latest close (USD)."""
    df = get_fx_macro("oil")
    if df is not None and not df.empty:
        return float(df['Close'].iloc[-1])
    return None

def get_fx_vix() -> Optional[float]:
    """VIX latest close."""
    df = get_fx_macro("vix")
    if df is not None and not df.empty:
        return float(df['Close'].iloc[-1])
    return None

def get_fx_gold_df():
    """Gold ETF daily data for cross-asset risk-off check."""
    return get_fx_macro("gold")

def get_fx_dxy_trend() -> str:
    """
    USD strength trend from UUP ETF.
    Returns 'STRONG' | 'NEUTRAL' | 'WEAK'
    """
    df = get_fx_macro("dxy")
    if df is None or len(df) < 10:
        return "NEUTRAL"
    close    = df['Close']
    sma10    = close.rolling(10).mean().iloc[-1]
    current  = close.iloc[-1]
    pct      = (current - sma10) / sma10 * 100
    if pct > 0.3:   return "STRONG"
    if pct < -0.3:  return "WEAK"
    return "NEUTRAL"
