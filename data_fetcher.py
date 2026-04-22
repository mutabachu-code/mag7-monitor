"""
data_fetcher.py
---------------
Single data fetching layer for the entire dashboard.
Batches ALL yfinance calls into as few requests as possible.
Caches results for 60 seconds so the dashboard never blocks.

Before this fix: 48 individual yfinance calls per refresh
After this fix : 4 batched yfinance calls per refresh
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import streamlit as st
from typing import Dict, Optional

# Ticker mappings
MAG7         = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
NAS100_YF    = '^NDX'
VIX_YF       = '^VIX'
ALL_YF       = MAG7 + [NAS100_YF, VIX_YF]

CACHE_TTL    = 65   # seconds — slightly longer than 60s refresh interval


def _cache_valid(key: str) -> bool:
    ts = st.session_state.get(f"{key}_ts", 0)
    return (time.time() - ts) < CACHE_TTL


def _store(key: str, data):
    st.session_state[key]        = data
    st.session_state[f"{key}_ts"] = time.time()


def _load(key: str):
    return st.session_state.get(key)


# ── BATCH FETCH ───────────────────────────────────────────────────────────────

def fetch_all_data() -> bool:
    """
    Master fetch function — call once per refresh cycle.
    Downloads all price data in 3 batched calls and stores in session state.
    Returns True if successful, False if data unavailable.
    """
    if _cache_valid("prices_5m"):
        return True   # cache still fresh, nothing to do

    try:
        tickers_str = " ".join(MAG7 + [NAS100_YF])

        # Batch call 1: 5-minute data for signals (all Mag7 + NAS100)
        df_5m = yf.download(
            tickers_str,
            period="5d",
            interval="5m",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            timeout=20,
        )

        # Batch call 2: 1-hour data for SMA200 + MACD (all Mag7 + NAS100)
        df_1h = yf.download(
            tickers_str,
            period="60d",
            interval="1h",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            timeout=20,
        )

        # Batch call 3: daily data for IV rank/percentile + heatmap (all + VIX)
        df_1d = yf.download(
            " ".join(MAG7 + [NAS100_YF, VIX_YF]),
            period="365d",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            timeout=20,
        )

        _store("prices_5m", df_5m)
        _store("prices_1h", df_1h)
        _store("prices_1d", df_1d)
        return True

    except Exception as e:
        print(f"[data_fetcher] Batch fetch error: {e}")
        return False


# ── DATA ACCESSORS ────────────────────────────────────────────────────────────

def get_5m(ticker: str) -> Optional[pd.DataFrame]:
    """Get 5m OHLCV for a single ticker from the batched data."""
    df = _load("prices_5m")
    if df is None or df.empty:
        return None
    yfticker = NAS100_YF if ticker == 'NAS100' else ticker
    try:
        if isinstance(df.columns, pd.MultiIndex):
            data = df[yfticker].copy() if yfticker in df.columns.get_level_values(0) else None
            if data is None:
                # Try ticker directly (sometimes no MultiIndex for single ticker)
                data = df.copy()
        else:
            data = df.copy()
        if data is not None:
            data = data.ffill().bfill()
            data.columns = [c.capitalize() for c in data.columns]
        return data
    except Exception as e:
        print(f"[data_fetcher] get_5m error {ticker}: {e}")
        return None


def get_1h(ticker: str) -> Optional[pd.DataFrame]:
    """Get 1h OHLCV for a single ticker from the batched data."""
    df = _load("prices_1h")
    if df is None or df.empty:
        return None
    yfticker = NAS100_YF if ticker == 'NAS100' else ticker
    try:
        if isinstance(df.columns, pd.MultiIndex):
            data = df[yfticker].copy() if yfticker in df.columns.get_level_values(0) else None
            if data is None:
                data = df.copy()
        else:
            data = df.copy()
        if data is not None:
            data = data.ffill().bfill()
            data.columns = [c.capitalize() for c in data.columns]
        return data
    except Exception as e:
        print(f"[data_fetcher] get_1h error {ticker}: {e}")
        return None


def get_1d(ticker: str) -> Optional[pd.DataFrame]:
    """Get daily OHLCV for a single ticker from the batched data."""
    df = _load("prices_1d")
    if df is None or df.empty:
        return None
    yfticker = NAS100_YF if ticker == 'NAS100' else ticker
    try:
        if isinstance(df.columns, pd.MultiIndex):
            data = df[yfticker].copy() if yfticker in df.columns.get_level_values(0) else None
            if data is None:
                data = df.copy()
        else:
            data = df.copy()
        if data is not None:
            data = data.ffill().bfill()
            data.columns = [c.capitalize() for c in data.columns]
        return data
    except Exception as e:
        print(f"[data_fetcher] get_1d error {ticker}: {e}")
        return None


def get_vix() -> Optional[float]:
    """Get latest VIX close from batched daily data."""
    df = get_1d('VIX')
    if df is None or df.empty:
        return None
    try:
        return float(df['Close'].iloc[-1])
    except Exception:
        return None
