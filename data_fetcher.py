"""
data_fetcher.py  —  v3
-----------------------
Fetches market data with:
- Individual yf.Ticker calls (more reliable than yf.download on Streamlit Cloud)
- Hard 8-second timeout per ticker via threading
- 65-second session cache — only fetches when cache expires
- Never blocks the Streamlit thread
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import streamlit as st
import threading
from typing import Dict, Optional, Tuple

MAG7         = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
NAS100_LABEL = 'NAS100'
NAS100_YF    = 'QQQ'   # QQQ ETF = reliable Nasdaq-100 proxy (^NDX often blocked by yfinance)
VIX_YF       = '^VIX'
ALL_LABELS   = [NAS100_LABEL] + MAG7

CACHE_TTL    = 65    # seconds
FETCH_TIMEOUT = 8   # seconds per ticker before giving up


# ── TIMEOUT WRAPPER ───────────────────────────────────────────────────────────

def _fetch_with_timeout(func, timeout: int = FETCH_TIMEOUT):
    """
    Run func() in a thread. Return result or None if it exceeds timeout.
    Prevents yfinance hangs from freezing the entire Streamlit app.
    """
    result = [None]
    error  = [None]

    def target():
        try:
            result[0] = func()
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        print(f"[data_fetcher] Timeout after {timeout}s")
        return None
    if error[0]:
        print(f"[data_fetcher] Fetch error: {error[0]}")
        return None
    return result[0]


# ── CACHE HELPERS ─────────────────────────────────────────────────────────────

def _cache_valid() -> bool:
    last = st.session_state.get("data_fetch_ts", 0)
    return (time.time() - last) < CACHE_TTL


def _store_cache(key: str, data):
    st.session_state[key] = data


def _load_cache(key: str):
    return st.session_state.get(key)


# ── SINGLE TICKER FETCH ───────────────────────────────────────────────────────

def _fetch_ticker(label: str) -> Tuple[Optional[pd.DataFrame],
                                        Optional[pd.DataFrame],
                                        Optional[pd.DataFrame]]:
    """
    Fetch 5m, 1h, and 1d data for one ticker.
    Returns (df_5m, df_1h, df_1d) — any can be None on failure.
    """
    yfticker = NAS100_YF if label == NAS100_LABEL else label

    def get_5m():
        df = yf.Ticker(yfticker).history(
            period="5d", interval="5m", prepost=True
        ).ffill().bfill()
        return df if not df.empty else None

    def get_1h():
        df = yf.Ticker(yfticker).history(
            period="60d", interval="1h"
        ).ffill().bfill()
        return df if not df.empty else None

    def get_1d():
        df = yf.Ticker(yfticker).history(
            period="365d", interval="1d"
        ).ffill().bfill()
        return df if not df.empty else None

    df_5m = _fetch_with_timeout(get_5m, FETCH_TIMEOUT)
    df_1h = _fetch_with_timeout(get_1h, FETCH_TIMEOUT)
    df_1d = _fetch_with_timeout(get_1d, FETCH_TIMEOUT)

    return df_5m, df_1h, df_1d


# ── VIX FETCH ─────────────────────────────────────────────────────────────────

def _fetch_vix() -> Optional[float]:
    def get():
        df = yf.Ticker(VIX_YF).history(period="5d", interval="1d")
        return float(df['Close'].iloc[-1]) if not df.empty else None
    return _fetch_with_timeout(get, FETCH_TIMEOUT)


# ── PARALLEL FETCH ALL ────────────────────────────────────────────────────────

def fetch_all_data() -> bool:
    """
    Fetch all tickers in parallel threads with timeouts.
    Stores results in session_state. Returns True if at least some data loaded.
    """
    if _cache_valid():
        return True   # cache still fresh

    print(f"[data_fetcher] Starting parallel fetch for {len(ALL_LABELS)} tickers")
    start = time.time()

    results: Dict[str, Tuple] = {}
    threads = []

    def fetch_and_store(label):
        results[label] = _fetch_ticker(label)

    # Launch all fetches in parallel
    for label in ALL_LABELS:
        t = threading.Thread(target=fetch_and_store, args=(label,), daemon=True)
        threads.append(t)
        t.start()

    # Also fetch VIX in parallel
    vix_result = [None]
    def fetch_vix_thread():
        vix_result[0] = _fetch_vix()
    vix_thread = threading.Thread(target=fetch_vix_thread, daemon=True)
    vix_thread.start()

    # Wait for all (max FETCH_TIMEOUT + 2s buffer)
    for t in threads:
        t.join(timeout=FETCH_TIMEOUT + 2)
    vix_thread.join(timeout=FETCH_TIMEOUT + 2)

    # Store results
    any_success = False
    for label, (df_5m, df_1h, df_1d) in results.items():
        if df_5m is not None or df_1h is not None:
            any_success = True
        _store_cache(f"df_5m_{label}", df_5m)
        _store_cache(f"df_1h_{label}", df_1h)
        _store_cache(f"df_1d_{label}", df_1d)

    _store_cache("vix_value", vix_result[0])

    if any_success:
        st.session_state["data_fetch_ts"] = time.time()
        print(f"[data_fetcher] Fetch complete in {time.time()-start:.1f}s")

    return any_success


# ── PUBLIC ACCESSORS ──────────────────────────────────────────────────────────

def get_5m(label: str) -> Optional[pd.DataFrame]:
    return _load_cache(f"df_5m_{label}")

def get_1h(label: str) -> Optional[pd.DataFrame]:
    return _load_cache(f"df_1h_{label}")

def get_1d(label: str) -> Optional[pd.DataFrame]:
    return _load_cache(f"df_1d_{label}")

def get_vix() -> Optional[float]:
    return _load_cache("vix_value")

def get_heatmap_data(label: str) -> Optional[pd.DataFrame]:
    """
    Returns 1h data for heatmap (best resolution).
    Falls back to 1d if 1h unavailable.
    Uses QQQ for NAS100 label.
    """
    df = _load_cache(f"df_1h_{label}")
    if df is not None and not df.empty:
        return df
    return _load_cache(f"df_1d_{label}")
