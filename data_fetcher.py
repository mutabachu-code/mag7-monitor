"""
data_fetcher.py
---------------
Data fetcher using yf.download() instead of yf.Ticker().history().
yf.download() uses a different code path with better cookie/session handling
on cloud servers (Streamlit Cloud, Heroku, Railway etc).

Provides the same interface as the original:
  fetch_all_data()  — fetches all tickers, returns True/False
  get_5m(label)     — returns cached 5m DataFrame
  get_1h(label)     — returns cached 1H DataFrame
  get_1d(label)     — returns cached daily DataFrame
  get_vix()         — returns VIX float
  get_heatmap_data(label) — returns 1H data for heatmap
  get_qqq_ndx_ratio()     — QQQ→NAS100 multiplier
  get_gold_df()           — Gold daily
  get_macro_df(key)       — macro instrument daily
  MAG7                    — list of Mag7 tickers
  NAS100_LABEL            — "NAS100"
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import time
import threading
from typing import Optional

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
NAS100_LABEL = "NAS100"
QQQ_TICKER   = "QQQ"    # proxy for NAS100 (multiply by ratio)
NDX_TICKER   = "^NDX"   # NAS100 index for ratio calculation

MAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]

MACRO_TICKERS = {
    "vix":   "^VIX",
    "gold":  "GC=F",
    "oil":   "CL=F",
    "tnx":   "^TNX",
    "qqqe":  "QQQE",
    "ndx":   "^NDX",
    "dxy":   "DX-Y.NYB",
    "spy":   "SPY",
}

ALL_TICKERS = [QQQ_TICKER] + MAG7 + list(MACRO_TICKERS.values())

# Cache TTLs
TTL_5M  = 60    # 1 min
TTL_1H  = 300   # 5 min
TTL_1D  = 900   # 15 min

# ── SESSION PATCH — applied once ──────────────────────────────────────────────
def _patch_session():
    """Apply yfinance session patch for cloud compatibility."""
    if st.session_state.get("_df_patched"):
        return
    try:
        import os, shutil

        # Clear stale SQLite cache
        try:
            from platformdirs import user_cache_dir
            cache_dir = user_cache_dir("py-yfinance")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
                os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass

        # Try curl_cffi (best for cloud — mimics real browser TLS)
        try:
            from curl_cffi import requests as cffi_req
            _s = cffi_req.Session(impersonate="chrome110")
            yf.utils.get_json.__globals__['requests'] = _s
        except Exception:
            pass

        st.session_state["_df_patched"] = True
    except Exception:
        st.session_state["_df_patched"] = True


# ── CACHE HELPERS ─────────────────────────────────────────────────────────────
def _cv(key: str, ttl: int) -> bool:
    return (time.time() - st.session_state.get(f"_df_ts_{key}", 0)) < ttl

def _store(key: str, data):
    st.session_state[f"_df_{key}"] = data
    st.session_state[f"_df_ts_{key}"] = time.time()

def _load(key: str):
    return st.session_state.get(f"_df_{key}")


# ── CORE FETCH — yf.download() with threading timeout ─────────────────────────
def _download(tickers, period, interval, timeout_s=20) -> Optional[pd.DataFrame]:
    """
    Thread-safe yf.download() with hard timeout.
    Returns MultiIndex DataFrame or None on failure/timeout.
    """
    result = [None]
    error  = [None]

    def _fetch():
        try:
            df = yf.download(
                tickers,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                group_by="ticker" if len(tickers) > 1 else "column",
                threads=True,
                ignore_tz=True,
            )
            result[0] = df if not df.empty else None
        except Exception as e:
            error[0] = str(e)

    t = threading.Thread(target=_fetch, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        print(f"[data_fetcher] Timeout fetching {tickers} {interval}")
        return None
    if error[0]:
        print(f"[data_fetcher] Error fetching {tickers} {interval}: {error[0]}")
        return None
    return result[0]


def _extract(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Extract single ticker from multi-ticker download result."""
    if df is None or df.empty:
        return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(0):
                sub = df[ticker].dropna(how="all")
            elif ticker in df.columns.get_level_values(1):
                sub = df.xs(ticker, axis=1, level=1).dropna(how="all")
            else:
                return None
        else:
            sub = df.copy()
        sub.columns = [c.capitalize() for c in sub.columns]
        return sub if not sub.empty else None
    except Exception as e:
        print(f"[data_fetcher] Extract error {ticker}: {e}")
        return None


# ── QQQ→NAS100 RATIO ─────────────────────────────────────────────────────────
def get_qqq_ndx_ratio() -> float:
    cached = _load("ratio")
    if cached:
        return cached
    try:
        df = _download([QQQ_TICKER, NDX_TICKER], "5d", "1d", timeout_s=15)
        if df is not None:
            qqq = _extract(df, QQQ_TICKER)
            ndx = _extract(df, NDX_TICKER)
            if qqq is not None and ndx is not None:
                qqq_p = float(qqq['Close'].iloc[-1])
                ndx_p = float(ndx['Close'].iloc[-1])
                ratio = ndx_p / qqq_p if qqq_p > 0 else 40.0
                _store("ratio", ratio)
                return ratio
    except Exception as e:
        print(f"[data_fetcher] Ratio error: {e}")
    return st.session_state.get("_df_ratio", 40.0)  # default


# ── MAIN FETCH ────────────────────────────────────────────────────────────────
def fetch_all_data() -> bool:
    """
    Fetch all required data. Returns True if at least NAS100 5m data is available.
    Uses yf.download() in batches to minimise requests and avoid rate limiting.
    """
    _patch_session()

    success = False
    now = time.time()

    # ── BATCH 1: 5m data (NAS100 + Mag7 + macro instruments) ─────────────────
    need_5m = [QQQ_TICKER] + MAG7
    if not _cv(f"5m_{QQQ_TICKER}", TTL_5M):
        try:
            df5 = _download(need_5m, "2d", "5m", timeout_s=25)
            if df5 is not None:
                for ticker in need_5m:
                    label = NAS100_LABEL if ticker == QQQ_TICKER else ticker
                    sub   = _extract(df5, ticker)
                    if sub is not None:
                        _store(f"5m_{ticker}", sub)
                        if ticker == QQQ_TICKER:
                            success = True
        except Exception as e:
            print(f"[data_fetcher] 5m batch error: {e}")

    # Check stale cache
    if not success:
        cached = _load(f"5m_{QQQ_TICKER}")
        if cached is not None and not cached.empty:
            success = True   # stale but usable

    # ── BATCH 2: 1H data ──────────────────────────────────────────────────────
    need_1h = [QQQ_TICKER] + MAG7
    if not _cv(f"1h_{QQQ_TICKER}", TTL_1H):
        try:
            df1h = _download(need_1h, "60d", "1h", timeout_s=20)
            if df1h is not None:
                for ticker in need_1h:
                    sub = _extract(df1h, ticker)
                    if sub is not None:
                        _store(f"1h_{ticker}", sub)
        except Exception as e:
            print(f"[data_fetcher] 1H batch error: {e}")

    # ── BATCH 3: Daily data ────────────────────────────────────────────────────
    need_1d = [QQQ_TICKER, NDX_TICKER] + MAG7
    if not _cv(f"1d_{QQQ_TICKER}", TTL_1D):
        try:
            df1d = _download(need_1d, "252d", "1d", timeout_s=20)
            if df1d is not None:
                for ticker in need_1d:
                    label = NAS100_LABEL if ticker == QQQ_TICKER else ticker
                    sub   = _extract(df1d, ticker)
                    if sub is not None:
                        _store(f"1d_{ticker}", sub)
                get_qqq_ndx_ratio()  # refresh ratio from fresh data
        except Exception as e:
            print(f"[data_fetcher] Daily batch error: {e}")

    # ── BATCH 4: Macro instruments ────────────────────────────────────────────
    macro_tickers = list(MACRO_TICKERS.values())
    if not _cv("macro", TTL_1D):
        try:
            dfm = _download(macro_tickers, "30d", "1d", timeout_s=20)
            if dfm is not None:
                for key, ticker in MACRO_TICKERS.items():
                    sub = _extract(dfm, ticker)
                    if sub is not None:
                        _store(f"macro_{key}", sub)
                _store("macro_ts", now)
        except Exception as e:
            print(f"[data_fetcher] Macro error: {e}")

    return success


# ── GETTERS ───────────────────────────────────────────────────────────────────

def get_5m(label: str) -> Optional[pd.DataFrame]:
    ticker = QQQ_TICKER if label == NAS100_LABEL else label
    return _load(f"5m_{ticker}")

def get_1h(label: str) -> Optional[pd.DataFrame]:
    ticker = QQQ_TICKER if label == NAS100_LABEL else label
    return _load(f"1h_{ticker}")

def get_1d(label: str) -> Optional[pd.DataFrame]:
    ticker = QQQ_TICKER if label == NAS100_LABEL else label
    return _load(f"1d_{ticker}")

def get_heatmap_data(label: str) -> Optional[pd.DataFrame]:
    """1H data used for the hourly heatmap."""
    return get_1h(label)

def get_vix() -> Optional[float]:
    df = _load("macro_^VIX")
    if df is None:
        df = _load("macro_vix")
    if df is not None and not df.empty:
        try:
            return float(df['Close'].iloc[-1])
        except Exception:
            pass
    return None

def get_gold_df() -> Optional[pd.DataFrame]:
    return _load("macro_GC=F") or _load("macro_gold")

def get_macro_df(key: str) -> Optional[pd.DataFrame]:
    """
    Returns macro instrument daily DataFrame by key.
    Keys: 'vix', 'gold', 'oil', 'tnx', 'qqqe', 'ndx', 'dxy', 'spy'
    Also accepts raw ticker like '^TNX', 'GC=F' etc.
    """
    # Try by key first
    df = _load(f"macro_{key}")
    if df is not None:
        return df
    # Try by ticker value
    ticker = MACRO_TICKERS.get(key.lower())
    if ticker:
        df = _load(f"macro_{ticker}")
        if df is not None:
            return df
    # Try raw ticker key
    return _load(f"macro_{key.upper()}")
