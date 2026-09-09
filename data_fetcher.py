"""
data_fetcher.py  — v6 (Alpaca + yfinance hybrid)
-------------------------------------------------
PERMANENT FIX for Yahoo Finance blocking on Streamlit Cloud.

Data sources:
  Intraday (5m, 1H):  Alpaca Markets REST API (free, no cookies, never breaks)
  Daily (1D):         Alpaca Markets REST API
  Macro daily:        yfinance (daily bars only — far less rate-limited than intraday)
  Fallback:           yfinance for anything Alpaca cannot provide

Alpaca free tier covers:
  QQQ, AAPL, MSFT, NVDA, AMZN, META, GOOGL, TSLA — all 8 tickers
  Real-time IEX feed, unlimited REST calls, no expiry, no cookies

SETUP (one-time, 2 minutes):
  1. Sign up at alpaca.markets (free, no credit card)
  2. Go to Paper Trading → API Keys → Generate
  3. Add to Streamlit Secrets:
       ALPACA_API_KEY    = "PKXXXXXXXXXXXXXXXXXXXXXXXX"
       ALPACA_API_SECRET = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
  Never expires. Never needs updating.

If Alpaca keys not set → falls back to yfinance with cookie injection.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import requests
import threading
import time
import os
import shutil
from typing import Optional, Tuple, Dict
from datetime import datetime, timezone, timedelta

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MAG7         = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
NAS100_LABEL = 'NAS100'
NAS100_YF    = 'QQQ'
ALL_LABELS   = [NAS100_LABEL] + MAG7

ALPACA_BASE  = "https://data.alpaca.markets/v2"
ALPACA_TICKERS = {
    NAS100_LABEL: "QQQ",
    "AAPL": "AAPL", "MSFT": "MSFT", "GOOGL": "GOOGL", "AMZN": "AMZN",
    "TSLA": "TSLA", "META": "META", "NVDA": "NVDA",
}
# Macro via Alpaca (ETF proxies — all tradeable on US exchanges)
ALPACA_MACRO = {
    "vix":  "VIXY",   # ProShares VIX Short-Term Futures ETF
    "gold": "GLD",    # SPDR Gold Trust
    "oil":  "USO",    # United States Oil Fund
    "bond": "TLT",    # iShares 20+ Year Treasury Bond ETF
    "qqqe": "QQQE",   # Direxion Nasdaq-100 Equal Weight
    "spy":  "SPY",    # S&P 500 ETF
}
# yfinance macro tickers (for instruments not on Alpaca)
YF_MACRO = {
    "tnx":  "^TNX",   # 10Y Treasury yield (index, not ETF)
    "ndx":  "^NDX",   # Nasdaq-100 index (for ratio calculation)
    "vix_raw": "^VIX", # VIX index raw
}

CACHE_TTL     = 60    # seconds
FETCH_TIMEOUT = 15    # seconds per batch

# ── CACHE ─────────────────────────────────────────────────────────────────────
def _cache_valid(key: str = "data_fetch_ts") -> bool:
    return (time.time() - st.session_state.get(key, 0)) < CACHE_TTL

def _store(key: str, data):
    st.session_state[key] = data

def _load(key: str):
    return st.session_state.get(key)


# ── ALPACA SESSION ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_alpaca_session() -> Optional[requests.Session]:
    """Build authenticated Alpaca session from Streamlit Secrets."""
    try:
        key    = st.secrets.get("ALPACA_API_KEY", "")
        secret = st.secrets.get("ALPACA_API_SECRET", "")
        if not key or not secret:
            return None
        session = requests.Session()
        session.headers.update({
            "APCA-API-KEY-ID":     key,
            "APCA-API-SECRET-KEY": secret,
            "Accept":              "application/json",
        })
        return session
    except Exception:
        return None


def _alpaca_bars(symbols: list, timeframe: str, limit: int,
                 session: requests.Session) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical bars from Alpaca for multiple symbols.
    timeframe: "5Min" | "1Hour" | "1Day"
    Returns dict: {symbol: DataFrame with OHLCV columns}
    """
    results = {}
    # Alpaca supports multi-symbol in one call
    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "limit": limit,
        "adjustment": "all",
        "feed": "iex",   # free tier feed
    }
    try:
        r = session.get(f"{ALPACA_BASE}/stocks/bars", params=params, timeout=FETCH_TIMEOUT)
        if r.status_code != 200:
            print(f"[data_fetcher] Alpaca {r.status_code}: {r.text[:100]}")
            return results
        data = r.json().get("bars", {})
        for sym, bars in data.items():
            if not bars:
                continue
            df = pd.DataFrame(bars)
            df["t"] = pd.to_datetime(df["t"])
            df = df.set_index("t").rename(columns={
                "o": "Open", "h": "High", "l": "Low",
                "c": "Close", "v": "Volume",
            })
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.index = df.index.tz_convert("UTC")
            df = df.ffill().bfill()
            results[sym] = df
    except Exception as e:
        print(f"[data_fetcher] Alpaca bars error: {e}")
    return results


def _alpaca_fetch_all(session: requests.Session) -> bool:
    """Fetch all price data from Alpaca in parallel batches."""
    success = False
    price_syms = list(ALPACA_TICKERS.values())
    macro_syms = list(ALPACA_MACRO.values())

    def fetch_5m():
        bars = _alpaca_bars(price_syms, "5Min", 390, session)
        for label, sym in ALPACA_TICKERS.items():
            if sym in bars:
                _store(f"df_5m_{label}", bars[sym])

    def fetch_1h():
        bars = _alpaca_bars(price_syms, "1Hour", 500, session)
        for label, sym in ALPACA_TICKERS.items():
            if sym in bars:
                _store(f"df_1h_{label}", bars[sym])

    def fetch_1d():
        bars = _alpaca_bars(price_syms + macro_syms, "1Day", 400, session)
        for label, sym in ALPACA_TICKERS.items():
            if sym in bars:
                _store(f"df_1d_{label}", bars[sym])
        for key, sym in ALPACA_MACRO.items():
            if sym in bars:
                _store(f"macro_{key}", bars[sym])

    threads = [
        threading.Thread(target=fetch_5m, daemon=True),
        threading.Thread(target=fetch_1h, daemon=True),
        threading.Thread(target=fetch_1d, daemon=True),
    ]
    for t in threads: t.start()
    for t in threads: t.join(timeout=FETCH_TIMEOUT + 5)

    # Check success
    qqq_5m = _load(f"df_5m_{NAS100_LABEL}")
    if qqq_5m is not None and not qqq_5m.empty:
        success = True
        # Compute ratio from daily data
        qqq_1d = _load(f"df_1d_{NAS100_LABEL}")
        if qqq_1d is not None and not qqq_1d.empty:
            # Fetch NDX index for ratio via yfinance daily (stable)
            try:
                ndx_df = yf.download("^NDX", period="5d", interval="1d", progress=False)
                if not ndx_df.empty:
                    ndx_p = float(ndx_df["Close"].iloc[-1])
                    qqq_p = float(qqq_1d["Close"].iloc[-1])
                    if qqq_p > 0:
                        _store("qqq_ndx_ratio", ndx_p / qqq_p)
                        return success
            except Exception:
                pass
        _store("qqq_ndx_ratio", 40.0)   # default ratio
    return success


# ── YFINANCE FALLBACK ──────────────────────────────────────────────────────────
def _apply_yf_patch():
    """Apply yfinance patches for cloud compatibility."""
    if getattr(_apply_yf_patch, "_done", False):
        return
    try:
        # Clear SQLite cache
        try:
            from platformdirs import user_cache_dir
            cache_dir = user_cache_dir("py-yfinance")
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir, ignore_errors=True)
                os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            pass
        # Inject Yahoo cookie from Secrets if available
        try:
            cookie_val = st.secrets.get("YAHOO_COOKIE", "")
            if cookie_val:
                session = requests.Session()
                session.headers.update({
                    "User-Agent": st.secrets.get("YAHOO_UA",
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"),
                    "Cookie": cookie_val,
                })
                yf.utils.get_json.__globals__['requests'] = session
        except Exception:
            pass
    except Exception:
        pass
    _apply_yf_patch._done = True

_apply_yf_patch()


def _yf_fetch_fallback() -> bool:
    """yfinance fallback — used when Alpaca keys not configured."""
    print("[data_fetcher] Using yfinance fallback (configure Alpaca keys for reliability)")
    success = False
    tickers = [NAS100_YF] + MAG7

    def fetch_ticker(label):
        sym = NAS100_YF if label == NAS100_LABEL else label
        try:
            df5 = yf.download(sym, period="5d", interval="5m", auto_adjust=True,
                              progress=False, threads=False)
            if not df5.empty:
                if isinstance(df5.columns, pd.MultiIndex):
                    df5.columns = df5.columns.get_level_values(0)
                df5.columns = [c.capitalize() for c in df5.columns]
                _store(f"df_5m_{label}", df5.ffill().bfill())
        except Exception as e:
            print(f"[data_fetcher] yf 5m {label}: {e}")
        try:
            df1h = yf.download(sym, period="60d", interval="1h", auto_adjust=True,
                               progress=False, threads=False)
            if not df1h.empty:
                if isinstance(df1h.columns, pd.MultiIndex):
                    df1h.columns = df1h.columns.get_level_values(0)
                df1h.columns = [c.capitalize() for c in df1h.columns]
                _store(f"df_1h_{label}", df1h.ffill().bfill())
        except Exception as e:
            print(f"[data_fetcher] yf 1h {label}: {e}")
        try:
            df1d = yf.download(sym, period="365d", interval="1d", auto_adjust=True,
                               progress=False, threads=False)
            if not df1d.empty:
                if isinstance(df1d.columns, pd.MultiIndex):
                    df1d.columns = df1d.columns.get_level_values(0)
                df1d.columns = [c.capitalize() for c in df1d.columns]
                _store(f"df_1d_{label}", df1d.ffill().bfill())
        except Exception as e:
            print(f"[data_fetcher] yf 1d {label}: {e}")

    threads = [threading.Thread(target=fetch_ticker, args=(l,), daemon=True)
               for l in ALL_LABELS]
    for t in threads: t.start()
    for t in threads: t.join(timeout=FETCH_TIMEOUT + 5)

    # Macro
    macro_map = {"^VIX": "macro_vix", "^TNX": "macro_tnx", "QQQE": "macro_qqqe",
                 "^NDX": "macro_ndx", "GLD": "macro_gold", "BZ=F": "macro_oil"}
    for sym, key in macro_map.items():
        try:
            df = yf.download(sym, period="30d", interval="1d", auto_adjust=True,
                             progress=False, threads=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.capitalize() for c in df.columns]
                _store(key, df.ffill().bfill())
        except Exception:
            pass

    qqq_5m = _load(f"df_5m_{NAS100_LABEL}")
    if qqq_5m is not None and not qqq_5m.empty:
        success = True
    return success


# ── MACRO yfinance fetch (daily only — very stable) ───────────────────────────
def _fetch_yf_macro_daily():
    """Fetch ^TNX and ^NDX via yfinance daily (daily bars are rarely blocked)."""
    for sym, key in YF_MACRO.items():
        try:
            df = yf.download(sym, period="30d", interval="1d", auto_adjust=True,
                             progress=False, threads=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.capitalize() for c in df.columns]
                _store(f"macro_{sym.replace('^','').lower()}", df.ffill().bfill())
        except Exception as e:
            print(f"[data_fetcher] yf macro {sym}: {e}")


# ── MASTER FETCH ──────────────────────────────────────────────────────────────
def fetch_all_data() -> bool:
    """
    Public entry point. Tries Alpaca first (reliable), falls back to yfinance.
    Returns True if at least NAS100 5m data is available.
    """
    if _cache_valid():
        return _load(f"df_5m_{NAS100_LABEL}") is not None

    success = False

    # Try Alpaca first
    alpaca_session = _get_alpaca_session()
    if alpaca_session:
        print("[data_fetcher] Using Alpaca (primary)")
        success = _alpaca_fetch_all(alpaca_session)
        if success:
            # Also fetch yfinance macro daily for ^TNX, ^NDX
            threading.Thread(target=_fetch_yf_macro_daily, daemon=True).start()

    # Fall back to yfinance if Alpaca not configured or failed
    if not success:
        success = _yf_fetch_fallback()

    if success:
        st.session_state["data_fetch_ts"] = time.time()
        print(f"[data_fetcher] Fetch complete — source: "
              f"{'Alpaca' if alpaca_session else 'yfinance'}")

    return success


# ── PUBLIC ACCESSORS ──────────────────────────────────────────────────────────

def get_5m(label: str) -> Optional[pd.DataFrame]:
    return _load(f"df_5m_{label}")

def get_1h(label: str) -> Optional[pd.DataFrame]:
    return _load(f"df_1h_{label}")

def get_1d(label: str) -> Optional[pd.DataFrame]:
    return _load(f"df_1d_{label}")

def get_qqq_ndx_ratio() -> float:
    return _load("qqq_ndx_ratio") or 40.0

def get_heatmap_data(label: str) -> Optional[pd.DataFrame]:
    df = _load(f"df_1h_{label}")
    if df is not None and not df.empty:
        return df
    return _load(f"df_1d_{label}")

def get_vix() -> Optional[float]:
    # Try VIX index first, then VIXY ETF proxy
    for key in ["macro_vix_raw", "macro_vix"]:
        df = _load(key)
        if df is not None and not df.empty:
            val = float(df["Close"].iloc[-1])
            if val < 100:   # sanity check
                return val
    # Estimate from VIXY (VIXY ≈ VIX / 10 roughly)
    df = _load("macro_vix")
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    return None

def get_gold_df() -> Optional[pd.DataFrame]:
    return _load("macro_gold")

def get_macro_df(instrument: str) -> Optional[pd.DataFrame]:
    """
    Returns macro instrument daily DataFrame.
    Keys: 'tnx' | 'vix' | 'qqqe' | 'ndx' | 'oil' | 'gold' | 'spy'
    """
    # Try multiple key patterns
    for key in [f"macro_{instrument.lower()}",
                f"macro_{instrument.replace('^','').lower()}"]:
        df = _load(key)
        if df is not None:
            return df
    return None

def get_yield_10y() -> Optional[float]:
    df = get_macro_df("tnx")
    if df is not None and not df.empty:
        val = float(df["Close"].iloc[-1])
        return val if val < 15 else round(max(0, 10 - (val / 11)), 2)
    return None

def get_oil_price() -> Optional[float]:
    df = get_macro_df("oil")
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1])
    return None

def get_qqqe_df() -> Optional[pd.DataFrame]:
    return get_macro_df("qqqe")

def get_qqq_1d() -> Optional[pd.DataFrame]:
    return _load(f"df_1d_{NAS100_LABEL}")
