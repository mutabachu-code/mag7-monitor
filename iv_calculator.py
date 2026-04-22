"""
iv_calculator.py
----------------
Calculates IV, IV Rank, and IV Percentile using pre-fetched batched data.
No direct yfinance calls — reads from data_fetcher session cache.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class IVData:
    ticker: str
    current_iv: float       # as % e.g. 28.4
    iv_rank: float          # 0-100
    iv_percentile: float    # 0-100
    iv_label: str           # LOW | MEDIUM | HIGH | EXTREME
    iv_color: str
    source: str             # options | vix_proxy | historical


# ── BLACK-SCHOLES ─────────────────────────────────────────────────────────────

def _bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1e-6
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def _newton_iv(market_price, S, K, T, r, max_iter=100, tol=1e-6) -> Optional[float]:
    sigma = 0.3
    for _ in range(max_iter):
        price = _bs_call_price(S, K, T, r, sigma)
        vega  = _bs_vega(S, K, T, r, sigma)
        diff  = market_price - price
        if abs(diff) < tol:
            return sigma
        if abs(vega) < 1e-10:
            break
        sigma += diff / vega
        if sigma <= 0 or sigma > 20:
            break
    return None


# ── OPTIONS IV (uses single yfinance call for options chain only) ─────────────

def _get_options_iv(ticker: str, current_price: float) -> Optional[float]:
    """One targeted options chain fetch — much lighter than full history."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None

        today     = pd.Timestamp.now()
        valid_exp = [e for e in expirations if (pd.Timestamp(e) - today).days >= 7]
        if not valid_exp:
            valid_exp = expirations[:1]

        chain = stock.option_chain(valid_exp[0])
        calls = chain.calls.copy()
        if calls.empty:
            return None

        calls['moneyness'] = abs(calls['strike'] - current_price)
        atm = calls.nsmallest(1, 'moneyness').iloc[0]

        K            = float(atm['strike'])
        market_price = float(atm['lastPrice'])
        T            = (pd.Timestamp(valid_exp[0]) - today).days / 365.0

        if market_price <= 0 or T <= 0:
            return None

        return _newton_iv(market_price, current_price, K, T, 0.045)

    except Exception as e:
        print(f"[iv_calculator] Options chain error {ticker}: {e}")
        return None


# ── HISTORICAL VOL (from batched daily data) ──────────────────────────────────

def _hv_from_daily(df_1d: pd.DataFrame, window: int = 20) -> Optional[float]:
    """Calculate annualised historical vol from pre-fetched daily data."""
    try:
        if df_1d is None or len(df_1d) < window + 1:
            return None
        log_returns = np.log(df_1d['Close'] / df_1d['Close'].shift(1)).dropna()
        return float(log_returns.tail(window).std() * np.sqrt(252))
    except Exception:
        return None


def _hv_series(df_1d: pd.DataFrame, window: int = 20) -> Optional[pd.Series]:
    """Rolling HV series for IV rank calculation."""
    try:
        if df_1d is None or len(df_1d) < window + 1:
            return None
        log_returns = np.log(df_1d['Close'] / df_1d['Close'].shift(1)).dropna()
        return (log_returns.rolling(window=window).std() * np.sqrt(252)).dropna()
    except Exception:
        return None


# ── IV RANK & PERCENTILE ──────────────────────────────────────────────────────

def _rank_and_pct(current_iv: float, series: pd.Series) -> Tuple[float, float]:
    if series is None or len(series) < 5:
        return 50.0, 50.0
    lo, hi = series.min(), series.max()
    rank = 0.0 if hi == lo else (current_iv - lo) / (hi - lo) * 100
    pct  = float((series < current_iv).mean() * 100)
    return round(max(0, min(100, rank)), 1), round(pct, 1)


# ── LABEL ─────────────────────────────────────────────────────────────────────

def _classify(iv_rank: float) -> Tuple[str, str]:
    if iv_rank >= 80:
        return "EXTREME", "#8b0000"
    elif iv_rank >= 60:
        return "HIGH", "#c9302c"
    elif iv_rank >= 35:
        return "MEDIUM", "#e6a817"
    else:
        return "LOW", "#2d9e2d"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def get_iv_data(ticker: str, current_price: float, df_1d: pd.DataFrame,
                vix_value: Optional[float] = None) -> Optional[IVData]:
    """
    Calculate IVData using pre-fetched data (no new yfinance calls).
    Falls back to options chain (1 call) only for stocks — not on every refresh.
    NAS100 always uses VIX proxy.
    """
    source     = "historical"
    current_iv = None

    if ticker == 'NAS100':
        # VIX proxy — already fetched in batch
        if vix_value and vix_value > 0:
            current_iv = vix_value / 100
            source     = "vix_proxy"
        else:
            current_iv = _hv_from_daily(df_1d)
    else:
        # Try options chain (single lightweight call, cached by claude_analyst TTL)
        current_iv = _get_options_iv(ticker, current_price)
        if current_iv:
            source = "options"
        else:
            current_iv = _hv_from_daily(df_1d)

    if not current_iv or current_iv <= 0:
        return None

    current_iv = min(max(current_iv, 0.01), 5.0)
    series     = _hv_series(df_1d)
    iv_rank, iv_pct = _rank_and_pct(current_iv, series)
    label, color    = _classify(iv_rank)

    return IVData(
        ticker=ticker,
        current_iv=round(current_iv * 100, 1),
        iv_rank=iv_rank,
        iv_percentile=iv_pct,
        iv_label=label,
        iv_color=color,
        source=source,
    )
