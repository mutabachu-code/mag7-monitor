"""
iv_calculator.py
----------------
Calculates Implied Volatility (IV), IV Rank, and IV Percentile
for Mag 7 stocks and NAS100 (via VIX proxy).

IV Method: Newton-Raphson on Black-Scholes for nearest ATM call option.
IV Rank  : (Current IV - 52w Low) / (52w High - 52w Low) * 100
IV Pct   : % of past 252 trading days where IV was below current IV
"""

import numpy as np
import yfinance as yf
import pandas as pd
from scipy.stats import norm
from typing import Optional
from dataclasses import dataclass
import time


@dataclass
class IVData:
    ticker: str
    current_iv: float        # current IV as decimal e.g. 0.32 = 32%
    iv_rank: float           # 0-100 scale
    iv_percentile: float     # 0-100 scale
    iv_label: str            # "LOW" | "MEDIUM" | "HIGH" | "EXTREME"
    iv_color: str            # "green" | "orange" | "red" | "darkred"
    source: str              # "options" | "vix_proxy" | "historical"


# ── BLACK-SCHOLES HELPERS ─────────────────────────────────────────────────────

def _bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega (sensitivity of price to volatility)."""
    if T <= 0 or sigma <= 0:
        return 1e-6
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def _newton_raphson_iv(market_price, S, K, T, r,
                        max_iter=100, tol=1e-6) -> Optional[float]:
    """
    Newton-Raphson solver for implied volatility.
    Returns IV as decimal, or None if no convergence.
    """
    sigma = 0.3  # initial guess 30%
    for _ in range(max_iter):
        price = _bs_call_price(S, K, T, r, sigma)
        vega  = _bs_vega(S, K, T, r, sigma)
        diff  = market_price - price
        if abs(diff) < tol:
            return sigma
        if abs(vega) < 1e-10:
            break
        sigma = sigma + diff / vega
        if sigma <= 0 or sigma > 20:  # bounds check
            break
    return None


# ── IV FROM OPTIONS CHAIN ─────────────────────────────────────────────────────

def _get_iv_from_options(ticker: str, risk_free_rate: float = 0.045) -> Optional[float]:
    """
    Fetch nearest ATM call option and solve for IV via Newton-Raphson.
    Returns IV as decimal e.g. 0.28 for 28%, or None on failure.
    """
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None

        # Pick the nearest expiration that's at least 7 days out
        today = pd.Timestamp.now()
        valid_exp = [
            e for e in expirations
            if (pd.Timestamp(e) - today).days >= 7
        ]
        if not valid_exp:
            valid_exp = expirations[:1]

        expiry = valid_exp[0]
        chain  = stock.option_chain(expiry)
        calls  = chain.calls

        if calls.empty:
            return None

        # Current stock price
        hist = stock.history(period="1d", interval="1m")
        if hist.empty:
            return None
        S = hist['Close'].iloc[-1]

        # Find nearest ATM call (strike closest to current price)
        calls = calls.copy()
        calls['moneyness'] = abs(calls['strike'] - S)
        atm_call = calls.nsmallest(1, 'moneyness').iloc[0]

        K            = float(atm_call['strike'])
        market_price = float(atm_call['lastPrice'])
        T            = (pd.Timestamp(expiry) - today).days / 365.0

        if market_price <= 0 or T <= 0:
            return None

        iv = _newton_raphson_iv(market_price, S, K, T, risk_free_rate)
        return iv

    except Exception as e:
        print(f"[iv_calculator] Options IV error for {ticker}: {e}")
        return None


# ── HISTORICAL VOLATILITY FALLBACK ────────────────────────────────────────────

def _get_historical_vol(ticker: str, window: int = 20) -> Optional[float]:
    """
    Calculate realised historical volatility as IV proxy.
    Uses 20-day annualised standard deviation of log returns.
    """
    try:
        yfticker = '^NDX' if ticker == 'NAS100' else ticker
        df = yf.Ticker(yfticker).history(period="60d", interval="1d")
        if len(df) < window + 1:
            return None
        log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        hv = log_returns.tail(window).std() * np.sqrt(252)
        return float(hv)
    except Exception as e:
        print(f"[iv_calculator] HV error for {ticker}: {e}")
        return None


# ── IV RANK & PERCENTILE ──────────────────────────────────────────────────────

def _get_iv_rank_and_percentile(ticker: str, current_iv: float):
    """
    Calculate IV Rank and IV Percentile from 252-day historical volatility series.
    Returns (iv_rank, iv_percentile) both as 0-100 floats.
    """
    try:
        yfticker = '^NDX' if ticker == 'NAS100' else ticker
        df = yf.Ticker(yfticker).history(period="365d", interval="1d")
        if len(df) < 30:
            return 50.0, 50.0

        # Build rolling 20-day HV series as IV proxy for rank calculation
        log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        hv_series   = log_returns.rolling(window=20).std() * np.sqrt(252)
        hv_series   = hv_series.dropna()

        if len(hv_series) < 10:
            return 50.0, 50.0

        iv_52w_high = hv_series.max()
        iv_52w_low  = hv_series.min()

        # IV Rank: where current IV sits in 52w range
        if iv_52w_high == iv_52w_low:
            iv_rank = 50.0
        else:
            iv_rank = (current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low) * 100

        # IV Percentile: % of days IV was below current
        iv_percentile = float((hv_series < current_iv).mean() * 100)

        return round(iv_rank, 1), round(iv_percentile, 1)

    except Exception as e:
        print(f"[iv_calculator] IV rank error for {ticker}: {e}")
        return 50.0, 50.0


# ── LABEL & COLOUR ────────────────────────────────────────────────────────────

def _classify_iv(iv_rank: float):
    """Return (label, color) based on IV rank."""
    if iv_rank >= 80:
        return "EXTREME", "#8b0000"
    elif iv_rank >= 60:
        return "HIGH", "#c9302c"
    elif iv_rank >= 35:
        return "MEDIUM", "#e6a817"
    else:
        return "LOW", "#2d9e2d"


# ── VIX PROXY FOR NAS100 ──────────────────────────────────────────────────────

def _get_vix_iv() -> Optional[float]:
    """Fetch VIX as IV proxy for NAS100."""
    try:
        vix = yf.Ticker('^VIX').history(period="2d", interval="1d")
        if vix.empty:
            return None
        return float(vix['Close'].iloc[-1]) / 100  # VIX is in %, convert to decimal
    except Exception as e:
        print(f"[iv_calculator] VIX error: {e}")
        return None


# ── MAIN PUBLIC FUNCTION ──────────────────────────────────────────────────────

def get_iv_data(ticker: str) -> Optional[IVData]:
    """
    Main entry point. Returns IVData for a ticker.
    Tries options chain first, falls back to historical volatility.
    NAS100 uses VIX as proxy.
    """
    source = "options"

    if ticker == 'NAS100':
        # NAS100: use VIX as proxy
        current_iv = _get_vix_iv()
        source = "vix_proxy"
        if current_iv is None:
            current_iv = _get_historical_vol(ticker)
            source = "historical"
    else:
        # Mag 7: try live options chain first
        current_iv = _get_iv_from_options(ticker)
        if current_iv is None:
            current_iv = _get_historical_vol(ticker)
            source = "historical"

    if current_iv is None or current_iv <= 0:
        return None

    # Clamp to reasonable range
    current_iv = min(max(current_iv, 0.01), 5.0)

    iv_rank, iv_percentile = _get_iv_rank_and_percentile(ticker, current_iv)
    iv_label, iv_color     = _classify_iv(iv_rank)

    return IVData(
        ticker=ticker,
        current_iv=round(current_iv * 100, 1),   # store as % e.g. 28.4
        iv_rank=iv_rank,
        iv_percentile=iv_percentile,
        iv_label=iv_label,
        iv_color=iv_color,
        source=source,
    )
