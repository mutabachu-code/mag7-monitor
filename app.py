import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Mag 7 Advanced Risk Monitor", layout="wide")
st_autorefresh(interval=60000, key="datarefresh")

st.title("🛡️ Mag 7 Risk & Confluence Monitor")
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']

# --- CALCULATIONS ENGINE ---

def get_greeks(S, K, T, r, sigma):
    """Calculates Delta, Gamma, and Vega using Black-Scholes"""
    if T <= 0 or sigma <= 0 or S <= 0: return 0.5, 0.0, 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return delta, gamma, vega

# Fetch Market Data (SPY) once for Beta calculation
market_data = yf.download("SPY", period="60d", interval="1h", progress=False)['Close']
market_returns = market_data.pct_change().dropna()

cols = st.columns(4)

for i, ticker in enumerate(TICKERS):
    try:
# --- BULLETPROOF DATA FETCH ---
        stock = yf.Ticker(ticker)
        raw_df_5m = stock.history(period="5d", interval="5m")
        raw_df_1h = stock.history(period="60d", interval="1h")
        
        # Force flatten the multi-index if it exists
        df_5m = raw_df_5m.copy()
        df_1h = raw_df_1h.copy()
        if isinstance(df_5m.columns, pd.MultiIndex):
            df_5m.columns = df_5m.columns.get_level_values(0)
        if isinstance(df_1h.columns, pd.MultiIndex):
            df_1h.columns = df_1h.columns.get_level_values(0)

        df_5m = df_5m.ffill().bfill()
        df_1h = df_1h.ffill().bfill()

        # 1. MACD Calculation
        close_5m = df_5m['Close'].astype(float)
        exp1 = close_5m.ewm(span=12, adjust=False).mean()
        exp2 = close_5m.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Extract SCALARS (the actual fix for the float error)
        current_macd = float(macd_line.iloc[-1])
        current_signal = float(signal_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        prev_signal = float(signal_line.iloc[-2])

        # 2. Beta (vs SPY)
        close_1h = df_1h['Close'].astype(float)
        stock_returns = close_1h.pct_change().dropna()
        combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
        covariance = float(combined.cov().iloc[0, 1])
        market_variance = float(market_returns.var())
        beta_val = covariance / market_variance if market_variance > 0 else 1.0

        # 3. SMA 200 & Price
        curr_p = float(close_5m.iloc[-1])
        sma200 = float(close_1h.rolling(window=200).mean().iloc[-1])
        
        # 4. Volatility & Greeks
        ann_vol = float(close_5m.pct_change().std() * np.sqrt(252 * 78))
        open_p = float(df_5m['Open'].iloc[0])
        delta, gamma, vega = get_greeks(curr_p, open_p, 1/252, 0.045, ann_vol)

        # --- CONFLUENCE LOGIC ---
        trend_up = curr_p > sma200
        macd_cross_up = bool((current_macd > current_signal) and (prev_macd <= prev_signal))
        macd_cross_down = bool((current_macd < current_signal) and (prev_macd >= prev_signal))
        # --- UI ---
        with cols[i % 4]:
            st.metric(label=ticker, value=f"${curr_p:.2f}", delta=f"Beta: {beta_val:.2f}")
            st.markdown(f"**Action:** :{sig_color}[{signal}]")
            
            with st.expander("Greeks & Risk"):
                st.write(f"Gamma (Accel): {gamma:.4f}")
                st.write(f"Vega (IV Sens): {vega:.2f}")
                st.write(f"MACD Hist: {macd_hist.iloc[-1]:.2f}")
            st.divider()

    except Exception as e:
        st.error(f"Error {ticker}: {e}")
