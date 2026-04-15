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
# --- CALCULATIONS ENGINE (INDESTRUCTIBLE VERSION) ---
        stock = yf.Ticker(ticker)
        df_5m = stock.history(period="5d", interval="5m").ffill().bfill()
        df_1h = stock.history(period="60d", interval="1h").ffill().bfill()
        
        # 1. MACD Calculation
        exp1 = df_5m['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df_5m['Close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Pulling raw values using .values.flatten() to ensure we get a single float
        current_macd = float(macd_line.values.flatten()[-1])
        current_signal = float(signal_line.values.flatten()[-1])
        prev_macd = float(macd_line.values.flatten()[-2])
        prev_signal = float(signal_line.values.flatten()[-2])

        # 2. Beta (vs SPY)
        stock_returns = df_1h['Close'].pct_change().dropna()
        combined = pd.concat([stock_returns, market_returns], axis=1).dropna()
        # Ensure covariance is a scalar
        cov_matrix = combined.cov()
        covariance = float(cov_matrix.values[0, 1])
        market_variance = float(market_returns.var())
        beta_val = covariance / market_variance if market_variance > 0 else 1.0

        # 3. SMA 200 & Price
        curr_p = float(df_5m['Close'].values.flatten()[-1])
        sma_series = df_1h['Close'].rolling(window=200).mean()
        sma200 = float(sma_series.values.flatten()[-1])
        
        # 4. Volatility & Greeks
        returns = df_5m['Close'].pct_change().dropna()
        ann_vol = float(returns.std() * np.sqrt(252 * 78))
        open_p = float(df_5m['Open'].values.flatten()[0])
        delta, gamma, vega = get_greeks(curr_p, open_p, 1/252, 0.045, ann_vol)

        # --- CONFLUENCE LOGIC ---
        trend_up = curr_p > sma200
        macd_cross_up = (current_macd > current_signal) and (prev_macd <= prev_signal)
        macd_cross_down = (current_macd < current_signal) and (prev_macd >= prev_signal)
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
