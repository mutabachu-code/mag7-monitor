import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm

# --- DASHBOARD SETUP ---
st.set_page_config(page_title="Mag 7 Live Monitor", layout="wide")
st.title("🛡️ Mag 7 Technical Monitor")
st.caption("Real-time Intraday Analysis | Premarket | Greeks Proxy")

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']

def get_bs_delta(S, K, T, r, sigma):
    """Calculates a theoretical Delta as a sentiment proxy."""
    if T <= 0 or sigma <= 0 or S <= 0: return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def clean_data(df):
    """Fills missing values to prevent the RuntimeWarnings seen in your logs."""
    return df.ffill().bfill().dropna()

# Create a refresh button
if st.button('🔄 Refresh Market Data'):
    st.rerun()

cols = st.columns(4) # Create a grid for the display

for i, ticker in enumerate(TICKERS):
    try:
        stock = yf.Ticker(ticker)
        # Using 2 days to ensure we have enough data for a stable RSI calculation
        df = stock.history(period="2d", interval="5m", prepost=True)
        df = clean_data(df)

        if len(df) < 15:
            st.warning(f"Not enough data for {ticker}")
            continue

        # --- TECHNICALS ---
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Current Metrics
        curr_p = df['Close'].iloc[-1]
        prev_p = df['Close'].iloc[-2]
        curr_rsi = rsi.iloc[-1]
        
        # Volume Analysis (Prevents the 'round' error from your screenshot)
        last_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].tail(10).mean()
        vol_ratio = round(last_vol / avg_vol, 2) if avg_vol > 0 else 1.0
        
        # Greeks Proxy (Annualized Vol)
        ann_vol = df['Close'].pct_change().std() * np.sqrt(252 * 78)
        delta_val = get_bs_delta(curr_p, curr_p, 1/252, 0.045, ann_vol)

        # --- UI DISPLAY ---
        with cols[i % 4]:
            price_diff = curr_p - prev_p
            st.metric(label=ticker, value=f"${curr_p:.2f}", delta=f"{price_diff:.2f}")
            
            # Color-coded Sentiment
            if curr_rsi > 70:
                st.error(f"RSI: {curr_rsi:.1f} (Overbought)")
            elif curr_rsi < 30:
                st.success(f"RSI: {curr_rsi:.1f} (Oversold)")
            else:
                st.info(f"RSI: {curr_rsi:.1f}")

            st.write(f"**Vol Ratio:** {vol_ratio}x")
            st.write(f"**Delta Proxy:** {delta_val:.2f}")
            st.divider()

    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
