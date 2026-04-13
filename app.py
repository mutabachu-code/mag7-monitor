import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

# --- DASHBOARD SETUP ---
st.set_page_config(page_title="Mag 7 Live Monitor", layout="wide")

# AUTO REFRESH: Every 60 seconds (60000ms)
st_autorefresh(interval=60000, key="datarefresh")

st.title("🛡️ Mag 7 Technical Monitor")
st.caption(f"Last Update: {pd.Timestamp.now().strftime('%H:%M:%S')} (Auto-refreshes every 1 min)")

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']

def get_bs_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

cols = st.columns(4)

for i, ticker in enumerate(TICKERS):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period="2d", interval="5m", prepost=True).ffill().bfill()

        if len(df) < 15: continue

        # --- CALCULATIONS ---
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        curr_p = df['Close'].iloc[-1]
        prev_p = df['Close'].iloc[-2]
        
        # Volume Analysis
        last_vol = df['Volume'].iloc[-1]
        avg_vol = df['Volume'].tail(15).mean()
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0
        vol_status = "High 📈" if vol_ratio > 1.5 else "Low 📉" if vol_ratio < 0.7 else "Normal"
        
        # Greeks
        ann_vol = df['Close'].pct_change().std() * np.sqrt(252 * 78)
        delta_val = get_bs_delta(curr_p, curr_p, 1/252, 0.045, ann_vol)

        # --- SIGNAL LOGIC ---
        if rsi < 35 and vol_ratio > 1.2:
            signal = "🟢 BUY (Oversold + Vol)"
            color = "green"
        elif rsi > 65 and vol_ratio > 1.2:
            signal = "🔴 SELL (Overbought + Vol)"
            color = "red"
        else:
            signal = "🟡 WAIT (Neutral)"
            color = "gray"

        # --- UI DISPLAY ---
        with cols[i % 4]:
            st.metric(label=ticker, value=f"${curr_p:.2f}", delta=f"{curr_p-prev_p:.2f}")
            
            # Signal Box
            st.markdown(f"**Signal:** :{color}[{signal}]")
            
            # Details
            st.write(f"**RSI:** {rsi:.1f}")
            st.write(f"**Volume:** {vol_status} ({vol_ratio:.1f}x)")
            st.write(f"**Delta:** {delta_val:.2f}")
            st.divider()

    except Exception as e:
        st.error(f"Error {ticker}: {e}")
