import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

# --- DASHBOARD SETUP ---
st.set_page_config(page_title="Mag 7 Live MTF Monitor", layout="wide")

# AUTO REFRESH: Every 60 seconds
st_autorefresh(interval=60000, key="datarefresh")

st.title("🛡️ Mag 7 MTF Technical Monitor")
st.caption(f"Last Update: {pd.Timestamp.now().strftime('%H:%M:%S')} | Confluence: 5m Signals + 1H Trend")

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']

def get_bs_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0: return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

cols = st.columns(4)

for i, ticker in enumerate(TICKERS):
    try:
        # 1. FETCH DATA (Multi-Timeframe)
        stock = yf.Ticker(ticker)
        
        # 5-minute data for RSI and current price
        df_5m = stock.history(period="2d", interval="5m", prepost=True).ffill().bfill()
        
        # 1-hour data for SMA 200 Trend
        df_1h = stock.history(period="1mo", interval="1h").ffill().bfill()

        if len(df_5m) < 20 or len(df_1h) < 200:
            with cols[i % 4]:
                st.warning(f"{ticker}: Not enough data for SMA 200")
            continue

        # --- CALCULATIONS ---
        
        # A. SMA 200 Trend (1H Timeframe)
        sma200_1h = df_1h['Close'].rolling(window=200).mean().iloc[-1]
        curr_p = df_5m['Close'].iloc[-1]
        trend_status = "BULLISH" if curr_p > sma200_1h else "BEARISH"
        trend_color = "green" if trend_status == "BULLISH" else "red"

        # B. RSI (5m Timeframe)
        delta_p = df_5m['Close'].diff()
        gain = (delta_p.where(delta_p > 0, 0)).rolling(window=14).mean()
        loss = (-delta_p.where(delta_p < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # C. MTF Volume Analysis (Comparing 5m vol to 1H avg vol)
        last_vol = df_5m['Volume'].iloc[-1]
        avg_vol_1h = df_1h['Volume'].tail(20).mean() # Avg volume of last 20 hours
        vol_ratio = last_vol / (avg_vol_1h / 12) # Normalize 1h vol to a 5m segment
        
        # D. The Delta Fix (Using a fixed Strike from start of day to see directional drift)
        start_price = df_5m['Open'].iloc[0]
        ann_vol = df_5m['Close'].pct_change().std() * np.sqrt(252 * 78)
        # Using Start of Day Price as 'K' helps Delta move as price drifts away from open
        delta_val = get_bs_delta(curr_p, start_price, 1/252, 0.045, ann_vol)

        # --- UPGRADED SIGNAL LOGIC ---
        # 1. Check SMA 200 Confluence first
        # 2. Check Volume Confirmation
        # 3. Trigger on RSI
        
        if curr_p > sma200_1h and rsi < 40 and vol_ratio > 1.1:
            signal = "🟢 STRONG BUY (Trend + Oversold)"
            sig_color = "green"
        elif curr_p < sma200_1h and rsi > 60 and vol_ratio > 1.1:
            signal = "🔴 STRONG SELL (Trend + Overbought)"
            sig_color = "red"
        elif rsi < 30:
            signal = "🟡 Scalp Buy (Counter-Trend)"
            sig_color = "blue"
        else:
            signal = "⚪ Neutral"
            sig_color = "gray"

        # --- UI DISPLAY ---
        with cols[i % 4]:
            st.metric(label=ticker, value=f"${curr_p:.2f}", 
                      delta=f"{curr_p - df_5m['Close'].iloc[-2]:.2f}")
            
            # Trend Indicator
            st.markdown(f"**Trend (1H SMA200):** :{trend_color}[{trend_status}]")
            
            # Signal Box
            st.markdown(f"**Action:** :{sig_color}[{signal}]")
            
            # Detailed Stats
            with st.expander("Technical Details"):
                st.write(f"RSI (5m): {rsi:.1f}")
                st.write(f"Vol Score: {vol_ratio:.2f}x")
                st.write(f"Opt. Delta: {delta_val:.2f}")
                st.write(f"SMA 200: ${sma200_1h:.2f}")
            st.divider()

    except Exception as e:
        st.error(f"Error {ticker}: {e}")
