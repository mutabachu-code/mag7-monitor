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
st.caption(f"Last Update: {pd.Timestamp.now().strftime('%H:%M:%S')} | Confluence: 5m Signals + 1H MACD Trend")

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
        df_5m = stock.history(period="5d", interval="5m", prepost=True).ffill().bfill()
        df_1h = stock.history(period="60d", interval="1h").ffill().bfill()

        if len(df_1h) < 200: continue

        # --- CALCULATIONS ---
        
        # A. SMA 200 Trend (1H Timeframe)
        sma200_1h = df_1h['Close'].rolling(window=200).mean().iloc[-1]
        curr_p = df_5m['Close'].iloc[-1]
        trend_status = "BULLISH" if curr_p > sma200_1h else "BEARISH"
        trend_color = "green" if trend_status == "BULLISH" else "red"

        # B. NEW: MACD Confluence (1H Timeframe)
        ema12 = df_1h['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_1h['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

        # C. RSI (5m Timeframe)
        delta_p = df_5m['Close'].diff()
        gain = (delta_p.where(delta_p > 0, 0)).rolling(window=14).mean()
        loss = (-delta_p.where(delta_p < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # D. MOVING AVERAGE VOLUME ANALYSIS
        df_5m['vol_ma_long'] = df_5m['Volume'].rolling(window=20).mean()
        df_5m['vol_ma_short'] = df_5m['Volume'].rolling(window=5).mean()
        vol_ratio = df_5m['vol_ma_short'].iloc[-1] / df_5m['vol_ma_long'].iloc[-1] if df_5m['vol_ma_long'].iloc[-1] > 0 else 1.0
        
        # E. Delta Calculation
        start_price = df_5m['Open'].iloc[0]
        ann_vol = df_5m['Close'].pct_change().std() * np.sqrt(252 * 78)
        delta_val = get_bs_delta(curr_p, start_price, 1/252, 0.045, ann_vol)

        # --- REFINED SIGNAL LOGIC ---
        # STRONG signals now require MACD confirmation
        if curr_p > sma200_1h and rsi < 40 and vol_ratio > 1.1 and macd_bullish:
            signal = "🟢 STRONG BUY (Full Confluence)"
            sig_color = "green"
        elif curr_p < sma200_1h and rsi > 60 and vol_ratio > 1.1 and not macd_bullish:
            signal = "🔴 STRONG SELL (Full Confluence)"
            sig_color = "red"
        elif curr_p > sma200_1h and rsi < 35:
            signal = "🟡 Caution Buy (No MACD Confirm)"
            sig_color = "orange"
        else:
            signal = "⚪ Neutral"
            sig_color = "gray"

        # --- UI DISPLAY ---
        with cols[i % 4]:
            st.metric(label=ticker, value=f"${curr_p:.2f}", 
                      delta=f"{curr_p - df_5m['Close'].iloc[-2]:.2f}")
            
            st.markdown(f"**Trend (1H SMA200):** :{trend_color}[{trend_status}]")
            st.markdown(f"**Action:** :{sig_color}[{signal}]")
            
            with st.expander("Technical Details"):
                st.write(f"RSI (5m): {rsi:.1f}")
                st.write(f"Vol Surge: {vol_ratio:.2f}x")
                st.write(f"Opt. Delta: {delta_val:.2f}")
                st.write(f"MACD Status: {'Bullish 📈' if macd_bullish else 'Bearish 📉'}")
            st.divider()

    except Exception as e:
        st.error(f"Error {ticker}: {e}")
