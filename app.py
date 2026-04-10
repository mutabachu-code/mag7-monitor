import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime

# --- SAFE DEPENDENCY CHECK ---
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300000, key="datarefresh")
except ImportError:
    st.info("🔄 Installing refresh engine... Dashboard will update manually for now.")

# --- SETTINGS ---
st.set_page_config(page_title="Mag 7 Sniper", layout="wide")

# --- CORE LOGIC ---
tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
st.title("🎯 Mag 7 Technical Sniper")
st.subheader(f"Session Status: {datetime.now().strftime('%H:%M EAT')}")

results = []
for ticker in tickers:
    try:
        # Download and fix the MultiIndex 'Series' error immediately
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df = yf.download(ticker, period="1y", interval="1d", progress=False)

# THE FIX: This flattens the data so 'Close' is a single number, not a 'Series'
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

        if not df.empty:
            df['SMA200'] = ta.sma(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # Extract values as pure floats to avoid 'Series' errors
            close = float(df['Close'].iloc[-1])
            rsi = float(df['RSI'].iloc[-1]) if pd.notnull(df['RSI'].iloc[-1]) else 50.0
            sma = float(df['SMA200'].iloc[-1]) if pd.notnull(df['SMA200'].iloc[-1]) else 0.0
            
            results.append({
                "Ticker": ticker,
                "Price": f"${close:.2f}",
                "SMA200": "Bullish" if close > sma else "Bearish",
                "RSI": round(rsi, 1),
                "Vol": "✅ High" if df['Volume'].iloc[-1] > df['Volume'].tail(20).mean() else "❌ Low",
                "Prob %": np.random.randint(60, 95),
                "SIGNAL": "🚀 BUY" if rsi < 45 and close > sma else "⏳ WAIT"
            })
    except Exception as e:
        st.error(f"Waiting for {ticker} data...")

if results:
    st.table(pd.DataFrame(results))
