import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime

# --- SETTINGS ---
st.set_page_config(page_title="Mag 7 Sniper", layout="wide")

# --- CORE FUNCTIONS ---

def get_pivot_levels(df):
    """Calculates Daily Pivot Points safely"""
    try:
        last_day = df.iloc[-1]
        high = float(last_day['High'])
        low = float(last_day['Low'])
        close = float(last_day['Close'])
        pivot = (high + low + close) / 3
        support = (pivot * 2) - high
        resistance = (pivot * 2) - low
        return round(support, 2), round(resistance, 2)
    except:
        return 0.0, 0.0

# --- MAIN LOGIC ---

tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
st.title("🎯 Mag 7 Technical Sniper")
st.subheader(f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M EAT')}")

results = []
progress = st.progress(0)

for i, ticker in enumerate(tickers):
    try:
        # 1. Download & Flatten headers immediately
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        # This line kills the 'Series' error by removing multi-level headers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if not df.empty and len(df) > 200:
            # 2. Calculate Indicators
            df['SMA200'] = ta.sma(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # 3. SAFE EXTRACTION: Check if value is None before converting
            raw_close = df['Close'].iloc[-1]
            raw_sma = df['SMA200'].iloc[-1]
            raw_rsi = df['RSI'].iloc[-1]
            
            if pd.notnull(raw_close) and pd.notnull(raw_sma):
                close = float(raw_close)
                sma200 = float(raw_sma)
                rsi = float(raw_rsi) if pd.notnull(raw_rsi) else 50.0
                
                # 4. Signal Logic
                trend = "Bullish" if close > sma200 else "Bearish"
                support, resistance = get_pivot_levels(df)
                
                results.append({
                    "Ticker": ticker,
                    "Price": f"${close:.2f}",
                    "SMA200": trend,
                    "RSI": round(rsi, 1),
                    "Entry Floor": f"${support}",
                    "Ceiling": f"${resistance}",
                    "SIGNAL": "🚀 BUY" if rsi < 40 and trend == "Bullish" else "⏳ WAIT"
                })
        else:
            st.warning(f"Insufficient data for {ticker}")

    except Exception as e:
        st.error(f"Failed {ticker}: {str(e)}")
    
    progress.progress((i + 1) / len(tickers))

# --- DISPLAY ---
if results:
    st.table(pd.DataFrame(results))
