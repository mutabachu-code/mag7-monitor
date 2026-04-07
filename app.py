import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import time

# --- APP CONFIGURATION ---
st.set_page_config(page_title="Mag 7 Live Monitor", layout="wide")

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_data(symbol):
    df_d = yf.download(symbol, period="1y", interval="1d", progress=False)
    df_m15 = yf.download(symbol, period="5d", interval="15m", progress=False)
    if df_d.empty or df_m15.empty: return None
    
    # Flatten Multi-index columns
    if isinstance(df_d.columns, pd.MultiIndex): df_d.columns = df_d.columns.get_level_values(0)
    if isinstance(df_m15.columns, pd.MultiIndex): df_m15.columns = df_m15.columns.get_level_values(0)
    
    return df_d, df_m15

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Strategy Settings")
rsi_threshold = st.sidebar.slider("RSI Entry Level", 20, 45, 35)
refresh_rate = st.sidebar.selectbox("Refresh Every", [1, 5, 10, 15], index=1)

# --- MAIN UI ---
st.title("🚀 Mag 7 Live Technical Monitor")
st.write(f"Last Updated: {datetime.now().strftime('%H:%M:%S')} (EAT)")

tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL"]
results = []

# --- THE ENGINE ---
with st.spinner("Scanning Markets..."):
    for ticker in tickers:
        data = get_data(ticker)
        if data:
            df_d, df_m15 = data
            
            # Indicators
            close = float(df_d['Close'].iloc[-1])
            sma200 = df_d['Close'].rolling(window=200).mean().iloc[-1]
            rsi = calculate_rsi(df_m15['Close']).iloc[-1]
            avg_vol = df_d['Volume'].rolling(window=10).mean().iloc[-1]
            curr_vol = df_d['Volume'].iloc[-1]
            
            # Logic
            trend = "Bullish" if close > sma200 else "Bearish"
            vol_status = "✅ High" if curr_vol > avg_vol else "❌ Low"
            
            status = "⏳ WAIT"
            if trend == "Bullish" and rsi < rsi_threshold and curr_vol > avg_vol:
                status = "🔥 BUY"
            elif trend == "Bearish":
                status = "⚠️ AVOID"
            elif rsi > 70:
                status = "✂️ TRIM"

            # Logic for Take Profit (2% Gain)
            target_price = f"${close * 1.02:.2f}" if status == "🔥 BUY" else "-"

            results.append({
    "Ticker": ticker,
    "Price": f"${close:.2f}",
    "Trend": trend,
    "RSI (15m)": f"{rsi:.1f}",  # Rounds to 1 decimal place
    "Volume": vol_status,
    "Signal": status,
    "Target (+2%)": target_price if target_price != "-" else "---"
})


# --- DISPLAY TABLE ---
df_display = pd.DataFrame(results)

def color_signal(val):
    color = 'white'
    if val == '🔥 BUY': color = '#2ecc71' # Green
    elif val == '⚠️ AVOID': color = '#e74c3c' # Red
    elif val == '⏳ WAIT': color = '#f1c40f' # Yellow
    return f'background-color: {color}; color: black; font-weight: bold'

st.table(df_display.style.map(color_signal, subset=['Signal']))

# --- AUTO-REFRESH SCRIPT ---
# This reloads the page automatically based on your setting
st.empty()
time.sleep(refresh_rate * 60)
st.rerun()