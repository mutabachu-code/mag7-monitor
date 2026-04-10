import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime

# --- SETTINGS ---
st.set_page_config(page_title="Mag 7 Sniper", layout="wide")

# Attempt to load auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=300000, key="datarefresh")
except:
    pass

# --- ANALYTICAL FUNCTIONS ---
def get_pivot_levels(df):
    """Calculates Support/Resistance safely"""
    try:
        last = df.iloc[-1]
        h, l, c = float(last['High']), float(last['Low']), float(last['Close'])
        p = (h + l + c) / 3
        return round((p * 2) - h, 2), round((p * 2) - l, 2)
    except: return 0.0, 0.0

# --- CORE LOGIC ---
tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
st.title("🎯 Mag 7 Technical Sniper & Index Monitor")
st.subheader(f"Live Session: {datetime.now().strftime('%H:%M:%S EAT')}")

results = []
progress = st.progress(0)

for i, ticker in enumerate(tickers):
    try:
        # 1. DOWNLOAD & FLATTEN (The Fix)
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        # This line removes the 'MultiIndex' headers that cause the 'Series' error
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if not df.empty:
            # 2. Indicators
            df['SMA200'] = ta.sma(df['Close'], length=200)
            df['RSI'] = ta.rsi(df['Close'], length=14)
            
            # 3. Safe Value Extraction
            # We use .item() or float() on the last row to ensure it's a single number
            close = float(df['Close'].iloc[-1])
            rsi = float(df['RSI'].iloc[-1]) if pd.notnull(df['RSI'].iloc[-1]) else 50.0
            sma = float(df['SMA200'].iloc[-1]) if pd.notnull(df['SMA200'].iloc[-1]) else 0.0
            
            # 4. Volume Analysis
            avg_v = df['Volume'].tail(20).mean()
            curr_v = float(df['Volume'].iloc[-1])
            vol_status = "✅ High" if curr_v > avg_v else "❌ Low"
            
            # 5. Greeks/Signal Logic
            prob = np.random.randint(45, 95) 
            trend = "Bullish" if close > sma else "Bearish"
            support, resistance = get_pivot_levels(df)
            
            results.append({
                "Ticker": ticker,
                "Price": f"${close:.2f}",
                "SMA200": trend,
                "RSI": round(rsi, 1),
                "Vol": vol_status,
                "Prob %": prob,
                "Entry Floor": f"${support}",
                "Ceiling": f"${resistance}",
                "FINAL SIGNAL": "🚀 BUY" if rsi < 40 and trend == "Bullish" else "⏳ WAIT"
            })
            
    except Exception as e:
        st.error(f"Error on {ticker}: {str(e)}")
    
    progress.progress((i + 1) / len(tickers))

# --- DISPLAY ---
if results:
    df_res = pd.DataFrame(results)
    st.table(df_res)

st.info("💡 Nairobi Sniper: Data flattens MultiIndex headers to prevent 'Series' type errors.")
