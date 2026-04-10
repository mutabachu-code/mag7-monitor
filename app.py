import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime

# --- SETTINGS & THEME ---
st.set_page_config(page_title="Mag 7 Sniper Dashboard", layout="wide")
st.markdown("""
    <style>
    .metric-card { background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .stMetric { color: white !important; }
    </style>
""", unsafe_allow_html=True)

# --- ANALYTICAL FUNCTIONS ---

def get_pivot_levels(df):
    """Calculates classic Pivot Points for Entry Floor and Ceiling"""
    last_day = df.iloc[-1]
    high = float(last_day['High'])
    low = float(last_day['Low'])
    close = float(last_day['Close'])
    
    pivot = (high + low + close) / 3
    support = (pivot * 2) - high
    resistance = (pivot * 2) - low
    return round(support, 2), round(resistance, 2)

def generate_consensus_signal(rsi, trend, vol, prob):
    """Multi-factor logic for the 'Final Signal'"""
    score = 0
    if trend == "Bullish": score += 30
    if prob >= 70: score += 30
    if rsi < 45: score += 20  
    if vol == "✅ High": score += 20
    
    if score >= 80: return "🚀 STRONG BUY"
    if score >= 60: return "✅ BUY"
    if score <= 30: return "📉 STRONG SELL"
    if score <= 50: return "⚠️ AVOID"
    return "⏳ WAIT"

def calculate_index_score(results_list):
    """Calculates a weighted score based on Mag 7 Market Cap"""
    weights = {"AAPL": 0.22, "MSFT": 0.22, "NVDA": 0.20, "GOOGL": 0.12, "AMZN": 0.12, "META": 0.08, "TSLA": 0.04}
    total_score = 0
    for r in results_list:
        ticker = r["Ticker"]
        sig = r["FINAL SIGNAL"]
        points = 100 if "BUY" in sig else (0 if "SELL" in sig else 50)
        total_score += points * weights.get(ticker, 0.10)
    return round(total_score, 1)

# --- CORE LOGIC ---

tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]

st.title("🎯 Mag 7 Technical Sniper & Index Monitor")
st.subheader(f"Market Status as of {datetime.now().strftime('%H:%M EAT')}")

results = []
progress = st.progress(0)

for i, ticker in enumerate(tickers):
    try:
        # Download data (Auto-adjusts for single ticker)
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        if not data.empty:
            # Technicals using pandas_ta
            data['SMA200'] = ta.sma(data['Close'], length=200)
            data['RSI'] = ta.rsi(data['Close'], length=14)
            
            # FIXED EXTRACTION: Using .iloc[-1] and float() conversion
            close = float(data['Close'].iloc[-1])
            sma200 = float(data['SMA200'].iloc[-1])
            rsi = float(data['RSI'].iloc[-1])
            
            # Volume Analysis
            avg_vol = data['Volume'].tail(20).mean()
            curr_vol = float(data['Volume'].iloc[-1])
            vol_status = "✅ High" if curr_vol > avg_vol else "❌ Low"
            
            # Trend & Levels
            trend = "Bullish" if close > sma200 else "Bearish"
            support, resistance = get_pivot_levels(data)
            
            # Probability Mock (Replace with your custom Logic if needed)
            prob = np.random.randint(40, 95) 
            
            # Final Consolidated Signal
            final_sig = generate_consensus_signal(rsi, trend, vol_status, prob)
            
            results.append({
                "Ticker": ticker,
                "Price": f"${close:.2f}",
                "SMA200": trend,
                "RSI": round(rsi, 1),
                "Vol": vol_status,
                "Prob %": prob,
                "Entry Floor": f"${support}",
                "Ceiling": f"${resistance}",
                "FINAL SIGNAL": final_sig
            })
        
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
    
    progress.progress((i + 1) / len(tickers))

# --- DASHBOARD DISPLAY ---

if results:
    index_val = calculate_index_score(results)
    col1, col2 = st.columns([1, 3])

    with col1:
        st.metric(
            label="Mag 7 Index Score", 
            value=f"{index_val}/100", 
            delta=f"{'BULLISH' if index_val > 50 else 'BEARISH'}"
        )

    with col2:
        if index_val >= 70:
            st.success("🔥 NASDAQ MOMENTUM: Institutional buying confirmed. Look
