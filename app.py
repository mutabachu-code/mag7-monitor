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
    high = last_day['High']
    low = last_day['Low']
    close = last_day['Close']
    
    pivot = (high + low + close) / 3
    support = (pivot * 2) - high
    resistance = (pivot * 2) - low
    return round(support, 2), round(resistance, 2)

def generate_consensus_signal(rsi, trend, vol, prob):
    """Multi-factor logic for the 'Final Signal'"""
    # Weighting: Trend (30%), Greeks (30%), RSI (20%), Volume (20%)
    score = 0
    if trend == "Bullish": score += 30
    if prob >= 70: score += 30
    if rsi < 45: score += 20  # Room to grow
    if vol == "✅ High": score += 20
    
    if score >= 80: return "🚀 STRONG BUY"
    if score >= 60: return "✅ BUY"
    if score <= 30: return "📉 STRONG SELL"
    if score <= 50: return "⚠️ AVOID"
    return "⏳ WAIT"

def calculate_index_score(results):
    """Calculates a weighted score based on Mag 7 Market Cap"""
    # Approximate market cap weights (AAPL, MSFT, NVDA are the kings)
    weights = {"AAPL": 0.22, "MSFT": 0.22, "NVDA": 0.20, "GOOGL": 0.12, "AMZN": 0.12, "META": 0.08, "TSLA": 0.04}
    
    total_score = 0
    for r in results:
        ticker = r["Ticker"]
        sig = r["FINAL SIGNAL"]
        # Convert signal to points
        points = 100 if "BUY" in sig else (0 if "SELL" in sig else 50)
        total_score += points * weights.get(ticker, 0.10)
    
    return round(total_score, 1)

# --- CORE LOGIC ---

tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]

st.title("🎯 Mag 7 Technical Sniper & Index Monitor")
st.subheader(f"Market Status as of {datetime.now().strftime('%H:%M EAT')}")

results = []

# Progress bar for data fetching
progress = st.progress(0)

# --- CORE LOGIC UPDATE ---

for i, ticker in enumerate(tickers):
    try:
        # 1. Download data
        data = yf.download(ticker, period="1y", interval="1d")
        
        # 2. Calculate Technicals
        data['SMA200'] = ta.sma(data['Close'], length=200)
        data['RSI'] = ta.rsi(data['Close'], length=14)
        
        # 3. FIXED EXTRACTION: Using .iloc[-1] to get single values
        close = float(data['Close'].iloc[-1])
        sma200 = float(data['SMA200'].iloc[-1])
        rsi = float(data['RSI'].iloc[-1])
        
        # 4. Volume Analysis
        avg_vol = data['Volume'].tail(20).mean()
        curr_vol = float(data['Volume'].iloc[-1])
        vol_status = "✅ High" if curr_vol > avg_vol else "❌ Low"
        
        trend = "Bullish" if close > sma200 else "Bearish"
        support, resistance = get_pivot_levels(data)
        
        # Replace these with your actual Greeks logic
        prob = np.random.randint(40, 95) 
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

# --- FIXED TABLE DISPLAY ---
df_results = pd.DataFrame(results)

# Changed applymap to map (newer pandas version fix)
def color_signal(val):
    color = 'green' if 'BUY' in val else ('red' if 'SELL' in val else 'white')
    return f'color: {color}; font-weight: bold'

st.table(df_results.style.map(color_signal, subset=['FINAL SIGNAL']))
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

# 1. THE INDEX SCORE (The Big Number)
index_val = calculate_index_score(results)
col1, col2 = st.columns([1, 3])

with col1:
    delta_val = index_val - 50
    st.metric(
        label="Mag 7 Index Sentiment", 
        value=f"{index_val}/100", 
        delta=f"{'BULLISH' if index_val > 50 else 'BEARISH'}",
        delta_color="normal"
    )

with col2:
    if index_val >= 70:
        st.success("🔥 NASDAQ MOMENTUM: Institutional buying confirmed. Look for Sniper Longs.")
    elif index_val <= 40:
        st.error("⚠️ NASDAQ WEAKNESS: Market Cap leaders are failing. Short bias preferred.")
    else:
        st.warning("⏳ CONSOLIDATION: Market is indecisive. Trade individual stock levels only.")

# 2. THE MAIN TABLE
df_results = pd.DataFrame(results)

def color_signal(val):
    color = 'green' if 'BUY' in val else ('red' if 'SELL' in val else 'white')
    return f'color: {color}; font-weight: bold'

st.table(df_results.style.applymap(color_signal, subset=['FINAL SIGNAL']))

st.divider()
st.info("💡 **Entry Floor:** The Pivot Point support where you should look for price to bounce. **Ceiling:** The Pivot Resistance where you should consider taking profit.")
