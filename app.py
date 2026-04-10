import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from datetime import datetime

# --- SETTINGS & THEME ---
st.set_page_config(page_title="Mag 7 Sniper Dashboard", layout="wide")

# Custom CSS for a professional dark-mode look
st.markdown("""
    <style>
    .stMetric { background-color: #1e1e1e; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    [data-testid="stMetricValue"] { color: #00ff00 !important; }
    </style>
""", unsafe_allow_html=True)

# --- ANALYTICAL FUNCTIONS ---

def get_pivot_levels(df):
    """Calculates Daily Pivot Points for precise entry and exit zones"""
    last_day = df.iloc[-1]
    high = float(last_day['High'])
    low = float(last_day['Low'])
    close = float(last_day['Close'])
    
    pivot = (high + low + close) / 3
    support = (pivot * 2) - high
    resistance = (pivot * 2) - low
    return round(support, 2), round(resistance, 2)

def generate_consensus_signal(rsi, trend, vol, prob):
    """The 'Brain' of the sniper: Requires multi-factor confirmation"""
    score = 0
    if trend == "Bullish": score += 30
    if prob >= 70: score += 30
    if rsi < 50: score += 20  # Room for upside
    if vol == "✅ High": score += 20
    
    if score >= 80: return "🚀 STRONG BUY"
    if score >= 60: return "✅ BUY"
    if score <= 30: return "📉 STRONG SELL"
    if score <= 50: return "⚠️ AVOID"
    return "⏳ WAIT"

def calculate_index_score(results_list):
    """Weights the Mag 7 by Market Cap (NVDA/AAPL/MSFT move the market more)"""
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
st.subheader(f"Session Status: {datetime.now().strftime('%Y-%m-%d %H:%M EAT')}")

results = []
progress_bar = st.progress(0)

for i, ticker in enumerate(tickers):
    try:
        # 1. Download data with silent progress
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        
        if not data.empty:
            # 2. Calculate Indicators
            data['SMA200'] = ta.sma(data['Close'], length=200)
            data['RSI'] = ta.rsi(data['Close'], length=14)
            
            # 3. Secure single values (Fixed the 'Series' vs 'Float' bug)
            close = float(data['Close'].iloc[-1])
            sma200 = float(data['SMA200'].iloc[-1])
            rsi = float(data['RSI'].iloc[-1])
            
            # 4. Volume Analysis
            avg_vol = data['Volume'].tail(20).mean()
            curr_vol = float(data['Volume'].iloc[-1])
            vol_status = "✅ High" if curr_vol > avg_vol else "❌ Low"
            
            # 5. Trend & Pivot Levels
            trend = "Bullish" if close > sma200 else "Bearish"
            support, resistance = get_pivot_levels(data)
            
            # 6. Sentiment Probability (Simulating your Greeks logic)
            prob = np.random.randint(45, 95) 
            
            # 7. Final Signal Generation
            final_sig = generate_consensus_signal(rsi, trend, vol_status, prob)
            
            # 8. Data Storage
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
    
    progress_bar.progress((i + 1) / len(tickers))

# --- DASHBOARD DISPLAY ---

if results:
    # Top Row: Weighted Index Score
    index_val = calculate_index_score(results)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            label="Mag 7 Weighted Score", 
            value=f"{index_val}/100", 
            delta=f"{'BULLISH' if index_val > 50 else 'BEARISH'}"
        )

    with col2:
        if index_val >= 70:
            st.success("🔥 BULLISH CONFLUENCE: Tech leaders are pushing. Favor long entries at 'Floor'.")
        elif index_val <= 40:
            st.error("⚠️ BEARISH BIAS: Major resistance detected. Tighten stop-losses.")
        else:
            st.warning("⏳ NEUTRAL: Mag 7 is split. Trade individual stock levels cautiously.")

    # Main Table
    df_results = pd.DataFrame(results)

    def color_signal(val):
        color = 'lime' if 'BUY' in val else ('crimson' if 'SELL' in val else 'white')
        return f'color: {color}; font-weight: bold'

    # Using .map instead of .applymap for modern Pandas compatibility
    st.table(df_results.style.map(color_signal, subset=['FINAL SIGNAL']))

st.divider()
st.caption("💡 Nairobi Sniper Guide: 'Entry Floor' is the Daily Pivot S1. 'Ceiling' is the Daily Pivot R1.")
