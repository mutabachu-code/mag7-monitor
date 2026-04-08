import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import time
from datetime import datetime, timedelta
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
import requests

# Create a custom session to bypass basic bot detection
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})
# --- 1. APP CONFIGURATION & SECURITY ---
st.set_page_config(page_title="Mag 7 Live Monitor", layout="wide")

# Hide GitHub Icon for privacy
hide_github_icon = """
    <style>
    #GithubIcon { visibility: hidden; }
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)

# --- 2. GREEKS ENGINE (Black-Scholes) ---
def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Delta & Theta (Existing)
        if option_type == 'call':
            delta = norm.cdf(d1)
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            delta = norm.cdf(d1) - 1
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        # Gamma (The Acceleration Factor)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        
        return {
            "Delta": round(delta, 3), 
            "Gamma": round(gamma, 4),
            "Theta": round(theta / 365, 3),
            "Prob %": round(delta * 100, 1) # Probability of success
        }
    except:
        return {"Delta": "N/A", "Gamma": "N/A", "Theta": "N/A", "Prob %": "N/A"}

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# --- 3. SIDEBAR & REFRESH ---
st.sidebar.header("⚙️ Strategy Settings")
rsi_threshold = st.sidebar.slider("RSI Entry Level", 20, 45, 35)
refresh_choice = st.sidebar.selectbox("Refresh Every (Min)", [1, 5, 10, 15], index=1)

# Auto-refresh trigger
st_autorefresh(interval=refresh_choice * 60000, key="data_refresh")

# --- 4. MAIN UI ---
st.title("🚀 Mag 7 Live Technical Monitor")
nairobi_time = datetime.utcnow() + timedelta(hours=3)
st.write(f"Last Updated: **{nairobi_time.strftime('%H:%M:%S')}** (EAT)")

tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL"]
results = []

# --- 5. THE DATA ENGINE ---
with st.spinner("Fetching Market Data & Greeks..."):
    for ticker in tickers:
        try:
            # Avoid Rate Limiting with a tiny pause
            time.sleep(1.5) 
            
            # Fetch Price Data
            tk = yf.Ticker(ticker)
            df_d = tk.history(period="1y")
            df_m15 = tk.history(period="5d", interval="15m")
            
            if df_d.empty: continue

            # Technical Indicators
            close = float(df_d['Close'].iloc[-1])
            sma200 = df_d['Close'].rolling(window=200).mean().iloc[-1]
            rsi = float(calculate_rsi(df_m15['Close']).iloc[-1])
            avg_vol = df_d['Volume'].rolling(window=10).mean().iloc[-1]
            curr_vol = float(df_d['Volume'].iloc[-1])
            
            trend = "Bullish" if close > sma200 else "Bearish"
            vol_status = "✅ High" if curr_vol > avg_vol else "❌ Low"
            
            # Signal Logic
            status = "⏳ WAIT"
            if trend == "Bullish" and rsi < rsi_threshold and curr_vol > avg_vol:
                status = "🔥 BUY"
            elif trend == "Bearish":
                status = "⚠️ AVOID"
            
            target_display = f"${close * 1.02:.2f}" if status == "🔥 BUY" else "---"

            # --- FETCH GREEKS ---
            greeks = {"Delta": "N/A", "Theta": "N/A"}
            if tk.options:
                expiry = tk.options[0]
                opts = tk.option_chain(expiry)
                calls = opts.calls
                # Find nearest ATM Strike
                atm_call = calls.iloc[(calls['strike'] - close).abs().argsort()[:1]].iloc[0]
                
                expiry_dt = datetime.strptime(expiry, '%Y-%m-%d')
                T = max((expiry_dt - datetime.now()).days, 1) / 365
                greeks = calculate_greeks(close, atm_call['strike'], T, 0.04, atm_call['impliedVolatility'])

            results.append({
                "Ticker": ticker,
                "Price": f"${close:.2f}",
                "Delta": greeks["Delta"],
                "Gamma": greeks["Gamma"], # New Column
                "Prob %": greeks["Prob %"], # New Column
                "Theta": greeks["Theta"],
                "Signal": status,
                "Target (+2%)": target_display
            })
        except Exception as e:
            st.error(f"Error loading {ticker}: {e}")

# --- 6. DISPLAY ---
if results:
    df_display = pd.DataFrame(results)
    
    def color_signal(val):
        color = 'white'
        if val == '🔥 BUY': color = '#2ecc71'
        elif val == '⚠️ AVOID': color = '#e74c3c'
        elif val == '⏳ WAIT': color = '#f1c40f'
        return f'background-color: {color}; color: black; font-weight: bold'

    st.table(df_display.style.map(color_signal, subset=['Signal']))
else:
    st.warning("No data available. Check logs for Rate Limit errors.")
