import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import time
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    # Basic Black-Scholes parameters
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(d2))
    else: # put
        delta = norm.cdf(d1) - 1
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2))
        
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100 # Vega per 1% change
    
    return {
        "Delta": round(delta, 3),
        "Gamma": round(gamma, 4),
        "Theta": round(theta / 365, 3), # Daily decay
        "Vega": round(vega, 3)
    }
# Insert this right after your st.set_page_config
hide_github_icon = """
    <style>
    #GithubIcon { visibility: hidden; }
    </style>
"""
st.markdown(hide_github_icon, unsafe_allow_html=True)
# This forces the display to show Nairobi time regardless of server location
nairobi_time = datetime.utcnow() + timedelta(hours=3)
st.write(f"Last Updated: {nairobi_time.strftime('%H:%M:%S')} (EAT)")
# Add this to your imports
from streamlit_autorefresh import st_autorefresh

# Add this right after st.title(...)
# 300,000 milliseconds = 5 minutes
count = st_autorefresh(interval=300000, key="fpl_refresh")
# --- APP CONFIGURATION ---
st.set_page_config(page_title="Mag 7 Live Monitor", layout="wide")
from datetime import datetime, timedelta

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
# --- INSIDE THE LOOP (for ticker in tickers:) ---
with st.spinner("Scanning Markets..."):
    import time

# ... inside your ticker loop ...
for ticker in tickers:
    try:
        tk = yf.Ticker(ticker)
        # 1. Add a small sleep to avoid Rate Limit
        time.sleep(2) 
        
        # 2. Safely check for options
        if tk.options:
            expiry = tk.options[0]
            opts = tk.option_chain(expiry)
            # ... (rest of your Greeks logic here) ...
        else:
            greeks = {"Delta": "N/A", "Theta": "N/A"}
            
    except Exception as e:
        # If one stock fails (like AAPL did in your log), 
        # the app will keep moving to the next stock instead of crashing.
        st.warning(f"Could not fetch Greeks for {ticker}. Using defaults.")
        greeks = {"Delta": "N/A", "Theta": "N/A"}
            
            # Trend and Volume Status
            trend = "Bullish" if close > sma200 else "Bearish"
            vol_status = "✅ High" if curr_vol > avg_vol else "❌ Low"
            
            # Core Strategy Logic
            status = "⏳ WAIT"
            if trend == "Bullish" and rsi < rsi_threshold and curr_vol > avg_vol:
                status = "🔥 BUY"
            elif trend == "Bearish":
                status = "⚠️ AVOID"
            elif rsi > 70:
                status = "✂️ TRIM"

            # Target Logic: Only show +2% if it's a BUY signal
            tp_level = close * 1.02
            target_display = f"${tp_level:.2f}" if status == "🔥 BUY" else "---"

            results.append({
                "Ticker": ticker,
                "Price": f"${close:.2f}",
                "Trend": trend,
                "RSI (15m)": f"{rsi:.1f}", # Rounds the long decimal
                "Volume": vol_status,
                "Signal": status,
                "Target (+2%)": target_display # Fills the empty column
            })
            # 1. Fetch Option Chain (Nearest Expiry)
        tk = yf.Ticker(ticker)
        if tk.options:
            expiry = tk.options[0] # Gets the closest expiry date
            opts = tk.option_chain(expiry)
            
            # 2. Find the At-The-Money (ATM) Call
            # We look for the strike price closest to current price
            calls = opts.calls
            atm_call = calls.iloc[(calls['strike'] - close).abs().argsort()[:1]].iloc[0]
            
            # 3. Prepare Inputs
            # T = Time to expiry in years (approximate)
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
            days_to_expiry = (expiry_date - datetime.now()).days
            T = max(days_to_expiry, 1) / 365
            
            r = 0.04  # Risk-free rate (approx 4% currently)
            iv = atm_call['impliedVolatility']
            
            # 4. Run Calculation
            greeks = calculate_greeks(close, atm_call['strike'], T, r, iv)
            
            # Add to your results list
            # Make sure your results dictionary includes the new keys
results.append({
    "Ticker": ticker,
    "Price": f"${close:.2f}",
    "RSI (15m)": f"{rsi:.1f}",
    "Delta": greeks.get("Delta", "N/A"), # Use .get() to avoid crashes if data is missing
    "Theta": greeks.get("Theta", "N/A"),
    "Signal": status,
    "Target (+2%)": target_display
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
