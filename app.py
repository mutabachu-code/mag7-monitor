import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ... (Previous imports and Delta function) ...

for ticker in TICKERS:
    try:
        stock = yf.Ticker(ticker)
        # Fetch 3 days to get a better rolling average for volume
        df = stock.history(period="3d", interval="5m", prepost=True).ffill().bfill()
        
        # --- ENHANCED VOLUME LOGIC ---
        last_vol = df['Volume'].iloc[-1]
        
        # Filter out "placeholder" volumes (like 0 or 1) that break the ratio
        if last_vol <= 1:
            vol_status = "No Activity ⭕"
            vol_ratio = 1.0
        else:
            # Compare current volume to the median of the last 20 bars
            # Median is better than Mean because it ignores one-off massive spikes
            median_vol = df['Volume'].tail(20).median()
            vol_ratio = last_vol / median_vol if median_vol > 0 else 1.0
            
            if vol_ratio > 2.0: vol_status = "SURGE 🚀"
            elif vol_ratio > 1.2: vol_status = "High 📈"
            elif vol_ratio < 0.5: vol_status = "Low 📉"
            else: vol_status = "Normal"

        # --- REFINED SIGNAL ---
        # Only give a BUY/SELL if volume is at least "Normal" or "High"
        # This prevents fake signals during thin premarket trading
        if rsi < 30 and vol_ratio > 1.2:
            signal = "🟢 STRONG BUY"
        elif rsi > 70 and vol_ratio > 1.2:
            signal = "🔴 STRONG SELL"
        elif vol_ratio > 2.5:
            signal = "⚠️ VOL SPIKE - WATCH"
        else:
            signal = "🟡 WAIT"

        # ... (UI Display Code) ...
