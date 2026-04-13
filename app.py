import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

# --- CONFIGURATION ---
MAG_7 = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
INTERVAL = "5m"  # Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
PERIOD = "1d"    # Lookback period

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_delta_proxy(S, K, T, r, sigma):
    """Simplified Black-Scholes Delta for Call Option as a sentiment proxy"""
    if T <= 0 or sigma <= 0: return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def analyze_mag7():
    print(f"--- Fetching Mag 7 Data (Intraday + Premarket) | {datetime.now().strftime('%H:%M:%S')} ---")
    results = []

    for ticker in MAG_7:
        try:
            # 1. Fetch Intraday Data (including Pre/Post Market)
            stock = yf.Ticker(ticker)
            df = stock.history(period=PERIOD, interval=INTERVAL, prepost=True)
            
            if df.empty: continue

            # 2. Technical Calculations
            df['RSI'] = calculate_rsi(df)
            current_price = df['Close'].iloc[-1]
            last_rsi = df['RSI'].iloc[-1]
            avg_volume = df['Volume'].tail(10).mean()
            last_volume = df['Volume'].iloc[-1]
            
            # 3. Sentiment/Greeks Proxy
            # We use ATM (At-the-money) strike for sentiment; 0.05 (5%) risk-free rate; 1 day to expiry
            volatility = df['Close'].pct_change().std() * np.sqrt(252 * (6.5 * 12)) # Annualized intraday vol
            delta = get_delta_proxy(current_price, current_price, 1/252, 0.05, volatility)

            # 4. Generate Signals
            signal = "NEUTRAL"
            if last_rsi < 35 and last_volume > avg_volume:
                signal = "🚀 BULLISH (Oversold + Vol Spike)"
            elif last_rsi > 65 and last_volume > avg_volume:
                signal = "⚠️ BEARISH (Overbought + Vol Spike)"
            elif delta > 0.55:
                signal = "📈 MILD BULLISH (Delta Strength)"
            elif delta < 0.45:
                signal = "📉 MILD BEARISH (Delta Weakness)"

            results.append({
                "Ticker": ticker,
                "Price": f"${current_price:.2f}",
                "RSI": round(last_rsi, 2),
                "Vol/Avg": round(last_volume/avg_volume, 2),
                "Delta Proxy": round(delta, 2),
                "Signal": signal
            })

        except Exception as e:
            print(f"Error analyzing {ticker}: {e}")

    # Display Results
    final_df = pd.DataFrame(results)
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    analyze_mag7()
