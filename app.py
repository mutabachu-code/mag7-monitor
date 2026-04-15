import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

from claude_analyst import analyse
from risk_manager import RiskConfig, init_risk_state, render_risk_sidebar, check_trade_allowed

# --- DASHBOARD SETUP ---
st.set_page_config(page_title="Mag 7 Live MTF Monitor", layout="wide")

# Initialise risk state (session persistence across 60s refreshes)
init_risk_state()

# AUTO REFRESH: Every 60 seconds
st_autorefresh(interval=60000, key="datarefresh")

# --- SIDEBAR: Risk controls ---
risk_config = RiskConfig(
    max_position_pct=2.0,
    daily_loss_limit_pct=5.0,
    account_size_usd=1000.0,   # ← update to your real account size
)
risk_config = render_risk_sidebar(risk_config)

# --- HEADER ---
st.title("🛡️ Mag 7 MTF Monitor + Claude AI")
st.caption(
    f"Last Update: {pd.Timestamp.now().strftime('%H:%M:%S')} | "
    "Confluence: 5m Signals + 1H MACD Trend + Claude Analysis"
)

TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']


def get_bs_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


cols = st.columns(4)

for i, ticker in enumerate(TICKERS):
    try:
        # 1. FETCH DATA (Multi-Timeframe)
        stock = yf.Ticker(ticker)
        df_5m = stock.history(period="5d", interval="5m", prepost=True).ffill().bfill()
        df_1h = stock.history(period="60d", interval="1h").ffill().bfill()

        if len(df_1h) < 200:
            continue

        # --- CALCULATIONS (unchanged from original) ---

        # A. SMA 200 Trend (1H)
        sma200_1h = df_1h['Close'].rolling(window=200).mean().iloc[-1]
        curr_p = df_5m['Close'].iloc[-1]
        trend_status = "BULLISH" if curr_p > sma200_1h else "BEARISH"
        trend_color = "green" if trend_status == "BULLISH" else "red"

        # B. MACD Confluence (1H)
        ema12 = df_1h['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_1h['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

        # C. RSI (5m)
        delta_p = df_5m['Close'].diff()
        gain = (delta_p.where(delta_p > 0, 0)).rolling(window=14).mean()
        loss = (-delta_p.where(delta_p < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # D. Volume Analysis
        df_5m['vol_ma_long'] = df_5m['Volume'].rolling(window=20).mean()
        df_5m['vol_ma_short'] = df_5m['Volume'].rolling(window=5).mean()
        vol_ratio = (
            df_5m['vol_ma_short'].iloc[-1] / df_5m['vol_ma_long'].iloc[-1]
            if df_5m['vol_ma_long'].iloc[-1] > 0 else 1.0
        )

        # E. Options Delta
        start_price = df_5m['Open'].iloc[0]
        ann_vol = df_5m['Close'].pct_change().std() * np.sqrt(252 * 78)
        delta_val = get_bs_delta(curr_p, start_price, 1/252, 0.045, ann_vol)

        # --- REFINED SIGNAL LOGIC (unchanged) ---
        if curr_p > sma200_1h and rsi < 40 and vol_ratio > 1.1 and macd_bullish:
            signal = "🟢 STRONG BUY (Full Confluence)"
            sig_color = "green"
        elif curr_p < sma200_1h and rsi > 60 and vol_ratio > 1.1 and not macd_bullish:
            signal = "🔴 STRONG SELL (Full Confluence)"
            sig_color = "red"
        elif curr_p > sma200_1h and rsi < 35:
            signal = "🟡 Caution Buy (No MACD Confirm)"
            sig_color = "orange"
        else:
            signal = "⚪ Neutral"
            sig_color = "gray"

        # --- UI DISPLAY (original card) ---
        with cols[i % 4]:
            st.metric(
                label=ticker,
                value=f"${curr_p:.2f}",
                delta=f"{curr_p - df_5m['Close'].iloc[-2]:.2f}",
            )
            st.markdown(f"**Trend (1H SMA200):** :{trend_color}[{trend_status}]")
            st.markdown(f"**Action:** :{sig_color}[{signal}]")

            with st.expander("Technical Details"):
                st.write(f"RSI (5m): {rsi:.1f}")
                st.write(f"Vol Surge: {vol_ratio:.2f}x")
                st.write(f"Opt. Delta: {delta_val:.2f}")
                st.write(f"MACD Status: {'Bullish 📈' if macd_bullish else 'Bearish 📉'}")

            # --- CLAUDE AI ANALYSIS ---
            # Only call Claude when the dashboard sees a non-neutral signal
            # (saves API cost on neutral tickers)
            if signal != "⚪ Neutral":
                with st.expander("🤖 Claude AI Analysis", expanded=True):

                    trade_allowed, reason = check_trade_allowed(risk_config)

                    ai_signal = analyse(
                        ticker=ticker,
                        current_price=curr_p,
                        raw_signal=signal,
                        rsi=rsi,
                        vol_ratio=vol_ratio,
                        macd_bullish=macd_bullish,
                        trend_status=trend_status,
                        delta_val=delta_val,
                        sma200=sma200_1h,
                    )

                    if ai_signal is None:
                        st.warning("Claude analysis unavailable.")
                    else:
                        # Display the signal
                        action_color = {
                            "BUY": "green",
                            "SELL": "red",
                            "HOLD": "gray",
                        }.get(ai_signal.action, "gray")

                        st.markdown(
                            f"**Decision:** :{action_color}[{ai_signal.action}] "
                            f"| Confidence: **{ai_signal.confidence}**"
                        )
                        st.caption(ai_signal.reasoning)

                        if ai_signal.action != "HOLD":
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Entry", f"${ai_signal.entry_price:.2f}")
                            c2.metric("Stop Loss", f"${ai_signal.stop_loss:.2f}")
                            c3.metric("Take Profit", f"${ai_signal.take_profit:.2f}")

                            pos_usd = (
                                risk_config.account_size_usd
                                * min(ai_signal.position_size_pct, risk_config.max_position_pct)
                                / 100
                            )
                            st.caption(f"Recommended position: ${pos_usd:.2f} "
                                       f"({min(ai_signal.position_size_pct, risk_config.max_position_pct):.1f}% of account)")

                            # EXECUTE button (gated by risk manager)
                            if trade_allowed:
                                if st.button(
                                    f"⚡ Execute {ai_signal.action} {ticker}",
                                    key=f"exec_{ticker}",
                                    type="primary",
                                ):
                                    # --- BROKER CALL GOES HERE ---
                                    # from broker import place_order
                                    # result = place_order(
                                    #     ticker=ticker,
                                    #     action=ai_signal.action,
                                    #     entry=ai_signal.entry_price,
                                    #     stop_loss=ai_signal.stop_loss,
                                    #     take_profit=ai_signal.take_profit,
                                    #     position_usd=pos_usd,
                                    # )
                                    st.success(
                                        f"✅ Order queued: {ai_signal.action} {ticker} "
                                        f"@ ${ai_signal.entry_price:.2f} | "
                                        f"SL ${ai_signal.stop_loss:.2f} | "
                                        f"TP ${ai_signal.take_profit:.2f}"
                                    )
                                    # Uncomment when broker.py is ready:
                                    # record_trade_result(0)  # update after close
                            else:
                                st.error(reason)

            st.divider()

    except Exception as e:
        st.error(f"Error {ticker}: {e}")
