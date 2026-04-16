import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

from claude_analyst import analyse
from risk_manager import RiskConfig, init_risk_state, render_risk_sidebar, check_trade_allowed, record_trade_opened

# ── SETUP ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Mag 7 Live MTF Monitor", layout="wide")
init_risk_state()
st_autorefresh(interval=60000, key="datarefresh")

risk_config = RiskConfig(
    account_size_usd=100.0,
    lot_size=0.02,
    daily_loss_limit_pct=5.0,
    max_trades_per_day=3,
)
risk_config = render_risk_sidebar(risk_config)

st.title("🛡️ Mag 7 MTF Monitor + Claude AI")
st.caption(
    f"Last Update: {pd.Timestamp.now().strftime('%H:%M:%S')} | "
    "5m Signals · 1H MACD · Claude Analysis (News + Sentiment + Technicals)"
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
        # ── FETCH DATA ────────────────────────────────────────────────────────
        stock  = yf.Ticker(ticker)
        df_5m  = stock.history(period="5d",  interval="5m",  prepost=True).ffill().bfill()
        df_1h  = stock.history(period="60d", interval="1h").ffill().bfill()

        if len(df_1h) < 200:
            continue

        # ── INDICATORS (unchanged) ────────────────────────────────────────────
        sma200_1h    = df_1h['Close'].rolling(window=200).mean().iloc[-1]
        curr_p       = df_5m['Close'].iloc[-1]
        trend_status = "BULLISH" if curr_p > sma200_1h else "BEARISH"
        trend_color  = "green" if trend_status == "BULLISH" else "red"

        ema12        = df_1h['Close'].ewm(span=12, adjust=False).mean()
        ema26        = df_1h['Close'].ewm(span=26, adjust=False).mean()
        macd_line    = ema12 - ema26
        signal_line  = macd_line.ewm(span=9, adjust=False).mean()
        macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

        delta_p  = df_5m['Close'].diff()
        gain     = (delta_p.where(delta_p > 0, 0)).rolling(window=14).mean()
        loss     = (-delta_p.where(delta_p < 0, 0)).rolling(window=14).mean()
        rs       = gain / loss
        rsi      = 100 - (100 / (1 + rs)).iloc[-1]

        df_5m['vol_ma_long']  = df_5m['Volume'].rolling(window=20).mean()
        df_5m['vol_ma_short'] = df_5m['Volume'].rolling(window=5).mean()
        vol_ratio = (
            df_5m['vol_ma_short'].iloc[-1] / df_5m['vol_ma_long'].iloc[-1]
            if df_5m['vol_ma_long'].iloc[-1] > 0 else 1.0
        )

        start_price = df_5m['Open'].iloc[0]
        ann_vol     = df_5m['Close'].pct_change().std() * np.sqrt(252 * 78)
        delta_val   = get_bs_delta(curr_p, start_price, 1/252, 0.045, ann_vol)

        # ── SIGNAL LOGIC (unchanged) ──────────────────────────────────────────
        if curr_p > sma200_1h and rsi < 40 and vol_ratio > 1.1 and macd_bullish:
            signal    = "🟢 STRONG BUY (Full Confluence)"
            sig_color = "green"
        elif curr_p < sma200_1h and rsi > 60 and vol_ratio > 1.1 and not macd_bullish:
            signal    = "🔴 STRONG SELL (Full Confluence)"
            sig_color = "red"
        elif curr_p > sma200_1h and rsi < 35:
            signal    = "🟡 Caution Buy (No MACD Confirm)"
            sig_color = "orange"
        else:
            signal    = "⚪ Neutral"
            sig_color = "gray"

        # ── CARD DISPLAY ──────────────────────────────────────────────────────
        with cols[i % 4]:
            st.metric(
                label=ticker,
                value=f"${curr_p:.2f}",
                delta=f"{curr_p - df_5m['Close'].iloc[-2]:.2f}",
            )
            st.markdown(f"**Trend (1H SMA200):** :{trend_color}[{trend_status}]")
            st.markdown(f"**Signal:** :{sig_color}[{signal}]")

            with st.expander("Technical Details"):
                st.write(f"RSI (5m): {rsi:.1f}")
                st.write(f"Vol Surge: {vol_ratio:.2f}x")
                st.write(f"Opt. Delta: {delta_val:.2f}")
                st.write(f"MACD: {'Bullish 📈' if macd_bullish else 'Bearish 📉'}")

            # ── CLAUDE AI ANALYSIS (only on non-neutral signals) ──────────────
            if signal != "⚪ Neutral":
                with st.expander("🤖 Claude AI Analysis", expanded=True):

                    trade_allowed, trade_reason = check_trade_allowed(risk_config, ticker)

                    ai = analyse(
                        ticker=ticker,
                        current_price=curr_p,
                        raw_signal=signal,
                        rsi=rsi,
                        vol_ratio=vol_ratio,
                        macd_bullish=macd_bullish,
                        trend_status=trend_status,
                        delta_val=delta_val,
                        sma200=sma200_1h,
                        account_balance=risk_config.account_size_usd,
                        lot_size=risk_config.lot_size,
                    )

                    if ai is None:
                        st.warning("Claude analysis unavailable — check API key.")
                    else:
                        action_color = {"BUY": "green", "SELL": "red", "HOLD": "gray"}.get(ai.action, "gray")

                        # Decision header
                        st.markdown(
                            f"**Decision:** :{action_color}[{ai.action}] "
                            f"| Confidence: **{ai.confidence}**"
                        )

                        # Reasoning
                        st.caption(f"📊 {ai.reasoning}")

                        # Sentiment/news line
                        if ai.sentiment_summary:
                            st.info(f"🗞️ {ai.sentiment_summary}")

                        if ai.action != "HOLD":
                            # Trade levels
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Entry",      f"${ai.entry_price:.2f}")
                            c2.metric("Stop Loss",  f"${ai.stop_loss:.2f}",
                                      delta=f"{((ai.stop_loss - ai.entry_price)/ai.entry_price*100):+.2f}%",
                                      delta_color="off")
                            c3.metric("Take Profit",f"${ai.take_profit:.2f}",
                                      delta=f"{((ai.take_profit - ai.entry_price)/ai.entry_price*100):+.2f}%",
                                      delta_color="off")

                            # Risk summary
                            risk_usd = abs(ai.entry_price - ai.stop_loss) * (ai.lot_size * 100)
                            reward_usd = abs(ai.take_profit - ai.entry_price) * (ai.lot_size * 100)
                            rr = reward_usd / risk_usd if risk_usd > 0 else 0
                            st.caption(
                                f"Lot: {ai.lot_size} | "
                                f"Risk: ~${risk_usd:.2f} | "
                                f"Reward: ~${reward_usd:.2f} | "
                                f"R:R = {rr:.1f}:1"
                            )

                            # Execute button — gated by risk manager
                            if trade_allowed:
                                if st.button(
                                    f"⚡ Execute {ai.action} {ticker}",
                                    key=f"exec_{ticker}",
                                    type="primary",
                                ):
                                    from broker import place_order
                                    result = place_order(
                                        ticker=ticker,
                                        action=ai.action,
                                        stop_loss=ai.stop_loss,
                                        take_profit=ai.take_profit,
                                        lot_size=ai.lot_size,
                                    )
                                    if result.success:
                                        record_trade_opened(
                                            ticker=ticker,
                                            action=ai.action,
                                            entry=result.filled_price or ai.entry_price,
                                            sl=ai.stop_loss,
                                            tp=ai.take_profit,
                                            lots=ai.lot_size,
                                        )
                                        st.success(result.message)
                                    else:
                                        st.error(result.message)
                            else:
                                st.error(trade_reason)

            st.divider()

    except Exception as e:
        st.error(f"Error {ticker}: {e}")
