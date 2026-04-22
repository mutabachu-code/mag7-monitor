import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh

from claude_analyst import analyse
from risk_manager import RiskConfig, init_risk_state, render_risk_sidebar, check_trade_allowed, record_trade_opened

# ── SETUP ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Mag 7 + NAS100 Monitor", layout="wide")
init_risk_state()
st_autorefresh(interval=60000, key="datarefresh")

risk_config = RiskConfig(
    account_size_usd=100.0,
    lot_size=0.02,
    daily_loss_limit_pct=5.0,
    max_trades_per_day=3,
)
risk_config = render_risk_sidebar(risk_config)

st.title("🛡️ Mag 7 + NAS100 MTF Monitor + Claude AI")
st.caption(
    f"Last Update: {pd.Timestamp.now().strftime('%H:%M:%S')} | "
    "5m Signals · 1H MACD · Claude Analysis (News + Sentiment + Technicals)"
)

# ── TICKERS ───────────────────────────────────────────────────────────────────
MAG7    = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']
NAS100_TICKER = '^NDX'   # Nasdaq-100 index via yfinance
NAS100_LABEL  = 'NAS100'


def get_bs_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# ── HEATMAP CALCULATION ───────────────────────────────────────────────────────
def compute_heatmap(tickers: list) -> pd.DataFrame:
    """
    Fetch hourly close prices for all Mag7 tickers + NAS100.
    Returns a DataFrame of 1-hour returns for heatmap rendering.
    Adds small delay between requests to avoid yfinance rate limits.
    """
    import time as _time
    rows = []
    for ticker in tickers:
        try:
            yfticker = NAS100_TICKER if ticker == NAS100_LABEL else ticker
            df = yf.Ticker(yfticker).history(period="2d", interval="1h").ffill().bfill()
            _time.sleep(0.3)
            if len(df) < 2:
                continue
            # Last 8 hourly candles (1 trading day)
            closes = df['Close'].tail(8)
            pct_changes = closes.pct_change().dropna() * 100
            row = {"Ticker": ticker}
            for j, (ts, val) in enumerate(pct_changes.items()):
                row[f"H{j+1}"] = round(val, 2)
            row["Day %"] = round(
                (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100, 2
            )
            row["Price"] = round(closes.iloc[-1], 2)
            rows.append(row)
        except Exception as e:
            print(f"[heatmap] Error for {ticker}: {e}")
    return pd.DataFrame(rows)


def color_cell(val):
    """Return green/red background CSS based on positive/negative return."""
    if pd.isna(val) or not isinstance(val, (int, float)):
        return ""
    if val > 1.0:
        return "background-color: #1a7a1a; color: white"
    elif val > 0.3:
        return "background-color: #2d9e2d; color: white"
    elif val > 0:
        return "background-color: #5cb85c; color: white"
    elif val > -0.3:
        return "background-color: #d9534f; color: white"
    elif val > -1.0:
        return "background-color: #c9302c; color: white"
    else:
        return "background-color: #8b0000; color: white"


def render_heatmap():
    """Render the NAS100 + Mag7 heatmap section."""
    st.subheader("📊 NAS100 + Mag 7 Hourly Heatmap")
    st.caption("Hourly % returns for today · Green = up · Red = down · Intensity = magnitude")

    all_tickers = [NAS100_LABEL] + MAG7

    # Cache heatmap for 5 minutes to avoid hammering yfinance on every 60s refresh
    import time as _time
    cache = st.session_state.get("heatmap_cache", None)
    cache_age = _time.time() - st.session_state.get("heatmap_ts", 0)

    if cache is None or cache_age > 300:
        with st.spinner("Loading heatmap data..."):
            df = compute_heatmap(all_tickers)
        st.session_state["heatmap_cache"] = df
        st.session_state["heatmap_ts"] = _time.time()
    else:
        df = cache

    if df.empty:
        st.warning("Heatmap data unavailable — market may be closed.")
        return

    # Style the heatmap
    hour_cols = [c for c in df.columns if c.startswith("H")]
    display_cols = ["Ticker", "Price", "Day %"] + hour_cols

    df_display = df[display_cols].set_index("Ticker")

    styled = df_display.style.map(
        color_cell, subset=["Day %"] + hour_cols
    ).format({
        "Price": "${:,.2f}",
        "Day %": "{:+.2f}%",
        **{h: "{:+.2f}%" for h in hour_cols}
    })

    st.dataframe(styled, use_container_width=True, height=320)
    st.divider()


# ── INDICATOR ENGINE (shared for Mag7 + NAS100) ───────────────────────────────
def compute_indicators(yfticker: str, label: str):
    """
    Fetch data and compute all indicators for one ticker.
    Returns a dict of indicator values, or None on failure.
    """
    stock  = yf.Ticker(yfticker)
    df_5m  = stock.history(period="5d",  interval="5m",  prepost=True).ffill().bfill()
    df_1h  = stock.history(period="60d", interval="1h").ffill().bfill()

    if len(df_1h) < 200 or len(df_5m) < 20:
        return None

    sma200_1h    = df_1h['Close'].rolling(window=200).mean().iloc[-1]
    curr_p       = df_5m['Close'].iloc[-1]
    prev_p       = df_5m['Close'].iloc[-2]
    trend_status = "BULLISH" if curr_p > sma200_1h else "BEARISH"
    trend_color  = "green" if trend_status == "BULLISH" else "red"

    ema12        = df_1h['Close'].ewm(span=12, adjust=False).mean()
    ema26        = df_1h['Close'].ewm(span=26, adjust=False).mean()
    macd_line    = ema12 - ema26
    signal_line  = macd_line.ewm(span=9, adjust=False).mean()
    macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

    delta_p = df_5m['Close'].diff()
    gain    = (delta_p.where(delta_p > 0, 0)).rolling(window=14).mean()
    loss    = (-delta_p.where(delta_p < 0, 0)).rolling(window=14).mean()
    rs      = gain / loss
    rsi     = 100 - (100 / (1 + rs)).iloc[-1]

    df_5m['vol_ma_long']  = df_5m['Volume'].rolling(window=20).mean()
    df_5m['vol_ma_short'] = df_5m['Volume'].rolling(window=5).mean()
    vol_ratio = (
        df_5m['vol_ma_short'].iloc[-1] / df_5m['vol_ma_long'].iloc[-1]
        if df_5m['vol_ma_long'].iloc[-1] > 0 else 1.0
    )

    start_price = df_5m['Open'].iloc[0]
    ann_vol     = df_5m['Close'].pct_change().std() * np.sqrt(252 * 78)
    delta_val   = get_bs_delta(curr_p, start_price, 1/252, 0.045, ann_vol)

    # Signal logic
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

    return dict(
        label=label,
        curr_p=curr_p,
        prev_p=prev_p,
        sma200_1h=sma200_1h,
        trend_status=trend_status,
        trend_color=trend_color,
        macd_bullish=macd_bullish,
        rsi=rsi,
        vol_ratio=vol_ratio,
        delta_val=delta_val,
        signal=signal,
        sig_color=sig_color,
    )


def render_ticker_card(ind: dict, col, risk_config: RiskConfig):
    """Render one ticker card — shared by Mag7 and NAS100."""
    with col:
        st.metric(
            label=ind["label"],
            value=f"${ind['curr_p']:,.2f}",
            delta=f"{ind['curr_p'] - ind['prev_p']:.2f}",
        )
        st.markdown(
            f"**Trend (1H SMA200):** :{ind['trend_color']}[{ind['trend_status']}]"
        )
        st.markdown(
            f"**Signal:** :{ind['sig_color']}[{ind['signal']}]"
        )

        with st.expander("Technical Details"):
            st.write(f"RSI (5m): {ind['rsi']:.1f}")
            st.write(f"Vol Surge: {ind['vol_ratio']:.2f}x")
            st.write(f"Opt. Delta: {ind['delta_val']:.2f}")
            st.write(f"MACD: {'Bullish 📈' if ind['macd_bullish'] else 'Bearish 📉'}")

        if ind["signal"] != "⚪ Neutral":
            with st.expander("🤖 Claude AI Analysis", expanded=True):
                trade_allowed, trade_reason = check_trade_allowed(risk_config, ind["label"])

                ai = analyse(
                    ticker=ind["label"],
                    current_price=ind["curr_p"],
                    raw_signal=ind["signal"],
                    rsi=ind["rsi"],
                    vol_ratio=ind["vol_ratio"],
                    macd_bullish=ind["macd_bullish"],
                    trend_status=ind["trend_status"],
                    delta_val=ind["delta_val"],
                    sma200=ind["sma200_1h"],
                    account_balance=risk_config.account_size_usd,
                    lot_size=risk_config.lot_size,
                )

                if ai is None:
                    st.warning("Claude analysis unavailable — check API key.")
                else:
                    action_color = {
                        "BUY": "green", "SELL": "red", "HOLD": "gray"
                    }.get(ai.action, "gray")

                    st.markdown(
                        f"**Decision:** :{action_color}[{ai.action}] "
                        f"| Confidence: **{ai.confidence}**"
                    )
                    st.caption(f"📊 {ai.reasoning}")
                    if ai.sentiment_summary:
                        st.info(f"🗞️ {ai.sentiment_summary}")

                    if ai.action != "HOLD":
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Entry",       f"${ai.entry_price:,.2f}")
                        c2.metric("Stop Loss",   f"${ai.stop_loss:,.2f}",
                                  delta=f"{((ai.stop_loss - ai.entry_price)/ai.entry_price*100):+.2f}%",
                                  delta_color="off")
                        c3.metric("Take Profit", f"${ai.take_profit:,.2f}",
                                  delta=f"{((ai.take_profit - ai.entry_price)/ai.entry_price*100):+.2f}%",
                                  delta_color="off")

                        risk_usd   = abs(ai.entry_price - ai.stop_loss) * (ai.lot_size * 100)
                        reward_usd = abs(ai.take_profit - ai.entry_price) * (ai.lot_size * 100)
                        rr         = reward_usd / risk_usd if risk_usd > 0 else 0
                        st.caption(
                            f"Lot: {ai.lot_size} | "
                            f"Risk: ~${risk_usd:.2f} | "
                            f"Reward: ~${reward_usd:.2f} | "
                            f"R:R = {rr:.1f}:1"
                        )

                        if trade_allowed:
                            if st.button(
                                f"⚡ Execute {ai.action} {ind['label']}",
                                key=f"exec_{ind['label']}",
                                type="primary",
                            ):
                                from broker import place_order
                                result = place_order(
                                    ticker=ind["label"],
                                    action=ai.action,
                                    stop_loss=ai.stop_loss,
                                    take_profit=ai.take_profit,
                                    lot_size=ai.lot_size,
                                )
                                if result.success:
                                    record_trade_opened(
                                        ticker=ind["label"],
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

# ── SECTION 1: HEATMAP ────────────────────────────────────────────────────────
render_heatmap()

# ── SECTION 2: NAS100 CARD ────────────────────────────────────────────────────
st.subheader("📈 Nasdaq 100 Cash CFD (NAS100)")
nas_col, _, _, _ = st.columns(4)

try:
    nas_ind = compute_indicators(NAS100_TICKER, NAS100_LABEL)
    if nas_ind:
        render_ticker_card(nas_ind, nas_col, risk_config)
    else:
        with nas_col:
            st.warning("NAS100 data unavailable")
except Exception as e:
    with nas_col:
        st.error(f"NAS100 error: {e}")

st.divider()

# ── SECTION 3: MAG 7 CARDS ────────────────────────────────────────────────────
st.subheader("🛡️ Magnificent 7 Stocks")
cols = st.columns(4)

for i, ticker in enumerate(MAG7):
    try:
        ind = compute_indicators(ticker, ticker)
        if ind:
            render_ticker_card(ind, cols[i % 4], risk_config)
    except Exception as e:
        with cols[i % 4]:
            st.error(f"Error {ticker}: {e}")
