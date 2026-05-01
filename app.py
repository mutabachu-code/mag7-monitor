import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

from data_fetcher import fetch_all_data
from macro_monitor import render_macro_panel, get_macro_snapshot, get_5m, get_1h, get_1d, get_vix, get_heatmap_data, get_qqq_ndx_ratio, MAG7
from iv_calculator import get_iv_data
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
    "5m Signals · 1H MACD · IV Analysis · Claude AI"
)

NAS100_LABEL = 'NAS100'


# ── SESSION GATE ──────────────────────────────────────────────────────────────
def _claude_allowed_stocks() -> tuple:
    """
    Gate Claude calls to NY market hours only.
    Saves API costs outside trading hours.
    Returns (allowed: bool, reason: str, prime_time: bool)
    """
    utc   = datetime.now(timezone.utc)
    start = utc.replace(hour=14, minute=30, second=0, microsecond=0)
    end   = utc.replace(hour=21, minute=0,  second=0, microsecond=0)
    prime = utc.replace(hour=16, minute=0,  second=0, microsecond=0)
    wday  = utc.weekday()

    if wday >= 5:
        return False, "🔴 Weekend — Claude resumes Monday 14:30 UTC", False
    if utc < start:
        mins = int((start - utc).total_seconds() / 60)
        return False, f"🕐 Claude activates at NY Open in {mins} min (14:30 UTC / 17:30 Nairobi)", False
    if utc > end:
        return False, "🔴 NY closed — Claude resumes tomorrow 14:30 UTC", False
    if utc <= prime:
        return True, "🟢 NY Open — Prime Time · Claude ACTIVE", True
    return True, "🟡 NY Mid-Session · Claude active (strong signals only)", False


# ── HELPERS ───────────────────────────────────────────────────────────────────
def get_bs_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.5
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


# ── DATA FETCH ────────────────────────────────────────────────────────────────
import time as _time
fetch_start = _time.time()
with st.spinner("Fetching market data..."):
    data_ok = fetch_all_data()
fetch_elapsed = _time.time() - fetch_start

if not data_ok:
    has_stale = any(
        st.session_state.get(f"df_5m_{t}") is not None
        for t in ([NAS100_LABEL] + MAG7)
    )
    if has_stale:
        st.warning("⚠️ Live fetch failed — showing last known prices. Retrying next refresh.")
    else:
        st.error("⚠️ No market data available. Market may be closed or yfinance is rate-limited.")
        st.stop()
elif fetch_elapsed > 15:
    st.caption(f"⏱️ Data loaded in {fetch_elapsed:.0f}s")

vix_value = get_vix()


# ── HEATMAP ───────────────────────────────────────────────────────────────────
def render_heatmap():
    st.subheader("📊 NAS100 + Mag 7 Hourly Heatmap")

    # Market status
    utc      = datetime.now(timezone.utc)
    ny_hour  = (utc.hour - 4) % 24
    ny_wday  = utc.weekday()
    ny_str   = f"{ny_hour:02d}:{utc.minute:02d} NY time"

    if ny_wday >= 5:
        st.error(f"🔴 Market CLOSED (Weekend) · {ny_str}")
    elif (ny_hour == 9 and utc.minute >= 30) or (10 <= ny_hour < 16):
        st.success(f"🟢 Market OPEN — Live data · {ny_str}")
    elif ny_hour < 9 or (ny_hour == 9 and utc.minute < 30):
        opens_in = (9 * 60 + 30) - (ny_hour * 60 + utc.minute)
        st.warning(f"🟡 Pre-Market — Opens in {opens_in}min · Showing last session data · {ny_str}")
    else:
        st.error(f"🔴 After-Market CLOSED · {ny_str}")

    st.caption("Hourly % returns · Green = up · Red = down · Intensity = magnitude")

    rows = []
    for label in [NAS100_LABEL] + MAG7:
        df = get_heatmap_data(label)
        if df is None or df.empty:
            continue
        try:
            df.columns = [c.capitalize() for c in df.columns]
            if 'Close' not in df.columns:
                continue
            closes      = df['Close'].dropna().tail(8)
            if len(closes) < 2:
                continue
            pct_changes = closes.pct_change().dropna() * 100
            row = {"Ticker": "NAS100(QQQ)" if label == NAS100_LABEL else label,
                   "Price":  round(float(closes.iloc[-1]), 2)}
            for j, (_, val) in enumerate(pct_changes.items()):
                row[f"H{j+1}"] = round(float(val), 2)
            row["Day %"] = round(float(
                (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100), 2)
            rows.append(row)
        except Exception as e:
            print(f"[heatmap] {label}: {e}")

    if not rows:
        st.warning("Heatmap data unavailable — market may be closed.")
        return

    df_hm     = pd.DataFrame(rows).set_index("Ticker")
    hour_cols = sorted([c for c in df_hm.columns if c.startswith("H")])
    for h in hour_cols:
        df_hm[h] = pd.to_numeric(df_hm[h], errors='coerce').fillna(0)
    df_hm["Day %"] = pd.to_numeric(df_hm["Day %"], errors='coerce').fillna(0)

    def color_cell(val):
        try:
            v = float(val)
        except:
            return ""
        if v > 1.5:    return "background-color:#1a7a1a;color:white"
        elif v > 0.5:  return "background-color:#2d9e2d;color:white"
        elif v > 0.1:  return "background-color:#5cb85c;color:white"
        elif v > -0.1: return "background-color:#888;color:white"
        elif v > -0.5: return "background-color:#d9534f;color:white"
        elif v > -1.5: return "background-color:#c9302c;color:white"
        else:          return "background-color:#8b0000;color:white"

    fmt = {"Price": "${:,.2f}", "Day %": "{:+.2f}%"}
    fmt.update({h: "{:+.2f}%" for h in hour_cols})

    styled = (
        df_hm[["Price", "Day %"] + hour_cols]
        .style.map(color_cell, subset=["Day %"] + hour_cols)
        .format(fmt)
    )
    st.dataframe(styled, use_container_width=True, height=320)
    st.divider()


# ── INDICATOR ENGINE ──────────────────────────────────────────────────────────
def compute_indicators(label: str):
    df_5m = get_5m(label)
    df_1h = get_1h(label)
    df_1d = get_1d(label)

    if df_5m is None or df_1h is None:
        return None
    if len(df_1h) < 200 or len(df_5m) < 20:
        return None

    sma200_1h    = df_1h['Close'].rolling(window=200).mean().iloc[-1]
    curr_p       = df_5m['Close'].iloc[-1]
    prev_p       = df_5m['Close'].iloc[-2]

    # Scale QQQ → NAS100 index price
    if label == NAS100_LABEL:
        ratio     = get_qqq_ndx_ratio()
        curr_p    = round(curr_p    * ratio, 0)
        prev_p    = round(prev_p    * ratio, 0)
        sma200_1h = round(sma200_1h * ratio, 0)

    trend_status = "BULLISH" if curr_p > sma200_1h else "BEARISH"
    trend_color  = "green"   if trend_status == "BULLISH" else "red"

    ema12        = df_1h['Close'].ewm(span=12, adjust=False).mean()
    ema26        = df_1h['Close'].ewm(span=26, adjust=False).mean()
    macd_line    = ema12 - ema26
    signal_line  = macd_line.ewm(span=9, adjust=False).mean()
    macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]

    delta_p    = df_5m['Close'].diff()
    gain       = (delta_p.where(delta_p > 0, 0)).rolling(window=14).mean()
    loss       = (-delta_p.where(delta_p < 0, 0)).rolling(window=14).mean()
    rs         = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    rsi        = rsi_series.iloc[-1]

    df_5m      = df_5m.copy()
    df_5m['vol_ma_long']  = df_5m['Volume'].rolling(window=20).mean()
    df_5m['vol_ma_short'] = df_5m['Volume'].rolling(window=5).mean()
    vol_ratio = (
        df_5m['vol_ma_short'].iloc[-1] / df_5m['vol_ma_long'].iloc[-1]
        if df_5m['vol_ma_long'].iloc[-1] > 0 else 1.0
    )

    start_price = df_5m['Open'].iloc[0]
    ann_vol     = df_5m['Close'].pct_change().std() * np.sqrt(252 * 78)
    delta_val   = get_bs_delta(curr_p, start_price, 1/252, 0.045, ann_vol)

    # RSI momentum
    rsi_prev   = rsi_series.iloc[-6] if len(rsi_series) >= 6 else rsi
    rsi_rising = rsi > rsi_prev
    rsi_falling= rsi < rsi_prev
    price_mom  = (curr_p - df_5m['Close'].iloc[-13]) / df_5m['Close'].iloc[-13] * 100 if len(df_5m) >= 13 else 0

    # Signal logic
    if curr_p > sma200_1h and rsi < 40 and vol_ratio > 1.1 and macd_bullish:
        signal    = "🟢 STRONG BUY — Dip (Full Confluence)"
        sig_color = "green"
    elif (curr_p > sma200_1h and rsi > 60 and rsi < 80
          and rsi_rising and vol_ratio > 1.3 and macd_bullish and price_mom > 0.3):
        signal    = "🚀 MOMENTUM BUY — Breakout"
        sig_color = "green"
    elif curr_p < sma200_1h and rsi > 60 and vol_ratio > 1.1 and not macd_bullish:
        signal    = "🔴 STRONG SELL (Full Confluence)"
        sig_color = "red"
    elif (curr_p < sma200_1h and rsi < 40 and rsi > 20
          and rsi_falling and vol_ratio > 1.3 and not macd_bullish and price_mom < -0.3):
        signal    = "💥 MOMENTUM SELL — Breakdown"
        sig_color = "red"
    elif curr_p > sma200_1h and rsi < 35:
        signal    = "🟡 Caution Buy (No MACD Confirm)"
        sig_color = "orange"
    elif curr_p > sma200_1h and rsi > 75 and vol_ratio < 0.8:
        signal    = "⚠️ Overbought — Watch for Reversal"
        sig_color = "orange"
    else:
        signal    = "⚪ Neutral"
        sig_color = "gray"

    iv = get_iv_data(label, curr_p, df_1d, vix_value)

    return dict(
        label=label, curr_p=curr_p, prev_p=prev_p,
        sma200_1h=sma200_1h, trend_status=trend_status, trend_color=trend_color,
        macd_bullish=macd_bullish, rsi=rsi, vol_ratio=vol_ratio,
        delta_val=delta_val, signal=signal, sig_color=sig_color, iv=iv,
    )


# ── TICKER CARD ───────────────────────────────────────────────────────────────
def render_ticker_card(ind: dict, col, risk_config: RiskConfig):
    with col:
        st.metric(
            label=ind["label"],
            value=f"${ind['curr_p']:,.2f}",
            delta=f"{ind['curr_p'] - ind['prev_p']:.2f}",
        )
        st.markdown(f"**Trend (1H SMA200):** :{ind['trend_color']}[{ind['trend_status']}]")
        st.markdown(f"**Signal:** :{ind['sig_color']}[{ind['signal']}]")

        with st.expander("Technical Details"):
            st.write(f"RSI (5m): {ind['rsi']:.1f}")
            st.write(f"Vol Surge: {ind['vol_ratio']:.2f}x")
            st.write(f"Opt. Delta: {ind['delta_val']:.2f}")
            st.write(f"MACD: {'Bullish 📈' if ind['macd_bullish'] else 'Bearish 📉'}")
            iv = ind.get('iv')
            if iv:
                st.markdown("---")
                src = {'vix_proxy':'VIX','options':'options','historical':'hist vol'}.get(iv.source, iv.source)
                st.markdown(
                    f"**IV:** <span style='color:{iv.iv_color};font-weight:bold'>"
                    f"{iv.current_iv:.1f}%</span> — **{iv.iv_label}** *({src})*",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns(2)
                c1.metric("IV Rank",       f"{iv.iv_rank:.0f}/100")
                c2.metric("IV Percentile", f"{iv.iv_percentile:.0f}%")
                st.progress(
                    int(min(iv.iv_rank, 100)) / 100,
                    text=f"IV Rank: {iv.iv_rank:.0f}% of 52w range"
                )

        if ind["signal"] != "⚪ Neutral":
            with st.expander("🤖 Claude AI Analysis", expanded=True):

                # ── SESSION GATE ──────────────────────────────────────────────
                session_ok, session_msg, prime_time = _claude_allowed_stocks()

                # Mid-session: only fire on STRONG/MOMENTUM signals
                if session_ok and not prime_time:
                    weak = any(w in ind["signal"] for w in ["Caution", "Overbought"])
                    if weak:
                        session_ok  = False
                        session_msg = "🟡 Mid-session — Claude fires on STRONG signals only"

                if not session_ok:
                    st.caption(session_msg)
                else:
                    trade_allowed, trade_reason = check_trade_allowed(risk_config, ind["label"])

                    iv     = ind.get('iv')
                    iv_str = (
                        f"{iv.current_iv:.1f}% (Rank {iv.iv_rank:.0f}/100, {iv.iv_label})"
                        if iv else "unavailable"
                    )

                    # Pass macro risk context to Claude
                    macro  = get_macro_snapshot()
                    macro_context = (
                        f"10Y Yield: {macro.yield_10y:.2f}% ({macro.yield_signal}) | "
                        f"Oil: ${macro.oil_price:.1f} ({macro.oil_signal}) | "
                        f"Breadth: {macro.breadth_signal} | "
                        f"Risk Score: {macro.risk_score}/100 — {macro.risk_level}"
                    ) if macro else "unavailable"

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
                        implied_volatility=iv_str,
                        macro_context=macro_context,
                    )

                    if ai is None:
                        st.warning("Claude analysis unavailable — check API key.")
                    else:
                        a_color = {"BUY":"green","SELL":"red","HOLD":"gray"}.get(ai.action,"gray")
                        st.markdown(
                            f"**Decision:** :{a_color}[{ai.action}] "
                            f"| Confidence: **{ai.confidence}**"
                        )
                        st.caption(f"📊 {ai.reasoning}")
                        if ai.sentiment_summary:
                            st.info(f"🗞️ {ai.sentiment_summary}")

                        if ai.action != "HOLD":
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Entry",       f"${ai.entry_price:,.2f}")
                            c2.metric("Stop Loss",   f"${ai.stop_loss:,.2f}",
                                      delta=f"{((ai.stop_loss-ai.entry_price)/ai.entry_price*100):+.2f}%",
                                      delta_color="off")
                            c3.metric("Take Profit", f"${ai.take_profit:,.2f}",
                                      delta=f"{((ai.take_profit-ai.entry_price)/ai.entry_price*100):+.2f}%",
                                      delta_color="off")

                            risk_usd   = abs(ai.entry_price - ai.stop_loss) * (ai.lot_size * 100)
                            reward_usd = abs(ai.take_profit - ai.entry_price) * (ai.lot_size * 100)
                            rr         = reward_usd / risk_usd if risk_usd > 0 else 0
                            st.caption(
                                f"Lot: {ai.lot_size} | Risk: ~${risk_usd:.2f} | "
                                f"Reward: ~${reward_usd:.2f} | R:R = {rr:.1f}:1"
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

render_heatmap()
render_macro_panel()

# NAS100
st.subheader("📈 Nasdaq 100 Cash CFD (NAS100)")
nas_col, _, _, _ = st.columns(4)
try:
    nas_ind = compute_indicators(NAS100_LABEL)
    if nas_ind:
        render_ticker_card(nas_ind, nas_col, risk_config)
    else:
        with nas_col:
            st.warning("NAS100 data unavailable")
except Exception as e:
    with nas_col:
        st.error(f"NAS100 error: {e}")

st.divider()

# Mag 7
st.subheader("🛡️ Magnificent 7 Stocks")
cols = st.columns(4)
for i, ticker in enumerate(MAG7):
    try:
        ind = compute_indicators(ticker)
        if ind:
            render_ticker_card(ind, cols[i % 4], risk_config)
    except Exception as e:
        with cols[i % 4]:
            st.error(f"Error {ticker}: {e}")
