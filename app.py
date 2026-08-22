import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timezone

from data_fetcher import fetch_all_data, get_5m, get_1h, get_1d, get_vix, get_heatmap_data, get_qqq_ndx_ratio, get_gold_df, get_macro_df, MAG7
from macro_monitor import render_macro_panel, get_macro_snapshot
from regime_detector import detect_regime_stocks, render_regime_panel, render_regime_badge
from scalping_engine import analyse_nas100_scalp
from iv_calculator import get_iv_data
from claude_analyst import analyse
from risk_manager import RiskConfig, init_risk_state, render_risk_sidebar, check_trade_allowed, record_trade_opened

# ── NEW ENGINES ───────────────────────────────────────────────────────────────
from options_intelligence import (
    get_oi_heatmap, get_gex, get_expected_move,
    render_oi_heatmap, render_gex_panel, render_expected_move_panel,
)
from breadth_quality import get_breadth_quality, render_breadth_quality_panel
from master_signal import compute_master_signal, render_master_signal
from nas100_breadth import (
    get_nas100_breadth, compute_harmonized_signal,
    render_nas100_breadth, render_harmonized_signal,
)
from qqq_intelligence import get_qqq_report, render_qqq_intelligence
from nq_futures import get_nq_report, render_nq_panel
from final_signal import compute_unified_signal, render_unified_signal
from order_flow_sequence import compute_order_flow_sequence, render_order_flow_sequence
from mean_reversion_atr import compute_mean_reversion_setup, render_mean_reversion_setup

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
    "5m Signals · 1H MACD · IV Analysis · Options Intelligence · Claude AI"
)

NAS100_LABEL = 'NAS100'


# ── SESSION GATE ──────────────────────────────────────────────────────────────
def _claude_allowed_stocks() -> tuple:
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

# ── MACRO SNAPSHOT (regime computed after nas_ind — BUG 2 FIX) ───────────────
_macro_snap    = get_macro_snapshot()
_gold_df       = get_gold_df()
_tnx_df        = get_macro_df("tnx")
_global_regime = None   # populated below using cached 5m + macro data

# ── COMPUTE GLOBAL REGIME (safe, at top, uses cached data) ───────────────────
try:
    _nas_5m_for_regime = get_5m(NAS100_LABEL)
    _nas_1h_for_regime = get_1h(NAS100_LABEL)

    # Extract live technicals from cached data — no new fetch needed
    _r_trend_bullish = True
    _r_macd_bullish  = True
    _r_rsi           = 50.0
    _r_price_vs_sma  = 1.0

    if _nas_1h_for_regime is not None and len(_nas_1h_for_regime) >= 200:
        _r_sma200 = float(_nas_1h_for_regime['Close'].rolling(200).mean().iloc[-1])
        _r_price  = float(_nas_1h_for_regime['Close'].iloc[-1])
        _r_trend_bullish = _r_price > _r_sma200
        _r_price_vs_sma  = (_r_price - _r_sma200) / _r_sma200 * 100 if _r_sma200 > 0 else 1.0

        _ema12 = _nas_1h_for_regime['Close'].ewm(span=12, adjust=False).mean()
        _ema26 = _nas_1h_for_regime['Close'].ewm(span=26, adjust=False).mean()
        _macd  = _ema12 - _ema26
        _sig   = _macd.ewm(span=9, adjust=False).mean()
        _r_macd_bullish = float(_macd.iloc[-1]) > float(_sig.iloc[-1])

    if _nas_5m_for_regime is not None and len(_nas_5m_for_regime) >= 15:
        _dp   = _nas_5m_for_regime['Close'].diff()
        _gain = _dp.where(_dp > 0, 0).rolling(14).mean()
        _loss = (-_dp.where(_dp < 0, 0)).rolling(14).mean()
        _rs   = _gain / _loss.replace(0, 1e-10)
        _r_rsi = float((100 - 100 / (1 + _rs)).iloc[-1])

    _global_regime = detect_regime_stocks(
        vix=vix_value,
        df_5m=_nas_5m_for_regime,
        trend_bullish=_r_trend_bullish,
        macd_bullish=_r_macd_bullish,
        rsi=_r_rsi,
        price_vs_sma_pct=_r_price_vs_sma,
        breadth_ratio=_macro_snap.breadth_ratio if _macro_snap else None,
        breadth_signal=_macro_snap.breadth_signal if _macro_snap else None,
        macro_risk_score=_macro_snap.risk_score if _macro_snap else None,
        gold_df=_gold_df,
        tnx_df=_tnx_df,
    )
except Exception as _regime_err:
    print(f"[regime] Error: {_regime_err}")
    # Safe fallback — use original simple call that always works
    try:
        _global_regime = detect_regime_stocks(
            vix=vix_value,
            macro_risk_score=_macro_snap.risk_score if _macro_snap else None,
            gold_df=_gold_df,
            tnx_df=_tnx_df,
        )
    except Exception:
        pass


# ── HEATMAP ───────────────────────────────────────────────────────────────────
def render_heatmap():
    st.subheader("📊 NAS100 + Mag 7 Hourly Heatmap")

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

    is_nas100    = (label == NAS100_LABEL)
    sma200_1h    = df_1h['Close'].rolling(window=200).mean().iloc[-1]
    curr_p       = df_5m['Close'].iloc[-1]
    prev_p       = df_5m['Close'].iloc[-2]

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

    rsi_prev    = rsi_series.iloc[-6] if len(rsi_series) >= 6 else rsi
    rsi_rising  = rsi > rsi_prev
    rsi_falling = rsi < rsi_prev

    if len(df_5m) >= 13:
        price_13_ago = df_5m['Close'].iloc[-13]
        price_mom = (df_5m['Close'].iloc[-1] - price_13_ago) / price_13_ago * 100
    else:
        price_mom = 0

    vol_thresh_s = 1.05 if is_nas100 else 1.1
    vol_thresh_m = 1.10 if is_nas100 else 1.3
    mom_thresh   = 0.10 if is_nas100 else 0.30

    if curr_p > sma200_1h and rsi < 40 and vol_ratio > vol_thresh_s and macd_bullish:
        signal    = "🟢 STRONG BUY — Dip (Full Confluence)"
        sig_color = "green"
    elif (curr_p > sma200_1h and rsi > 55 and rsi < 82
          and rsi_rising and vol_ratio > vol_thresh_m
          and macd_bullish and price_mom > mom_thresh):
        signal    = "🚀 MOMENTUM BUY — Breakout"
        sig_color = "green"
    elif (is_nas100 and curr_p > sma200_1h and macd_bullish
          and 45 < rsi < 75 and rsi_rising and price_mom > 0.05):
        signal    = "📈 TREND BUY — NAS100 Momentum"
        sig_color = "green"
    elif curr_p < sma200_1h and rsi > 60 and vol_ratio > vol_thresh_s and not macd_bullish:
        signal    = "🔴 STRONG SELL (Full Confluence)"
        sig_color = "red"
    elif (curr_p < sma200_1h and rsi < 40 and rsi > 20
          and rsi_falling and vol_ratio > vol_thresh_m
          and not macd_bullish and price_mom < -mom_thresh):
        signal    = "💥 MOMENTUM SELL — Breakdown"
        sig_color = "red"
    elif (is_nas100 and curr_p < sma200_1h and not macd_bullish
          and 25 < rsi < 55 and rsi_falling and price_mom < -0.05):
        signal    = "📉 TREND SELL — NAS100 Momentum"
        sig_color = "red"
    elif curr_p > sma200_1h and rsi < 35:
        signal    = "🟡 Caution Buy (No MACD Confirm)"
        sig_color = "orange"
    elif curr_p > sma200_1h and rsi > 78 and vol_ratio < 0.8:
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

                session_ok, session_msg, prime_time = _claude_allowed_stocks()

                if session_ok and not prime_time:
                    weak = any(w in ind["signal"] for w in ["Caution", "Overbought"])
                    if weak:
                        session_ok  = False
                        session_msg = "🟡 Mid-session — Claude fires on STRONG signals only"

                if not session_ok:
                    st.caption(session_msg)
                elif _global_regime is not None and _global_regime.state == 2:
                    st.error(f"🔴 {_global_regime.icon} CRISIS regime — no new entries. {_global_regime.strategy_note}")
                else:
                    trade_allowed, trade_reason = check_trade_allowed(risk_config, ind["label"])

                    iv     = ind.get('iv')
                    iv_str = (
                        f"{iv.current_iv:.1f}% (Rank {iv.iv_rank:.0f}/100, {iv.iv_label})"
                        if iv else "unavailable"
                    )

                    effective_lot = round(
                        risk_config.lot_size * (_global_regime.lot_multiplier if _global_regime else 1.0), 2
                    )
                    effective_lot = max(effective_lot, 0.01)

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
                        lot_size=effective_lot,
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

# ── SMART BREADTH QUALITY ENGINE ──────────────────────────────────────────────
_breadth_quality = None
try:
    _breadth_quality = get_breadth_quality()
    render_breadth_quality_panel(_breadth_quality)
except Exception as _bqe:
    st.warning(f"Breadth quality error: {_bqe}")

# ── NAS100 ────────────────────────────────────────────────────────────────────
st.subheader("📈 Nasdaq 100 Cash CFD (NAS100)")
nas_col, _, _, _ = st.columns(4)
nas_ind = None
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

# ── REGIME PANEL ─────────────────────────────────────────────────────────────
if _global_regime:
    try:
        render_regime_panel(_global_regime, "Market Regime Detector")
    except Exception as _re:
        st.warning(f"Regime panel error: {_re}")

# ── OPTIONS INTELLIGENCE PANEL (NAS100 only) ──────────────────────────────────
# ── MASTER SIGNAL ─────────────────────────────────────────────────────────────
# Run scalping now (needed by master signal) even before the scalp panel renders
_nas_5m_early = get_5m(NAS100_LABEL)
_nas_1d_early = get_1d(NAS100_LABEL)
_nas_ratio_ms = float(get_qqq_ndx_ratio() or 40.0)
_scalp_for_ms = None
if _nas_5m_early is not None and nas_ind:
    try:
        _scalp_for_ms = analyse_nas100_scalp(
            _nas_5m_early,
            _nas_1d_early if _nas_1d_early is not None else _nas_5m_early,
            float(nas_ind['curr_p']),
            qqq_to_nas100_ratio=_nas_ratio_ms,
        )
    except Exception:
        pass

# ── ORDER FLOW SEQUENCE + ATR MEAN REVERSION (feed Layers 8 & 9 below) ────────
_ofs_ms = None
_mr_ms  = None
if _nas_5m_early is not None and nas_ind:
    try:
        _ofs_ms = compute_order_flow_sequence(
            _nas_5m_early, _scalp_for_ms.cpr if _scalp_for_ms else None,
            float(nas_ind['curr_p']), ratio=_nas_ratio_ms, scalp_report=_scalp_for_ms,
        )
    except Exception:
        pass
    try:
        _mr_ms = compute_mean_reversion_setup(
            _nas_5m_early, _ofs_ms, _scalp_for_ms,
            float(nas_ind['curr_p']), ratio=_nas_ratio_ms,
        )
    except Exception:
        pass

_nas_price_ms = float(nas_ind['curr_p']) if nas_ind else None
_gex_ms       = None
_heatmap_ms   = None
_em_ms        = None
if _nas_price_ms:
    try:
        _gex_ms     = get_gex(_nas_price_ms, _nas_ratio_ms)
        _heatmap_ms = get_oi_heatmap(_nas_price_ms, _nas_ratio_ms)
        _open_ms    = (float(_nas_5m_early['Open'].iloc[0]) * _nas_ratio_ms
                       if _nas_5m_early is not None and len(_nas_5m_early) > 0
                       else _nas_price_ms)
        _em_ms      = get_expected_move(_nas_price_ms, _open_ms, _nas_ratio_ms)
    except Exception:
        pass

_master_sig = None
_qqq_report_ms = None
_nq_report = None
_nas100_breadth = None

# ── QQQ + NQ FETCH (shared across master + unified signal) ────────────────────
try:
    _qqq_report_ms = get_qqq_report()
except Exception as _qe:
    st.warning(f"QQQ data error: {_qe}")

try:
    _nq_report = get_nq_report(
        qqq_ratio=_nas_ratio_ms,
        qqq_5m_df=get_5m(NAS100_LABEL),
    )
except Exception:
    pass   # _nq_report stays None — render_nq_panel handles None gracefully

# ── NAS100 COMPONENT BREADTH (15-min cache) ───────────────────────────────────
try:
    _nas100_breadth = get_nas100_breadth()
    render_nas100_breadth(_nas100_breadth)
except Exception as _bre:
    st.warning(f"NAS100 breadth error: {_bre}")

# ── MASTER SIGNAL (legacy — kept for layer scoring, not rendered) ─────────────
try:
    _master_sig = compute_master_signal(
        ind=nas_ind,
        macro_snap=_macro_snap,
        regime=_global_regime,
        gex=_gex_ms,
        heatmap=_heatmap_ms,
        expected_move=_em_ms,
        breadth_quality=_breadth_quality,
        scalp_report=_scalp_for_ms,
        qqq_report=_qqq_report_ms,
        nq_report=_nq_report,
        ofs=_ofs_ms,
        mr_setup=_mr_ms,
    )
except Exception as _mse:
    pass   # master signal feeds unified — failure degrades gracefully

# ── NQ FUTURES PANEL ──────────────────────────────────────────────────────────
try:
    if _nq_report is None:
        # get_nq_report raised before returning — create a safe unavailable report
        from nq_futures import NQReport as _NQReport
        _nq_report = _NQReport(
            price=None, volume=None, leadership=None,
            displacement=None, basis=None, score=None,
            fetched_at=pd.Timestamp.now().strftime("%H:%M:%S"),
            available=False,
        )
    qqq_intraday_for_nq = _qqq_report_ms.intraday if _qqq_report_ms else None
    render_nq_panel(_nq_report, qqq_intraday=qqq_intraday_for_nq)
except Exception as _nqre:
    st.subheader("📊 NQ Futures vs QQQ — Institutional Confirmation Engine")
    st.info(
        "NQ Futures data unavailable — market may be closed or "
        "NQ=F not accessible via yfinance at this time. "
        "All other dashboard signals remain active."
    )

st.divider()

# ── UNIFIED HARMONIZED FINAL SIGNAL ──────────────────────────────────────────
try:
    _unified = compute_unified_signal(
        nas100_ind=nas_ind,
        breadth_quality=_breadth_quality,
        macro_snap=_macro_snap,
        nas100_breadth=_nas100_breadth,
        gex=_gex_ms,
        heatmap=_heatmap_ms,
        expected_move=_em_ms,
        scalp_report=_scalp_for_ms,
        qqq_report=_qqq_report_ms,
        nq_report=_nq_report,
        regime=_global_regime,
        vix_value=vix_value,
        risk_config=risk_config,
    )
    render_unified_signal(_unified, risk_config)
except Exception as _use:
    st.warning(f"Unified signal error: {_use}")

st.divider()

if nas_ind:
    st.subheader("🧮 Options Intelligence — NAS100")
    _nas_price = float(nas_ind['curr_p'])
    _nas_ratio = float(get_qqq_ndx_ratio() or 40.0)

    oi_col, gex_col, em_col = st.columns(3)

    # OI Heatmap — reuse cached fetch from master signal
    with oi_col:
        try:
            _heatmap = _heatmap_ms or get_oi_heatmap(_nas_price, _nas_ratio)
            if _heatmap:
                render_oi_heatmap(_heatmap)
            else:
                st.caption("OI heatmap unavailable (market closed or data error)")
        except Exception as _e:
            st.caption(f"OI heatmap error: {_e}")

    # GEX — reuse cached fetch
    with gex_col:
        try:
            _gex = _gex_ms or get_gex(_nas_price, _nas_ratio)
            if _gex:
                render_gex_panel(_gex)
            else:
                st.caption("GEX unavailable")
        except Exception as _e:
            st.caption(f"GEX error: {_e}")

    # Expected Move — reuse cached fetch
    with em_col:
        try:
            _em = _em_ms or get_expected_move(_nas_price, _nas_price, _nas_ratio)
            if _em:
                render_expected_move_panel(_em)
            else:
                st.caption("Expected move unavailable")
        except Exception as _e:
            st.caption(f"Expected move error: {_e}")

    st.divider()

# ── NAS100 SCALPING PANEL ─────────────────────────────────────────────────────
st.subheader("🎯 NAS100 Sniper Scalping")
_nas_5m = get_5m(NAS100_LABEL)
_nas_1d = get_1d(NAS100_LABEL)

if _nas_5m is not None and nas_ind:
    try:
        _nas_price = float(nas_ind['curr_p'])
        _nas_ratio = float(get_qqq_ndx_ratio() or 40.0)
        # Reuse scalp already computed for master signal (same refresh cycle)
        _nas_scalp = _scalp_for_ms or analyse_nas100_scalp(
            _nas_5m,
            _nas_1d if _nas_1d is not None else _nas_5m,
            _nas_price,
            qqq_to_nas100_ratio=_nas_ratio,
        )

        sc1, sc2, sc3, sc4 = st.columns(4)

        with sc1:
            st.markdown("**📊 VWAP**")
            if _nas_scalp.vwap:
                vwap_col = "#2d9e2d" if _nas_price > _nas_scalp.vwap else "#c9302c"
                st.markdown(
                    f"<span style='font-size:1.2em;font-weight:bold;color:{vwap_col}'>"
                    f"{_nas_scalp.vwap:,.0f}</span>",
                    unsafe_allow_html=True
                )
                st.caption(f"Dev: {_nas_scalp.vwap_deviation_pct:+.2f}%")
                if _nas_scalp.vwap_setup:
                    vs = _nas_scalp.vwap_setup
                    d_col = "#2d9e2d" if vs.direction == "BUY" else "#c9302c"
                    st.markdown(
                        f"<span style='color:{d_col};font-weight:bold'>"
                        f"⚡ VWAP {vs.direction} — {vs.strength}</span>",
                        unsafe_allow_html=True
                    )
                    st.caption(vs.description[:100])
            else:
                st.caption("VWAP unavailable (pre-market or no volume)")

        with sc2:
            st.markdown("**📐 Gap Fill**")
            if _nas_scalp.gap_fill_setup:
                gs    = _nas_scalp.gap_fill_setup
                d_col = "#2d9e2d" if gs.direction == "BUY" else "#c9302c"
                st.markdown(
                    f"<span style='color:{d_col};font-weight:bold'>"
                    f"{gs.direction} → {gs.target:,.0f}</span>",
                    unsafe_allow_html=True
                )
                st.caption(gs.description[:100])
                st.caption(f"R:R {gs.pips_to_target/gs.risk_pips:.1f}:1 | "
                           f"Strength: {gs.strength}")
            else:
                st.caption("No significant gap today")

        with sc3:
            st.markdown("**🔑 Key Levels**")
            if _nas_scalp.key_levels:
                near = sorted(_nas_scalp.key_levels,
                              key=lambda l: abs(l - _nas_price))[:4]
                for lv in near:
                    dist  = lv - _nas_price
                    color = "#2d9e2d" if dist > 0 else "#c9302c"
                    st.markdown(
                        f"<span style='color:{color}'>"
                        f"{lv:,.0f} ({dist:+.0f} pts)</span>",
                        unsafe_allow_html=True
                    )
            if _nas_scalp.key_level_setup:
                kls = _nas_scalp.key_level_setup
                st.warning(f"⚡ AT KEY LEVEL — {kls.direction} bounce")

        with sc4:
            st.markdown("**🚀 Open Drive**")
            drive_colors = {
                "BULLISH":  "#2d9e2d", "BEARISH":  "#c9302c",
                "CHOPPY":   "#e6a817", "PRE-OPEN": "#888888",
                "AWAITING": "#888888", "UNKNOWN":  "#888888",
            }
            drive = _nas_scalp.open_drive or "UNKNOWN"
            color = drive_colors.get(drive, "#888888")
            st.markdown(
                f"<span style='font-size:1.3em;font-weight:bold;color:{color}'>"
                f"{drive}</span>",
                unsafe_allow_html=True
            )
            st.caption("First 15min NY direction bias")
            if drive == "BULLISH":
                st.success("Favour LONG scalps with VWAP support")
            elif drive == "BEARISH":
                st.error("Favour SHORT scalps below VWAP")
            elif drive == "CHOPPY":
                st.warning("Avoid momentum — range scalp only")

        # ── ENHANCED LIQUIDITY SWEEPS SUB-PANEL ──────────────────────────────
        if _nas_scalp.liquidity_sweeps or _nas_scalp.active_fade_setup:
            st.markdown("---")
            st.markdown("**🎣 Liquidity Sweep Detector**")

            if _nas_scalp.active_fade_setup:
                fade = _nas_scalp.active_fade_setup
                fade_color = "#2d9e2d" if fade.fade_direction == "BUY" else "#c9302c"
                st.markdown(
                    f"<div style='padding:8px;border-radius:6px;"
                    f"background:{fade_color}22;border-left:3px solid {fade_color}'>"
                    f"<span style='color:{fade_color};font-weight:bold'>"
                    f"{'🟢' if fade.fade_direction == 'BUY' else '🔴'} "
                    f"FADE {fade.fade_direction} — {fade.sweep_type.replace('_', ' ')}</span><br>"
                    f"<span style='font-size:0.9em'>{fade.signal_text}</span><br>"
                    f"<span style='color:#aaa;font-size:0.8em'>"
                    f"Swept level: {fade.swept_level:,.0f} | "
                    f"Sweep size: {fade.sweep_size_pts:.0f} pts | "
                    f"RVOL: {fade.rvol_ratio:.1f}x"
                    f"{'  ⚡ VOLUME CONFIRMED' if fade.rvol_spike else ''}"
                    f"</span></div>",
                    unsafe_allow_html=True
                )
            elif _nas_scalp.liquidity_sweeps:
                for sw in _nas_scalp.liquidity_sweeps[:2]:
                    st.caption(f"• {sw.signal_text[:120]}")

        # ── CPR PANEL ─────────────────────────────────────────────────────────
        if _nas_scalp.cpr:
            cpr = _nas_scalp.cpr
            st.markdown("---")
            st.markdown("**📐 Central Pivot Range (CPR)**")

            # CPR type banner
            cpr_banner_colors = {
                "NARROW":   ("#2d9e2d", "🎯"),
                "MODERATE": ("#e6a817", "⚖️"),
                "WIDE":     ("#c9302c", "🌊"),
            }
            b_color, b_icon = cpr_banner_colors.get(cpr.cpr_type, ("#888", "📐"))
            virgin_badge = (
                " &nbsp;|&nbsp; <span style='color:#aa44ff'>🔮 VIRGIN CPR</span>"
                if cpr.virgin else ""
            )
            st.markdown(
                f"<div style='padding:8px 12px;border-radius:6px;"
                f"background:{b_color}22;border-left:3px solid {b_color};"
                f"margin-bottom:8px'>"
                f"<span style='color:{b_color};font-weight:bold'>"
                f"{b_icon} {cpr.cpr_type} CPR</span>"
                f"<span style='color:#aaa;font-size:0.88em;margin-left:10px'>"
                f"Width: {cpr.cpr_width:.0f} pts ({cpr.cpr_width_pct:.2f}%)"
                f"{virgin_badge}</span><br>"
                f"<span style='color:#ccc;font-size:0.85em'>{cpr.cpr_type_bias}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Levels grid
            cpr_c1, cpr_c2, cpr_c3 = st.columns(3)

            with cpr_c1:
                st.markdown("**Key CPR Levels**")
                for label, val in [
                    ("R2", cpr.r2), ("R1", cpr.r1),
                    ("TC", cpr.tc), ("Pivot", cpr.pivot), ("BC", cpr.bc),
                    ("S1", cpr.s1), ("S2", cpr.s2),
                ]:
                    dist  = val - _nas_price
                    is_tc = label == "TC"
                    is_bc = label == "BC"
                    is_p  = label == "Pivot"
                    color = (
                        "#aa44ff" if (is_tc or is_bc or is_p) else
                        "#c9302c" if dist > 0 else "#2d9e2d"
                    )
                    bold = "font-weight:bold" if (is_tc or is_bc or is_p) else ""
                    st.markdown(
                        f"<span style='color:{color};{bold}'>"
                        f"{label}: {val:,.0f} "
                        f"<span style='color:#666;font-size:0.8em'>"
                        f"({dist:+.0f} pts)</span></span>",
                        unsafe_allow_html=True,
                    )

            with cpr_c2:
                st.markdown("**Price vs CPR**")
                pos_colors = {
                    "ABOVE_TC": "#2d9e2d",
                    "INSIDE":   "#e6a817",
                    "BELOW_BC": "#c9302c",
                }
                pos_labels = {
                    "ABOVE_TC": "▲ Above TC — Bullish",
                    "INSIDE":   "◆ Inside CPR — Indecision",
                    "BELOW_BC": "▼ Below BC — Bearish",
                }
                pc = cpr.price_vs_cpr
                st.markdown(
                    f"<span style='color:{pos_colors.get(pc,'#888')};"
                    f"font-size:1.05em;font-weight:bold'>"
                    f"{pos_labels.get(pc, pc)}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"TC: {cpr.tc:,.0f} | Pivot: {cpr.pivot:,.0f} | BC: {cpr.bc:,.0f}")
                if cpr.virgin:
                    st.markdown(
                        "<span style='color:#aa44ff;font-weight:bold'>"
                        "🔮 Virgin CPR — untested magnet zone</span>",
                        unsafe_allow_html=True,
                    )

            with cpr_c3:
                st.markdown("**CPR Signal**")
                if cpr.setup:
                    s = cpr.setup
                    s_col = "#2d9e2d" if s.direction == "BUY" else "#c9302c"
                    st.markdown(
                        f"<span style='color:{s_col};font-weight:bold'>"
                        f"{'🟢' if s.direction == 'BUY' else '🔴'} "
                        f"CPR {s.direction} — {s.strength}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(s.description[:130])
                    c_a, c_b = st.columns(2)
                    c_a.metric("Target",      f"{s.target:,.0f}")
                    c_b.metric("Invalidation",f"{s.invalidation:,.0f}")
                elif cpr.setup_description:
                    st.caption(cpr.setup_description)
                else:
                    st.caption("No active CPR signal at current price.")

        # ── ORDER FLOW SEQUENCE (additive) ──────────────────────────────────
        st.markdown("---")
        try:
            _ofs = compute_order_flow_sequence(
                _nas_5m, _nas_scalp.cpr, _nas_price, ratio=_nas_ratio,
                scalp_report=_nas_scalp,
            )
            render_order_flow_sequence(_ofs)
        except Exception as _ofse:
            st.caption(f"Order flow sequence unavailable: {_ofse}")
            _ofs = None

        # ── ATR MEAN REVERSION (additive) ────────────────────────────────────
        st.markdown("---")
        try:
            _mr = compute_mean_reversion_setup(
                _nas_5m, _ofs, _nas_scalp, _nas_price, ratio=_nas_ratio,
            )
            render_mean_reversion_setup(_mr)
        except Exception as _mre:
            st.caption(f"Mean reversion setup unavailable: {_mre}")

    except Exception as _se:
        st.warning(f"NAS100 scalping unavailable: {_se}")
else:
    st.caption("NAS100 data not yet loaded.")

st.divider()

# ── QQQ ETF INTELLIGENCE ──────────────────────────────────────────────────────
try:
    _qqq_report = get_qqq_report()
    render_qqq_intelligence(_qqq_report)
except Exception as _qqq_err:
    st.warning(f"QQQ Intelligence error: {_qqq_err}")

st.divider()

# ── MAG 7 ─────────────────────────────────────────────────────────────────────
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
