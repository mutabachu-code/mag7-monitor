import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from streamlit_autorefresh import st_autorefresh

from forex_data_fetcher import fetch_all_pairs, get_15m, get_1h, get_4h, get_1d, get_pip, PAIRS, get_fx_gold_df, get_fx_macro
from forex_volume_profile import compute_volume_profile
from forex_analyst import analyse_pair, FXSignal
from macro_monitor import get_macro_snapshot
from regime_detector import detect_regime_forex, render_regime_panel, render_regime_badge
from scalping_engine import analyse_forex_scalp, ScalpSetup
from cvd_calculator import calculate_cvd, render_cvd_badge, render_cvd_panel

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FX Major Pairs Monitor", layout="wide", page_icon="💱")
st_autorefresh(interval=60000, key="fx_refresh")


# ── SESSION GATE ──────────────────────────────────────────────────────────────
def _claude_allowed_forex(pair: str = "") -> tuple:
    """
    Gate Claude to London Open + London/NY Overlap only.
    During NY-only session, only USD pairs get analysed.
    Returns (allowed: bool, reason: str)
    """
    utc  = datetime.now(timezone.utc)
    h    = utc.hour
    wday = utc.weekday()

    if wday >= 5:
        return False, "🔴 Forex weekend — Claude resumes Monday 07:00 UTC"
    if 7 <= h < 12:
        return True,  "🟢 London Open — Claude ACTIVE"
    if 12 <= h < 17:
        return True,  "🟢 London/NY Overlap — Claude ACTIVE (peak liquidity)"
    if 17 <= h < 21:
        if pair and 'USD' not in pair:
            return False, f"🟡 NY Session — skipping {pair} (not a USD pair)"
        return True, "🟡 NY Session — USD pairs only"
    if h < 7:
        mins = (7 - h) * 60 - utc.minute
        return False, f"🕐 Asian Session — Claude activates at London Open in {mins}min (07:00 UTC)"
    return False, "🔴 After-hours — Claude resumes at London Open (07:00 UTC)"


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ FX Risk Controls")

for key, default in [
    ("fx_kill", False), ("fx_daily_pnl", 0.0),
    ("fx_trades_today", 0), ("fx_open_pair", None), ("fx_trade_log", [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.fx_kill:
    st.sidebar.error("🔴 KILL SWITCH: ON")
    if st.sidebar.button("🟢 Resume Trading", type="primary"):
        st.session_state.fx_kill = False; st.rerun()
else:
    if st.session_state.fx_open_pair:
        st.sidebar.warning(f"📊 Open: **{st.session_state.fx_open_pair}**")
    else:
        st.sidebar.success("🟢 Bot ACTIVE — No open position")
    if st.sidebar.button("🔴 HALT Trading", type="secondary"):
        st.session_state.fx_kill = True; st.rerun()

st.sidebar.divider()
pnl = st.session_state.fx_daily_pnl
st.sidebar.markdown(f"**Daily P&L:** :{'green' if pnl >= 0 else 'red'}[${pnl:+.2f}]")
st.sidebar.markdown(f"**Trades today:** {st.session_state.fx_trades_today} / 3")

st.sidebar.divider()
account_size = st.sidebar.number_input("Account size (USD)", 10.0, 10000.0, 100.0, 10.0)
lot_size     = st.sidebar.select_slider("Lot size", [0.01, 0.02, 0.03, 0.05], 0.02)
daily_limit  = st.sidebar.slider("Daily loss limit (%)", 1.0, 10.0, 5.0, 0.5)

if st.session_state.fx_trade_log:
    st.sidebar.divider()
    st.sidebar.markdown("**Today's trades**")
    st.sidebar.dataframe(
        pd.DataFrame(st.session_state.fx_trade_log),
        hide_index=True, use_container_width=True
    )

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("💱 FX Major Pairs — Volume Profile + Claude AI")
st.caption(
    f"Last Update: {pd.Timestamp.now().strftime('%H:%M:%S')} | "
    "Volume Profile (1H) · POC/VAH/VAL · Mean Reversion & Breakout · Claude Sentiment"
)

# ── FOREX SESSION STATUS ──────────────────────────────────────────────────────
utc      = datetime.now(timezone.utc)
utc_h    = utc.hour
utc_wday = utc.weekday()
utc_str  = utc.strftime("%H:%M UTC")

if utc_wday >= 5:
    st.error(f"🔴 FOREX CLOSED (Weekend) · {utc_str}")
elif 22 <= utc_h or utc_h < 7:
    st.warning(f"🟡 Asian Session · {utc_str} — Low liquidity on majors")
elif 7 <= utc_h < 12:
    st.success(f"🟢 London Session OPEN · {utc_str}")
elif 12 <= utc_h < 17:
    st.success(f"🟢 London + NY Overlap (PEAK LIQUIDITY) · {utc_str}")
elif 17 <= utc_h < 21:
    st.warning(f"🟡 New York Session · {utc_str}")
else:
    st.warning(f"🟡 Market transitioning · {utc_str}")

# ── 10Y YIELD PANEL ─────────────────────────────────────────────────────────
macro = get_macro_snapshot()
if macro:
    yield_col, oil_col, breadth_col, score_col = st.columns(4)
    with yield_col:
        st.metric("🏦 US 10Y Yield", f"{macro.yield_10y:.2f}%",
                  delta=f"{macro.yield_signal}", delta_color="off")
    with oil_col:
        st.metric("🛢️ Brent Crude", f"${macro.oil_price:.1f}",
                  delta=macro.oil_signal, delta_color="off")
    with breadth_col:
        st.metric("📊 Breadth", f"QQQ vs QQQE",
                  delta=macro.breadth_signal.split("—")[0].strip(), delta_color="off")
    with score_col:
        st.metric("⚡ Risk Score", f"{macro.risk_score}/100",
                  delta=macro.risk_icon + " " + macro.risk_level.split(":")[0],
                  delta_color="off")

    # Yield filter warning for forex
    if macro.yield_10y > 4.50:
        st.warning(
            f"⚠️ **Yield Trap Active** ({macro.yield_10y:.2f}%) — "
            "Mean Reversion BUY signals on USD pairs may be unreliable. "
            "Yields at this level support USD strength — favour BREAKOUT BUY on USDJPY/USDCHF."
        )
    if "EXHAUSTION" in macro.breadth_signal:
        st.error("🔴 Market Breadth Exhaustion detected — risk-off environment. Favour CHF/JPY safety flows.")

st.divider()

# ── FETCH DATA ────────────────────────────────────────────────────────────────
with st.spinner("Fetching forex data..."):
    data_ok = fetch_all_pairs()

if not data_ok:
    st.error("⚠️ Forex data unavailable. Retrying on next refresh.")
    st.stop()

# ── GLOBAL FOREX REGIME ───────────────────────────────────────────────────────
_fx_macro  = get_macro_snapshot()
_fx_gold   = get_fx_gold_df()
_fx_tnx    = get_fx_macro("tnx")
_fx_jpy    = get_fx_macro("usdjpy") if hasattr(get_fx_macro, "__call__") else None

# Use USDJPY 1h as JPY proxy for cross-asset check
from forex_data_fetcher import get_1h as fx_get_1h
_fx_jpy_df = fx_get_1h("USDJPY")

_global_fx_regime = detect_regime_forex(
    vix=_fx_macro.yield_10y if _fx_macro else None,
    macro_risk_score=_fx_macro.risk_score if _fx_macro else None,
    gold_df=_fx_gold,
    jpy_df=_fx_jpy_df,
    tnx_df=_fx_tnx,
)

# ── CORRELATION MATRIX ────────────────────────────────────────────────────────
render_regime_panel(_global_fx_regime, "Forex Market Regime")

def render_correlation_matrix():
    st.subheader("🔗 Correlation Matrix — Lead/Lag Analysis")
    st.caption("Values near +1.0 = highly correlated · Near -1.0 = inverse · Find the 'Leader' pair")

    closes = {}
    for pair in PAIRS:
        df = get_1h(pair)
        if df is not None and not df.empty:
            df.columns = [c.capitalize() for c in df.columns]
            closes[pair] = df['Close'].tail(48)

    if len(closes) < 2:
        st.warning("Insufficient data for correlation matrix.")
        return

    df_closes = pd.DataFrame(closes).dropna()
    corr      = df_closes.pct_change().dropna().corr().round(2)

    def color_corr(val):
        try:
            v = float(val)
            if v == 1.0:   return "background-color:#333;color:#333"
            elif v > 0.7:  return "background-color:#1a5c1a;color:white"
            elif v > 0.3:  return "background-color:#2d7a2d;color:white"
            elif v > -0.3: return "background-color:#555;color:white"
            elif v > -0.7: return "background-color:#7a2d2d;color:white"
            else:           return "background-color:#5c1a1a;color:white"
        except:
            return ""

    st.dataframe(
        corr.style.map(color_corr).format("{:.2f}"),
        use_container_width=True
    )
    st.caption("🟢 Strong positive (move together) · 🔴 Strong negative (inverse pairs)")
    st.divider()


render_correlation_matrix()

# ── VOLUME PROFILE SUMMARY TABLE ──────────────────────────────────────────────
st.subheader("📊 Volume Profile Overview")
st.caption("POC = Fair Value · VAH/VAL = Value Area edges · Signal colour = trade type")

vp_data      = {}
summary_rows = []

for pair in PAIRS:
    vp = compute_volume_profile(pair, get_1h(pair), get_4h(pair), get_1d(pair))
    if vp:
        vp_data[pair] = vp
        summary_rows.append({
            "Pair":     pair,
            "Price":    f"{vp.current_price:.5f}",
            "Day %":    f"{vp.day_pct:+.2f}%",
            "POC":      f"{vp.poc:.5f}",
            "VAH":      f"{vp.vah:.5f}",
            "VAL":      f"{vp.val:.5f}",
            "Location": vp.price_location.replace("_", " "),
            "RSI":      f"{vp.rsi_1h:.0f}",
            "ATR%":     f"{vp.atr_pct*100:.3f}%",
            "Signal":   f"{vp.signal_icon} {vp.signal}",
        })

if summary_rows:
    df_summary = pd.DataFrame(summary_rows).set_index("Pair")

    def color_signal(val):
        v = str(val)
        if "MEAN REVERSION BUY"  in v: return "background-color:#1a3a5c;color:#88bbff"
        if "MEAN REVERSION SELL" in v: return "background-color:#1a3a5c;color:#88bbff"
        if "MEAN REVERSION"      in v: return "background-color:#1a3a5c;color:#88bbff"
        if "BREAKOUT BUY"        in v: return "background-color:#1a4a1a;color:#88ff88"
        if "BREAKOUT SELL"       in v: return "background-color:#4a1a1a;color:#ff8888"
        if "HIGH RISK"           in v: return "background-color:#333;color:#888"
        if "INTERVENTION"        in v: return "background-color:#4a0000;color:#ff4444"
        if "POC LEVEL"           in v: return "background-color:#3a2a00;color:#ffcc44"
        return ""

    def color_day(val):
        try:
            v = float(str(val).replace('%', ''))
            if v > 0.3:    return "color:#44ff44;font-weight:bold"
            elif v < -0.3: return "color:#ff4444;font-weight:bold"
        except:
            pass
        return ""

    st.dataframe(
        df_summary.style
        .map(color_signal, subset=["Signal"])
        .map(color_day,    subset=["Day %"]),
        use_container_width=True, height=300
    )

st.divider()

# ── PAIR CARDS ────────────────────────────────────────────────────────────────
st.subheader("🔍 Pair Detail + Claude AI Analysis")
cols = st.columns(3)

for idx, pair in enumerate(PAIRS):
    vp = vp_data.get(pair)
    with cols[idx % 3]:
        if vp is None:
            st.warning(f"{pair}: Data unavailable")
            st.divider()
            continue

        # Price header
        pip        = get_pip(pair)
        delta_pips = (vp.current_price - vp.prev_price) / pip
        st.metric(
            label=f"💱 {pair}",
            value=f"{vp.current_price:.5f}",
            delta=f"{delta_pips:+.1f} pips",
        )

        trend_tag  = "🟢 BULLISH" if vp.trend_bullish   else "🔴 BEARISH"
        trend4h    = "🟢 Bull"    if vp.trend_4h_bullish else "🔴 Bear"
        st.markdown(f"**Trend:** {trend_tag} (Daily) · {trend4h} (4H)")

        sig_color_map = {
            "MEAN REVERSION BUY":  "blue",
            "MEAN REVERSION SELL": "blue",
            "MEAN REVERSION":      "blue",
            "BREAKOUT BUY":        "green",
            "BREAKOUT SELL":       "red",
            "HIGH RISK":           "gray",
            "INTERVENTION RISK":   "red",
            "POC LEVEL":           "orange",
        }
        sc = "gray"
        for k, v in sig_color_map.items():
            if k in vp.signal: sc = v; break
        st.markdown(f"**Signal:** :{sc}[{vp.signal_icon} {vp.signal}]")
        # Per-pair regime
        _pair_regime = detect_regime_forex(
            vix=None, atr_pct=vp.atr_pct, pair=pair,
            trend_bullish=vp.trend_bullish, rsi=vp.rsi_1h,
            vol_intensity=vp.volume_intensity,
            macro_risk_score=_fx_macro.risk_score if _fx_macro else None,
            gold_df=_fx_gold, jpy_df=_fx_jpy_df, tnx_df=_fx_tnx,
        )
        render_regime_badge(_pair_regime)

        # CVD + Institutional move detection
        _cvd_df  = get_15m(pair) if get_15m(pair) is not None else get_1h(pair)
        _cvd     = calculate_cvd(_cvd_df, lookback=20)
        render_cvd_badge(_cvd)

        # Institutional move alert
        if vp.inst_move and vp.inst_direction != "NONE":
            _inst_col = "#2d9e2d" if vp.inst_direction == "BUY" else "#c9302c"
            st.markdown(
                f"<div style='padding:3px 8px;border-radius:4px;"
                f"background:{_inst_col}22;border:1px solid {_inst_col};"
                f"font-size:0.82em;margin:2px 0'>"
                f"<b style='color:{_inst_col}'>🏦 INSTITUTIONAL {vp.inst_direction} "
                f"detected — large candle (1.5x ATR)</b></div>",
                unsafe_allow_html=True
            )

        # ── SCALPING SETUPS — always visible, uses 15m for precision ──────────
        _df_15m = get_15m(pair)
        _df_1h  = get_1h(pair)
        # Use 15m if available, fall back to 1h
        _scalp_df = _df_15m if _df_15m is not None and len(_df_15m) >= 20 else _df_1h
        _scalp = analyse_forex_scalp(pair, _scalp_df, vp.current_price)

        with st.expander("🎯 Scalping Setups", expanded=False):
            icons = {"ORDER_BLOCK":"🟦 OB","FVG":"⬜ FVG","LIQ_SWEEP":"💧 SWEEP"}
            tf_label = "15m" if (_df_15m is not None and len(_df_15m) >= 20) else "1H"
            st.caption(f"Scanning {tf_label} chart · Session: {_scalp.session}")

            # ── Asian Range ───────────────────────────────────────────────────
            ar = _scalp.asian_range
            if ar:
                ar_col1, ar_col2 = st.columns(2)
                ar_col1.metric("Asian High", f"{ar['high']:.5f}")
                ar_col2.metric("Asian Low",  f"{ar['low']:.5f}")
                range_color = "green" if ar['breakout_up'] else "red" if ar['breakout_down'] else "gray"
                st.markdown(f"**Range:** :{range_color}[{ar['status']}] · {ar['range_pips']:.0f} pips")
            else:
                st.caption("Asian range: calculating...")

            st.markdown("---")

            # ── Count all setups ──────────────────────────────────────────────
            total_ob    = len(_scalp.order_blocks)
            total_fvg   = len(_scalp.fvgs)
            total_sweep = len(_scalp.liq_sweeps)
            total_all   = total_ob + total_fvg + total_sweep

            if total_all == 0:
                st.caption("📭 No active scalping setups detected. "
                           "Price not near any OB, FVG, or sweep level.")
            else:
                st.caption(f"📋 Found: {total_ob} Order Block · "
                           f"{total_fvg} FVG · {total_sweep} Liquidity Sweep")

            # ── Best Setup ───────────────────────────────────────────────────
            bs = _scalp.best_setup
            if bs:
                d_col  = "#2d9e2d" if bs.direction == "BUY" else "#c9302c"
                s_icon = icons.get(bs.setup_type, "🎯")
                rr     = bs.pips_to_target / bs.risk_pips if bs.risk_pips > 0 else 0
                st.markdown(
                    f"<div style='padding:8px;border-radius:6px;"
                    f"background:{d_col}22;border:1px solid {d_col};margin:4px 0'>"
                    f"<b style='color:{d_col}'>{s_icon} {bs.direction}</b> · "
                    f"{bs.strength} · R:R {rr:.1f}:1"
                    f"</div>",
                    unsafe_allow_html=True
                )
                st.caption(bs.description)
                z1, z2, z3 = st.columns(3)
                z1.metric("Entry Zone",   f"{bs.entry_zone_low:.5f}")
                z2.metric("Target",       f"{bs.target:.5f}",
                          delta=f"+{bs.pips_to_target:.0f} pips")
                z3.metric("Invalidation", f"{bs.invalidation:.5f}",
                          delta=f"-{bs.risk_pips:.0f} pips", delta_color="inverse")

            # ── All Order Blocks ──────────────────────────────────────────────
            if _scalp.order_blocks:
                st.markdown("**🟦 Order Blocks**")
                for ob in _scalp.order_blocks:
                    d_col = "#2d9e2d" if ob.direction == "BUY" else "#c9302c"
                    rr    = ob.pips_to_target / ob.risk_pips if ob.risk_pips > 0 else 0
                    st.markdown(
                        f"<span style='color:{d_col}'>{'▲' if ob.direction=='BUY' else '▼'} "
                        f"{ob.direction}</span> · Zone: `{ob.entry_zone_low:.5f}`–`{ob.entry_zone_high:.5f}` · "
                        f"Target: `{ob.target:.5f}` · R:R {rr:.1f}:1 · {ob.strength}",
                        unsafe_allow_html=True
                    )

            # ── All FVGs ─────────────────────────────────────────────────────
            if _scalp.fvgs:
                st.markdown("**⬜ Fair Value Gaps**")
                for fvg in _scalp.fvgs:
                    d_col = "#2d9e2d" if fvg.direction == "BUY" else "#c9302c"
                    rr    = fvg.pips_to_target / fvg.risk_pips if fvg.risk_pips > 0 else 0
                    st.markdown(
                        f"<span style='color:{d_col}'>{'▲' if fvg.direction=='BUY' else '▼'} "
                        f"{fvg.direction}</span> · Gap: `{fvg.entry_zone_low:.5f}`–`{fvg.entry_zone_high:.5f}` · "
                        f"{fvg.pips_to_target:.0f} pips · R:R {rr:.1f}:1",
                        unsafe_allow_html=True
                    )

            # ── Liquidity Sweeps ─────────────────────────────────────────────
            if _scalp.liq_sweeps:
                st.markdown("**💧 Liquidity Sweeps**")
                for sw in _scalp.liq_sweeps:
                    d_col = "#2d9e2d" if sw.direction == "BUY" else "#c9302c"
                    rr    = sw.pips_to_target / sw.risk_pips if sw.risk_pips > 0 else 0
                    st.markdown(
                        f"<span style='color:{d_col}'>{'▲' if sw.direction=='BUY' else '▼'} "
                        f"{sw.direction}</span> · {sw.description[:70]} · R:R {rr:.1f}:1",
                        unsafe_allow_html=True
                    )

        with st.expander("📊 Volume Profile Details"):
            c1, c2, c3 = st.columns(3)
            c1.metric("POC", f"{vp.poc:.5f}")
            c2.metric("VAH", f"{vp.vah:.5f}")
            c3.metric("VAL", f"{vp.val:.5f}")
            st.write(f"**Location:** {vp.price_location.replace('_', ' ')}")
            st.write(f"**RSI (1H):** {vp.rsi_1h:.1f}")
            st.write(f"**ATR:** {vp.atr:.5f} ({vp.atr_pct*100:.3f}%)")
            st.write(f"**Vol Intensity:** {vp.volume_intensity:.2f} "
                     f"{'⚠️ Exhaustion' if vp.volume_intensity < 0.7 else '✅ Normal'}")
            st.write(f"**SMA 50 (4H):** {vp.sma50_4h:.5f}")
            st.write(f"**SMA 200 (D):** {vp.sma200_1d:.5f}")
            if vp.hvns:
                st.write(f"**HVN:** {', '.join([f'{p:.5f}' for p in vp.hvns[:3]])}")
            if vp.lvns:
                st.write(f"**LVN:** {', '.join([f'{p:.5f}' for p in vp.lvns[:3]])}")

        # Claude AI Analysis
        skip_signals   = ["NEUTRAL — Inside VA", "HIGH RISK"]
        should_analyse = not any(s in vp.signal for s in skip_signals)

        if should_analyse:
            with st.expander("🤖 Claude AI Analysis", expanded=True):

                session_ok, session_msg = _claude_allowed_forex(pair)

                if not session_ok:
                    st.caption(session_msg)
                elif _global_fx_regime.state == 2 and not _global_fx_regime.allowed_signals:
                    st.error(f"🔴 CRISIS regime — {_global_fx_regime.strategy_note}")
                else:
                    # Risk checks
                    trade_allowed = True
                    block_reason  = ""
                    if st.session_state.fx_kill:
                        trade_allowed = False
                        block_reason  = "🔴 Kill switch ON"
                    elif st.session_state.fx_open_pair and st.session_state.fx_open_pair != pair:
                        trade_allowed = False
                        block_reason  = f"⚠️ Position open on {st.session_state.fx_open_pair}"
                    elif st.session_state.fx_trades_today >= 3:
                        trade_allowed = False
                        block_reason  = "⚠️ Max 3 trades/day reached"
                    elif abs(min(st.session_state.fx_daily_pnl, 0)) / account_size * 100 >= daily_limit:
                        trade_allowed = False
                        block_reason  = "🔴 Daily loss limit reached"

                    # Build yield context for forex Claude prompt
                    macro_fx = get_macro_snapshot()
                    yield_ctx = (
                        f"10Y Yield: {macro_fx.yield_10y:.2f}% ({macro_fx.yield_signal}) | "
                        f"Risk Score: {macro_fx.risk_score}/100 ({macro_fx.risk_level})"
                    ) if macro_fx else "unavailable"

                    effective_fx_lot = round(lot_size * _pair_regime.lot_multiplier, 2)
                    if effective_fx_lot <= 0:
                        st.error(f"🔴 {_pair_regime.icon} Regime blocks trading: {_pair_regime.strategy_note}")
                        ai = None
                    else:
                        ai: FXSignal = analyse_pair(vp, effective_fx_lot, account_size, yield_context=yield_ctx)

                    if ai is None:
                        st.caption("🤖 Claude analysed this signal — conditions not fully met yet. Monitoring.")
                    else:
                        cb_color = {"HAWKISH":"green","DOVISH":"red","NEUTRAL":"gray"}.get(ai.cb_sentiment,"gray")
                        st.markdown(
                            f"**CB Sentiment ({pair[:3]}):** :{cb_color}[{ai.cb_sentiment}] | "
                            f"**Type:** {ai.signal_type.replace('_',' ')}"
                        )
                        a_color = {"BUY":"green","SELL":"red","HOLD":"gray"}.get(ai.action,"gray")
                        st.markdown(
                            f"**Decision:** :{a_color}[{ai.action}] | **Confidence:** {ai.confidence}"
                        )
                        st.caption(f"📊 {ai.reasoning}")
                        if ai.news_summary:
                            st.info(f"🗞️ {ai.news_summary}")

                        if ai.action != "HOLD":
                            rr       = ai.target_pips / ai.stop_pips if ai.stop_pips > 0 else 0
                            risk_usd = ai.stop_pips * lot_size * 10
                            st.write({
                                "Level": ["Entry", "Stop Loss", "Take Profit"],
                                "Price": [f"{ai.entry:.5f}", f"{ai.stop_loss:.5f}", f"{ai.take_profit:.5f}"],
                                "Pips":  ["—", f"{ai.stop_pips:.1f}", f"{ai.target_pips:.1f}"],
                            })
                            st.caption(
                                f"R:R = {rr:.1f}:1 | Risk ~${risk_usd:.2f} | "
                                f"Spread: {vp.spread_pips} pips"
                            )

                            if trade_allowed:
                                if st.button(
                                    f"⚡ Execute {ai.action} {pair}",
                                    key=f"fx_exec_{pair}",
                                    type="primary",
                                ):
                                    try:
                                        from broker import place_order
                                        result = place_order(
                                            ticker=pair,
                                            action=ai.action,
                                            stop_loss=ai.stop_loss,
                                            take_profit=ai.take_profit,
                                            lot_size=lot_size,
                                        )
                                        if result.success:
                                            st.session_state.fx_open_pair     = pair
                                            st.session_state.fx_trades_today += 1
                                            st.session_state.fx_trade_log.append({
                                                "time":   pd.Timestamp.now().strftime("%H:%M"),
                                                "pair":   pair,
                                                "action": ai.action,
                                                "entry":  ai.entry,
                                                "sl":     ai.stop_loss,
                                                "tp":     ai.take_profit,
                                            })
                                            st.success(result.message)
                                        else:
                                            st.error(result.message)
                                    except ImportError:
                                        st.success(
                                            f"✅ [DEMO] {ai.action} {pair} @ {ai.entry:.5f} | "
                                            f"SL {ai.stop_loss:.5f} | TP {ai.take_profit:.5f}"
                                        )
                            else:
                                st.error(block_reason)
        else:
            st.caption("📍 Price inside Value Area — no trade setup. Monitor for VA edge approach.")

        st.divider()
