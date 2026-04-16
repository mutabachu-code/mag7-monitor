"""
risk_manager.py
---------------
Hard guardrails for a $100 Exness account trading 0.02 lots.
- One trade at a time (enforced in broker.py + checked here)
- Daily loss limit: halts bot when threshold hit
- Kill switch: manual override in sidebar
- Lot size: fixed 0.02 (configurable but capped)
"""

import streamlit as st
import datetime
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RiskConfig:
    account_size_usd: float  = 100.0
    lot_size: float          = 0.02     # fixed per your spec
    max_lot_size: float      = 0.05     # hard cap — never exceed
    daily_loss_limit_pct: float = 5.0   # % of account; halts bot at $5 loss on $100
    max_trades_per_day: int  = 3        # avoid overtrading on small account


def init_risk_state():
    """Call once at app start — survives 60s auto-refreshes via session_state."""
    defaults = {
        "risk_daily_pnl":       0.0,
        "risk_trades_today":    0,
        "risk_kill_switch":     False,
        "risk_last_reset_date": datetime.date.today(),
        "risk_trade_log":       [],      # list of trade dicts for the log
        "risk_open_ticker":     None,    # which ticker currently has an open position
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _auto_reset_daily():
    today = datetime.date.today()
    if st.session_state.risk_last_reset_date != today:
        st.session_state.risk_daily_pnl      = 0.0
        st.session_state.risk_trades_today   = 0
        st.session_state.risk_kill_switch    = False
        st.session_state.risk_last_reset_date = today
        # Don't reset trade log — keep history


def check_trade_allowed(config: RiskConfig, ticker: str) -> Tuple[bool, str]:
    """
    Returns (allowed, reason).
    Call BEFORE sending any order to MT5.
    """
    _auto_reset_daily()

    if st.session_state.risk_kill_switch:
        return False, "🔴 Kill switch is ON — trading halted."

    # One trade at a time
    open_ticker = st.session_state.risk_open_ticker
    if open_ticker and open_ticker != ticker:
        return False, f"⚠️ Already in a position on {open_ticker}. Close it before opening {ticker}."

    # Daily loss limit
    loss_usd = abs(min(st.session_state.risk_daily_pnl, 0))
    loss_pct = (loss_usd / config.account_size_usd) * 100
    if loss_pct >= config.daily_loss_limit_pct:
        st.session_state.risk_kill_switch = True
        return False, (
            f"🔴 Daily loss limit reached ({loss_pct:.1f}% ≥ {config.daily_loss_limit_pct}%). "
            "Bot halted for today."
        )

    # Max trades per day
    if st.session_state.risk_trades_today >= config.max_trades_per_day:
        return False, f"⚠️ Max {config.max_trades_per_day} trades per day reached."

    # Lot size cap
    if config.lot_size > config.max_lot_size:
        return False, f"⚠️ Lot size {config.lot_size} exceeds hard cap of {config.max_lot_size}."

    return True, "✅ Trade allowed"


def record_trade_opened(ticker: str, action: str, entry: float, sl: float, tp: float, lots: float):
    """Call after a successful order_send."""
    st.session_state.risk_open_ticker   = ticker
    st.session_state.risk_trades_today += 1
    st.session_state.risk_trade_log.append({
        "time":   datetime.datetime.now().strftime("%H:%M:%S"),
        "ticker": ticker,
        "action": action,
        "entry":  entry,
        "sl":     sl,
        "tp":     tp,
        "lots":   lots,
        "pnl":    None,   # filled when closed
    })


def record_trade_closed(ticker: str, pnl_usd: float):
    """Call after a position is closed."""
    if st.session_state.risk_open_ticker == ticker:
        st.session_state.risk_open_ticker = None
    st.session_state.risk_daily_pnl += pnl_usd
    # Update last log entry for this ticker
    for entry in reversed(st.session_state.risk_trade_log):
        if entry["ticker"] == ticker and entry["pnl"] is None:
            entry["pnl"] = pnl_usd
            break


def render_risk_sidebar(config: RiskConfig) -> RiskConfig:
    """Render the full risk control panel in the sidebar. Returns updated config."""
    st.sidebar.header("⚙️ Risk Controls")

    # Kill switch
    if st.session_state.risk_kill_switch:
        st.sidebar.error("🔴 KILL SWITCH: ON — All trading halted")
        if st.sidebar.button("🟢 Resume Trading", type="primary"):
            st.session_state.risk_kill_switch = False
            st.rerun()
    else:
        open_ticker = st.session_state.risk_open_ticker
        if open_ticker:
            st.sidebar.warning(f"📊 Open position: **{open_ticker}**")
        else:
            st.sidebar.success("🟢 Bot ACTIVE — No open position")
        if st.sidebar.button("🔴 HALT All Trading", type="secondary"):
            st.session_state.risk_kill_switch = True
            st.rerun()

    st.sidebar.divider()

    # Daily stats
    pnl    = st.session_state.risk_daily_pnl
    trades = st.session_state.risk_trades_today
    color  = "green" if pnl >= 0 else "red"
    st.sidebar.markdown(f"**Daily P&L:** :{color}[${pnl:+.2f}]")
    st.sidebar.markdown(f"**Trades today:** {trades} / {config.max_trades_per_day}")

    loss_used = abs(min(pnl, 0)) / config.account_size_usd * 100
    st.sidebar.progress(
        min(loss_used / config.daily_loss_limit_pct, 1.0),
        text=f"Daily loss: {loss_used:.1f}% / {config.daily_loss_limit_pct}%",
    )

    st.sidebar.divider()

    # Editable settings
    config.account_size_usd = st.sidebar.number_input(
        "Account size (USD)", min_value=10.0, value=config.account_size_usd, step=10.0
    )
    config.lot_size = st.sidebar.select_slider(
        "Lot size", options=[0.01, 0.02, 0.03, 0.04, 0.05], value=config.lot_size
    )
    config.daily_loss_limit_pct = st.sidebar.slider(
        "Daily loss limit (%)", 1.0, 10.0, config.daily_loss_limit_pct, 0.5
    )
    config.max_trades_per_day = st.sidebar.slider(
        "Max trades / day", 1, 5, config.max_trades_per_day
    )

    # Trade log
    if st.session_state.risk_trade_log:
        st.sidebar.divider()
        st.sidebar.markdown("**Today's trades**")
        import pandas as pd
        df = pd.DataFrame(st.session_state.risk_trade_log)
        st.sidebar.dataframe(df[["time","ticker","action","entry","sl","tp","pnl"]],
                             hide_index=True, use_container_width=True)

    return config
