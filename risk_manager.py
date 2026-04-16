"""
risk_manager.py
---------------
Enforces hard guardrails before any trade is executed.
State is stored in Streamlit session so it persists across 60s refreshes.
"""

import streamlit as st
from dataclasses import dataclass
from typing import Tuple


@dataclass
class RiskConfig:
    max_position_pct: float = 2.0      # Max % of account per trade
    daily_loss_limit_pct: float = 5.0  # Bot halts if daily loss hits this %
    account_size_usd: float = 1000.0   # Your CFD account size in USD


def init_risk_state():
    """Call once at app startup to initialise session state."""
    if "risk_daily_pnl" not in st.session_state:
        st.session_state.risk_daily_pnl = 0.0       # running P&L today ($)
    if "risk_trades_today" not in st.session_state:
        st.session_state.risk_trades_today = 0
    if "risk_kill_switch" not in st.session_state:
        st.session_state.risk_kill_switch = False    # True = all trading halted
    if "risk_last_reset_date" not in st.session_state:
        import datetime
        st.session_state.risk_last_reset_date = datetime.date.today()


def _auto_reset_daily(config: RiskConfig):
    """Reset daily counters at midnight."""
    import datetime
    today = datetime.date.today()
    if st.session_state.risk_last_reset_date != today:
        st.session_state.risk_daily_pnl = 0.0
        st.session_state.risk_trades_today = 0
        st.session_state.risk_kill_switch = False   # reset kill switch daily
        st.session_state.risk_last_reset_date = today


def check_trade_allowed(config: RiskConfig) -> Tuple[bool, str]:
    """
    Returns (allowed: bool, reason: str).
    Call this BEFORE sending any order to the broker.
    """
    _auto_reset_daily(config)

    if st.session_state.risk_kill_switch:
        return False, "🔴 Kill switch is ON — all trading halted."

    daily_loss_usd = abs(min(st.session_state.risk_daily_pnl, 0))
    daily_loss_pct = (daily_loss_usd / config.account_size_usd) * 100

    if daily_loss_pct >= config.daily_loss_limit_pct:
        st.session_state.risk_kill_switch = True   # auto-halt
        return False, (
            f"🔴 Daily loss limit hit ({daily_loss_pct:.1f}% ≥ "
            f"{config.daily_loss_limit_pct}%). Bot halted for today."
        )

    return True, "✅ Trade allowed"


def record_trade_result(pnl_usd: float):
    """Call after a trade closes to update daily P&L."""
    st.session_state.risk_daily_pnl += pnl_usd
    st.session_state.risk_trades_today += 1


def render_risk_sidebar(config: RiskConfig):
    """
    Renders the risk control panel in the Streamlit sidebar.
    Returns the (possibly user-modified) config.
    """
    st.sidebar.header("⚙️ Risk Controls")

    # Kill switch — prominent red button
    if st.session_state.risk_kill_switch:
        st.sidebar.error("🔴 KILL SWITCH: ACTIVE — Trading halted")
        if st.sidebar.button("🟢 Resume Trading", type="primary"):
            st.session_state.risk_kill_switch = False
            st.rerun()
    else:
        st.sidebar.success("🟢 Bot: ACTIVE")
        if st.sidebar.button("🔴 HALT All Trading", type="secondary"):
            st.session_state.risk_kill_switch = True
            st.rerun()

    st.sidebar.divider()

    # Live daily stats
    pnl = st.session_state.risk_daily_pnl
    trades = st.session_state.risk_trades_today
    pnl_color = "green" if pnl >= 0 else "red"
    st.sidebar.markdown(f"**Daily P&L:** :{pnl_color}[${pnl:+.2f}]")
    st.sidebar.markdown(f"**Trades today:** {trades}")

    daily_loss_used = abs(min(pnl, 0)) / config.account_size_usd * 100
    st.sidebar.progress(
        min(daily_loss_used / config.daily_loss_limit_pct, 1.0),
        text=f"Daily loss: {daily_loss_used:.1f}% / {config.daily_loss_limit_pct}%",
    )

    st.sidebar.divider()

    # Editable config
    config.account_size_usd = st.sidebar.number_input(
        "Account size (USD)", min_value=100.0, value=config.account_size_usd, step=100.0
    )
    config.max_position_pct = st.sidebar.slider(
        "Max position size (%)", 0.5, 2.0, config.max_position_pct, 0.25
    )
    config.daily_loss_limit_pct = st.sidebar.slider(
        "Daily loss limit (%)", 1.0, 10.0, config.daily_loss_limit_pct, 0.5
    )

    return config
