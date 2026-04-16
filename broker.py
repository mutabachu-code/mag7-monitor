"""
broker.py  —  Exness MT5 execution layer
-----------------------------------------
Connects to a locally running MetaTrader 5 terminal (Windows).
MT5 must be installed, logged into your Exness account, and have
"Allow Algo Trading" enabled in Tools > Options > Expert Advisors.

Exness stock CFD symbols: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA
(confirm exact symbols in your MT5 Market Watch — they may have suffixes
like AAPLm on some Exness account types)
"""

import os
from dataclasses import dataclass
from typing import Optional

# MetaTrader5 is Windows-only. On Streamlit Cloud (Linux) we fall back
# to a stub so the dashboard still renders without crashing.
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False


# ── Exness MT5 credentials — set in Streamlit secrets or .env ────────────────
MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER", "Exness-MT5Real7")   # check your PA for exact server name
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[int]
    message: str
    filled_price: Optional[float] = None


def _connect() -> bool:
    """Initialise and authenticate with the MT5 terminal."""
    if not MT5_AVAILABLE:
        return False
    if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        print(f"[broker] MT5 init failed: {mt5.last_error()}")
        return False
    return True


def get_open_position(ticker: str) -> Optional[dict]:
    """Return the open position for a ticker, or None if flat."""
    if not _connect():
        return None
    positions = mt5.positions_get(symbol=ticker)
    if positions:
        p = positions[0]
        return {
            "ticket": p.ticket,
            "type":   "BUY" if p.type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume": p.volume,
            "open_price": p.price_open,
            "profit": p.profit,
        }
    return None


def close_position(ticker: str) -> OrderResult:
    """Close any open position on this ticker before opening a new one."""
    if not _connect():
        return OrderResult(False, None, "MT5 not available")

    positions = mt5.positions_get(symbol=ticker)
    if not positions:
        return OrderResult(True, None, "No open position to close")

    pos = positions[0]
    tick = mt5.symbol_info_tick(ticker)
    close_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
    order_type  = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY

    request = {
        "action":    mt5.TRADE_ACTION_DEAL,
        "symbol":    ticker,
        "volume":    pos.volume,
        "type":      order_type,
        "position":  pos.ticket,
        "price":     close_price,
        "deviation": 20,
        "magic":     20250416,
        "comment":   "claude_close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        return OrderResult(True, result.order, f"Closed position {pos.ticket}", close_price)
    return OrderResult(False, None, f"Close failed: {result.comment} (retcode {result.retcode})")


def place_order(
    ticker: str,
    action: str,           # "BUY" or "SELL"
    stop_loss: float,
    take_profit: float,
    lot_size: float = 0.02,
) -> OrderResult:
    """
    Place a market order on Exness via MT5.

    One-trade-at-a-time rule is enforced:
    - If a position already exists for this ticker, it is closed first.
    - Then the new order is opened.

    stop_loss and take_profit must be absolute price levels (not pips/points).
    """

    if not MT5_AVAILABLE:
        return OrderResult(
            False, None,
            "[STUB] MetaTrader5 library not installed. "
            "Run on Windows with MT5 terminal open to execute live trades."
        )

    if not _connect():
        return OrderResult(False, None, "Could not connect to MT5 terminal")

    # Enforce one-trade-at-a-time: close any existing position first
    existing = get_open_position(ticker)
    if existing:
        close_result = close_position(ticker)
        if not close_result.success:
            return OrderResult(False, None, f"Could not close existing position: {close_result.message}")

    # Fetch current market price
    tick = mt5.symbol_info_tick(ticker)
    if tick is None:
        return OrderResult(False, None, f"Cannot get price for {ticker} — check symbol name in MT5 Market Watch")

    price       = tick.ask if action == "BUY" else tick.bid
    order_type  = mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL

    # Validate SL/TP direction
    if action == "BUY" and stop_loss >= price:
        return OrderResult(False, None, f"BUY stop loss ({stop_loss}) must be below entry ({price:.2f})")
    if action == "SELL" and stop_loss <= price:
        return OrderResult(False, None, f"SELL stop loss ({stop_loss}) must be above entry ({price:.2f})")

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       ticker,
        "volume":       lot_size,
        "type":         order_type,
        "price":        price,
        "sl":           round(stop_loss, 2),
        "tp":           round(take_profit, 2),
        "deviation":    20,           # max slippage in points
        "magic":        20250416,     # EA identifier
        "comment":      "claude_signal",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result.retcode == mt5.TRADE_RETCODE_DONE:
        return OrderResult(
            success=True,
            order_id=result.order,
            message=f"✅ {action} {lot_size} lot {ticker} @ {result.price:.2f} | SL {stop_loss:.2f} | TP {take_profit:.2f}",
            filled_price=result.price,
        )
    else:
        return OrderResult(
            success=False,
            order_id=None,
            message=f"❌ Order failed: {result.comment} (retcode {result.retcode})",
        )
