"""
broker.py
---------
CFD broker integration layer.
Fill in the API details for your specific broker below.
Currently a stub — place_order() logs the trade but does NOT execute.

Once you confirm your broker, replace the stub with the real API calls.
"""

import requests
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str]
    message: str
    filled_price: Optional[float] = None


# ─────────────────────────────────────────────────────────────
# BROKER CONFIGURATION
# Set these in Streamlit secrets or as environment variables.
# NEVER hardcode credentials here.
# ─────────────────────────────────────────────────────────────
BROKER_API_KEY    = os.getenv("BROKER_API_KEY", "")
BROKER_API_SECRET = os.getenv("BROKER_API_SECRET", "")
BROKER_BASE_URL   = os.getenv("BROKER_BASE_URL", "https://api.yourbroker.com")
BROKER_ACCOUNT_ID = os.getenv("BROKER_ACCOUNT_ID", "")


def place_order(
    ticker: str,
    action: str,           # "BUY" or "SELL"
    entry: float,
    stop_loss: float,
    take_profit: float,
    position_usd: float,
) -> OrderResult:
    """
    Place a CFD market order with attached stop-loss and take-profit.

    Currently a STUB — logs the trade parameters and returns a simulated result.
    Replace the body below with your broker's actual API call.

    Common broker API patterns:
    ─────────────────────────────
    OANDA:
        POST https://api-fxtrade.oanda.com/v3/accounts/{id}/orders
        Headers: Authorization: Bearer {token}

    IG Group:
        POST https://api.ig.com/gateway/deal/positions/otc
        Headers: X-IG-API-KEY, X-SECURITY-TOKEN, CST

    Capital.com:
        POST https://api-capital.backend-capital.com/api/v1/positions
        Headers: X-CAP-API-KEY, CST, X-SECURITY-TOKEN

    Interactive Brokers (IBKR):
        POST https://localhost:5000/v1/api/iserver/account/{id}/orders
        (requires TWS Gateway running locally)

    Exness / XM / HotForex:
        Typically MT4/MT5 bridge or proprietary REST API.
        Check your broker's developer portal for endpoint docs.
    """

    # ── STUB: log and simulate ──────────────────────────────────
    print(
        f"[broker.py STUB] {action} {ticker} | "
        f"Entry: ${entry:.2f} | SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | "
        f"Size: ${position_usd:.2f}"
    )

    # TODO: Replace this block with your real broker API call, e.g.:
    #
    # units = int(position_usd / entry)
    # payload = {
    #     "order": {
    #         "type": "MARKET",
    #         "instrument": ticker,
    #         "units": str(units if action == "BUY" else -units),
    #         "takeProfitOnFill": {"price": str(round(take_profit, 5))},
    #         "stopLossOnFill": {"price": str(round(stop_loss, 5))},
    #     }
    # }
    # resp = requests.post(
    #     f"{BROKER_BASE_URL}/v3/accounts/{BROKER_ACCOUNT_ID}/orders",
    #     json=payload,
    #     headers={"Authorization": f"Bearer {BROKER_API_KEY}"},
    # )
    # data = resp.json()
    # return OrderResult(
    #     success=resp.ok,
    #     order_id=data.get("orderCreateTransaction", {}).get("id"),
    #     message=str(data),
    #     filled_price=entry,
    # )

    return OrderResult(
        success=True,
        order_id="STUB-001",
        message=f"[STUB] {action} {ticker} @ ${entry:.2f} queued (broker not connected yet)",
        filled_price=entry,
    )
