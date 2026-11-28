"""Interactive Brokers execution via ib_insync (TWS / IB Gateway).

Safety: orders are previewed (what-if) unless dry_run=False AND env QUANTETF_ALLOW_LIVE=1.
Use a paper account (port 7497 TWS / 4002 Gateway) while testing.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

from engines.rebalancing import Order


def _ib_insync():
    try:
        import ib_insync
    except ImportError:
        print("ib_insync is not installed: pip install -r requirements.txt", file=sys.stderr)
        raise
    return ib_insync


def order_spec(order: Order, order_type: str = "LMT", limit_offset_bps: float = 10.0) -> dict:
    """Broker-neutral order description. Limit buys priced slightly above, sells slightly below."""
    side = order.side.upper()
    if side not in ("BUY", "SELL"):
        raise ValueError(f"bad side {order.side}")
    spec = {"symbol": order.ticker, "action": side, "quantity": float(order.shares), "order_type": order_type}
    if order_type == "LMT":
        off = limit_offset_bps / 1e4
        px = order.est_price * (1 + off if side == "BUY" else 1 - off)
        spec["limit_price"] = round(px, 2)
    elif order_type != "MKT":
        raise ValueError("order_type must be LMT or MKT")
    return spec


class IBKRBroker:
    name = "ibkr"

    def __init__(self, host: str | None = None, port: int | None = None, client_id: int | None = None,
                 account: str | None = None, dry_run: bool = True, order_type: str = "LMT",
                 limit_offset_bps: float = 10.0):
        self.host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("IBKR_PORT", 7497))
        self.client_id = int(client_id or os.getenv("IBKR_CLIENT_ID", 17))
        self.account = account or os.getenv("IBKR_ACCOUNT") or None
        self.dry_run = dry_run
        self.order_type = order_type
        self.limit_offset_bps = limit_offset_bps
        self.ib = None

    @classmethod
    def from_config(cls, cfg: dict) -> "IBKRBroker":
        b = cfg.get("brokers", {}).get("ibkr", {})
        return cls(dry_run=b.get("dry_run", True), order_type=b.get("order_type", "LMT"),
                   limit_offset_bps=b.get("limit_offset_bps", 10))

    @property
    def live_allowed(self) -> bool:
        return not self.dry_run and os.getenv("QUANTETF_ALLOW_LIVE") == "1"

    # ----------------------------------------------------------- connection
    def connect(self, timeout: float = 10.0):
        ibs = _ib_insync()
        self.ib = ibs.IB()
        self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=timeout, readonly=not self.live_allowed)
        return self

    def disconnect(self):
        if self.ib is not None:
            self.ib.disconnect()
            self.ib = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.disconnect()

    def _require(self):
        if self.ib is None or not self.ib.isConnected():
            raise RuntimeError("Not connected to IBKR; call connect() first")

    # ------------------------------------------------------------- account
    def get_positions(self) -> pd.DataFrame:
        self._require()
        rows = [{"ticker": p.contract.symbol, "shares": float(p.position), "cost_basis": float(p.avgCost * p.position)}
                for p in self.ib.positions(self.account or "") if p.contract.secType in ("STK", "ETF")]
        return pd.DataFrame(rows, columns=["ticker", "shares", "cost_basis"]).set_index("ticker")

    def get_cash(self, currency: str = "USD") -> float:
        self._require()
        for v in self.ib.accountValues(self.account or ""):
            if v.tag == "TotalCashValue" and v.currency == currency:
                return float(v.value)
        return 0.0

    def get_prices(self, tickers: list[str]) -> pd.Series:
        self._require()
        ibs = _ib_insync()
        contracts = [ibs.Stock(t, "SMART", "USD") for t in tickers]
        self.ib.qualifyContracts(*contracts)
        tickers_data = self.ib.reqTickers(*contracts)
        return pd.Series({td.contract.symbol: float(td.marketPrice()) for td in tickers_data})

    # -------------------------------------------------------------- orders
    def submit_orders(self, orders: list[Order]) -> list[dict]:
        """Sells are sent before buys. Returns per-order status dicts."""
        self._require()
        ibs = _ib_insync()
        results = []
        for o in sorted(orders, key=lambda x: x.side != "SELL"):
            spec = order_spec(o, self.order_type, self.limit_offset_bps)
            contract = ibs.Stock(spec["symbol"], "SMART", "USD")
            self.ib.qualifyContracts(contract)
            ib_order = (ibs.LimitOrder(spec["action"], spec["quantity"], spec["limit_price"])
                        if spec["order_type"] == "LMT" else ibs.MarketOrder(spec["action"], spec["quantity"]))
            if self.account:
                ib_order.account = self.account
            if not self.live_allowed:
                state = self.ib.whatIfOrder(contract, ib_order)
                results.append({**spec, "status": "preview",
                                "commission": getattr(state, "commission", None),
                                "init_margin_change": getattr(state, "initMarginChange", None)})
            else:
                trade = self.ib.placeOrder(contract, ib_order)
                results.append({**spec, "status": trade.orderStatus.status, "order_id": trade.order.orderId})
        return results
