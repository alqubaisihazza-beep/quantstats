"""Manual execution: write order tickets, track them in the DB, confirm fills by hand."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.database import Database
from engines.rebalancing import Order


class ManualExecutionBroker:
    name = "manual"

    def __init__(self, db: Database, account: str = "manual", output_dir: str | Path = "reports"):
        self.db = db
        self.account = account
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------- positions
    def get_positions(self) -> pd.DataFrame:
        return self.db.get_holdings(self.account)

    def get_cash(self) -> float:
        return self.db.get_cash(self.account)

    def set_cash(self, amount: float) -> None:
        self.db.set_cash(self.account, amount)

    def import_positions_csv(self, path: str | Path) -> int:
        """CSV columns: ticker, shares[, cost_basis]. Use ticker CASH for the cash balance."""
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        n = 0
        for _, r in df.iterrows():
            t = str(r["ticker"]).strip().upper()
            if t == "CASH":
                self.db.set_cash(self.account, float(r["shares"]))
            else:
                self.db.set_holding(self.account, t, float(r["shares"]), float(r.get("cost_basis", 0) or 0))
                n += 1
        return n

    # --------------------------------------------------------------- orders
    def submit_orders(self, orders: list[Order], note: str = "") -> dict:
        """Record orders as pending and write a CSV + markdown ticket to place by hand."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        csv_path = self.output_dir / f"quantetf_orders_{stamp}.csv"
        md_path = self.output_dir / f"quantetf_orders_{stamp}.md"
        ids = []
        rows = []
        for o in orders:
            tid = self.db.record_trade(self.account, o.ticker, o.side, o.shares, o.est_price,
                                       status="pending", broker=self.name, note=note)
            ids.append(tid)
            rows.append({"id": tid, **o.to_dict()})
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "ticker", "side", "shares", "est_price", "est_value"])
            w.writeheader()
            w.writerows(rows)
        sells = sum(o.est_value for o in orders if o.side == "SELL")
        buys = sum(o.est_value for o in orders if o.side == "BUY")
        md = [
            f"# QuantETF Order Ticket ({stamp})",
            f"Account: `{self.account}`  ",
            f"Orders: {len(orders)} | Est. sells: ${sells:,.2f} | Est. buys: ${buys:,.2f}",
            "",
            "Place SELL orders first, then BUY orders. Prefer limit orders near the quoted price.",
            "",
            "| ID | Side | Ticker | Shares | Est. price | Est. value |",
            "|---|---|---|---|---|---|",
            *[f"| {r['id']} | {r['side']} | {r['ticker']} | {r['shares']:g} | ${r['est_price']:.2f} | ${r['est_value']:,.2f} |"
              for r in sorted(rows, key=lambda r: r["side"] != "SELL")],
            "",
            "After execution, confirm each fill: `python app.py fill <ID> --price <PRICE>`",
        ]
        md_path.write_text("\n".join(md), encoding="utf-8")
        return {"trade_ids": ids, "csv": csv_path, "ticket": md_path}

    def confirm_fill(self, trade_id: int, price: float, shares: float | None = None) -> None:
        trades = self.db.get_trades()
        row = trades[trades["id"] == trade_id]
        if row.empty:
            raise ValueError(f"No trade {trade_id}")
        r = row.iloc[0]
        if r["status"] == "filled":
            raise ValueError(f"Trade {trade_id} already filled")
        filled = float(shares if shares is not None else r["shares"])
        self.db.apply_fill(r["account"], r["ticker"], r["side"], filled, float(price))
        self.db.update_trade(trade_id, status="filled", price=float(price), shares=filled,
                             filled_at=datetime.now().isoformat(timespec="seconds"))

    def cancel(self, trade_id: int) -> None:
        self.db.update_trade(trade_id, status="cancelled")

    def pending(self) -> pd.DataFrame:
        return self.db.get_trades(self.account, status="pending")
