"""Drift measurement, rebalance triggers and order generation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

FREQUENCIES = ("none", "monthly", "quarterly", "annual", "threshold")


@dataclass
class Order:
    ticker: str
    side: str            # BUY / SELL
    shares: float
    est_price: float

    @property
    def est_value(self) -> float:
        return self.shares * self.est_price

    def to_dict(self) -> dict:
        d = asdict(self)
        d["est_value"] = round(self.est_value, 2)
        return d


def is_calendar_rebalance(date: pd.Timestamp, prev_date: pd.Timestamp | None, frequency: str) -> bool:
    """True on the first trading day of a new month/quarter/year."""
    if prev_date is None or frequency in ("none", "threshold"):
        return False
    if frequency == "monthly":
        return (date.year, date.month) != (prev_date.year, prev_date.month)
    if frequency == "quarterly":
        return (date.year, (date.month - 1) // 3) != (prev_date.year, (prev_date.month - 1) // 3)
    if frequency == "annual":
        return date.year != prev_date.year
    raise ValueError(f"Unknown rebalance frequency {frequency!r}")


class Rebalancer:
    def __init__(self, threshold: float = 0.05, min_trade_value: float = 50.0,
                 allow_fractional: bool = False, cash_buffer: float = 0.0):
        self.threshold = threshold
        self.min_trade_value = min_trade_value
        self.allow_fractional = allow_fractional
        self.cash_buffer = cash_buffer

    @classmethod
    def from_config(cls, cfg: dict) -> "Rebalancer":
        r = cfg.get("rebalancing", {})
        return cls(r.get("threshold", 0.05), r.get("min_trade_value", 50), r.get("allow_fractional", False),
                   r.get("cash_buffer", 0.0))

    @staticmethod
    def _values(holdings: dict, prices: pd.Series) -> pd.Series:
        return pd.Series({t: s * float(prices[t]) for t, s in holdings.items()}, dtype=float)

    def drift(self, holdings: dict[str, float], prices: pd.Series, target: pd.Series, cash: float = 0.0) -> pd.DataFrame:
        tickers = sorted(set(holdings) | set(target[target > 0].index))
        vals = self._values({t: holdings.get(t, 0.0) for t in tickers}, prices)
        total = vals.sum() + cash
        cur = vals / total if total > 0 else vals * 0
        tgt = target.reindex(tickers).fillna(0.0)
        tgt = tgt / tgt.sum() if tgt.sum() > 0 else tgt
        return pd.DataFrame({"current_weight": cur, "target_weight": tgt, "drift": cur - tgt, "value": vals})

    def needs_rebalance(self, holdings, prices, target, cash: float = 0.0) -> bool:
        d = self.drift(holdings, prices, target, cash)
        return bool((d["drift"].abs() > self.threshold).any())

    def _round(self, shares: float) -> float:
        return round(shares, 6) if self.allow_fractional else float(math.floor(shares + 1e-9))

    def generate_orders(self, holdings: dict[str, float], prices: pd.Series, target: pd.Series,
                        cash: float = 0.0, contribution: float = 0.0) -> list[Order]:
        """Full rebalance to target. Sells first; buys sized to the cash actually available."""
        cash = cash + contribution
        d = self.drift(holdings, prices, target, cash)
        total = d["value"].sum() + cash
        investable = total * (1 - self.cash_buffer)
        desired = d["target_weight"] * investable
        delta = desired - d["value"]

        orders: list[Order] = []
        avail = cash
        for t in delta.index:                                  # sells
            if delta[t] < -self.min_trade_value:
                px = float(prices[t])
                sh = min(self._round(-delta[t] / px), holdings.get(t, 0.0))
                if d.at[t, "target_weight"] == 0:
                    sh = holdings.get(t, 0.0)                  # fully exit dropped positions
                if sh > 0:
                    orders.append(Order(t, "SELL", sh, px))
                    avail += sh * px
        avail -= total * self.cash_buffer
        buys = delta[delta > self.min_trade_value].sort_values(ascending=False)
        for t, amt in buys.items():                            # buys
            px = float(prices[t])
            sh = self._round(min(amt, max(avail, 0)) / px)
            if sh > 0 and sh * px >= self.min_trade_value:
                orders.append(Order(t, "BUY", sh, px))
                avail -= sh * px
        return orders

    def cash_flow_orders(self, holdings: dict[str, float], prices: pd.Series, target: pd.Series,
                         cash: float) -> list[Order]:
        """Buy-only rebalance: direct new cash to the most underweight positions (no taxable sales)."""
        d = self.drift(holdings, prices, target, 0.0)
        total_after = d["value"].sum() + cash
        need = (d["target_weight"] * total_after - d["value"]).clip(lower=0)
        if need.sum() <= 0:
            need = d["target_weight"] * cash
        alloc = need / need.sum() * cash
        orders = []
        for t, amt in alloc.sort_values(ascending=False).items():
            px = float(prices[t])
            sh = self._round(amt / px)
            if sh > 0 and sh * px >= self.min_trade_value:
                orders.append(Order(t, "BUY", sh, px))
        return orders
