"""Dividend reinvestment (DRIP) simulation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

POLICIES = ("same", "underweight", "cash")


@dataclass
class DripResult:
    shares: pd.DataFrame      # shares held per day
    cash: pd.Series           # uninvested cash per day
    value: pd.Series          # total portfolio value per day
    income: pd.Series         # net dividend cash received per day
    events: pd.DataFrame      # one row per dividend payment

    @property
    def total_income(self) -> float:
        return float(self.income.sum())


def _pick_underweight(holdings: pd.Series, prices: pd.Series, target: pd.Series) -> str:
    vals = holdings * prices
    total = vals.sum()
    cur = vals / total if total > 0 else vals * 0
    gap = target.reindex(cur.index).fillna(0) - cur
    return str(gap.idxmax())


def simulate_drip(
    close: pd.DataFrame,
    dividends: dict[str, pd.Series],
    initial_shares: dict[str, float] | pd.Series,
    policy: str = "same",
    target_weights: pd.Series | None = None,
    fractional: bool = True,
    tax_rate: float = 0.0,
) -> DripResult:
    """Walk forward day by day; on each ex-date credit dividends and apply the reinvestment policy.

    policy:
      same        reinvest into the paying ETF
      underweight reinvest into the most underweight ETF vs target_weights
      cash        keep dividends as cash
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}")
    if policy == "underweight" and target_weights is None:
        raise ValueError("underweight policy requires target_weights")

    tickers = list(pd.Series(initial_shares).index)
    px = close[tickers].ffill().dropna(how="all")
    shares = pd.Series(initial_shares, dtype=float).reindex(tickers).fillna(0)
    cash = 0.0
    div_by_day = pd.DataFrame({t: dividends.get(t, pd.Series(dtype=float)) for t in tickers}).reindex(px.index).fillna(0)

    sh_hist, cash_hist, inc_hist, events = [], [], [], []
    for dt in px.index:
        p = px.loc[dt]
        income_today = 0.0
        for t in tickers:
            dps = div_by_day.at[dt, t]
            if dps <= 0 or shares[t] <= 0:
                continue
            net = shares[t] * dps * (1 - tax_rate)
            income_today += net
            cash += net
            target = t if policy == "same" else (_pick_underweight(shares, p, target_weights) if policy == "underweight" else None)
            bought = 0.0
            if target is not None and p[target] > 0:
                bought = cash / p[target] if fractional else float(int(cash // p[target]))
                shares[target] += bought
                cash -= bought * p[target]
            events.append({"date": dt, "ticker": t, "dps": dps, "net_income": net,
                           "reinvested_into": target, "shares_bought": bought})
        sh_hist.append(shares.copy())
        cash_hist.append(cash)
        inc_hist.append(income_today)

    sh_df = pd.DataFrame(sh_hist, index=px.index)
    cash_s = pd.Series(cash_hist, index=px.index)
    value = (sh_df * px).sum(axis=1) + cash_s
    return DripResult(sh_df, cash_s, value, pd.Series(inc_hist, index=px.index),
                      pd.DataFrame(events, columns=["date", "ticker", "dps", "net_income", "reinvested_into", "shares_bought"]))
