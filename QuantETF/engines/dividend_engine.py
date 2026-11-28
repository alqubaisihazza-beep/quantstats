"""Portfolio dividend income: forward income, yield on cost, monthly calendar, growth projection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.dividend_data import dividend_cagr, payment_frequency, payments_per_year, ttm_dividends


class DividendEngine:
    def __init__(self, dividends: dict[str, pd.Series], close: pd.DataFrame):
        self.dividends = dividends
        self.close = close

    def _price(self, t: str) -> float:
        return float(self.close[t].dropna().iloc[-1]) if t in self.close.columns else np.nan

    def _divs(self, t: str) -> pd.Series:
        return self.dividends.get(t, pd.Series(dtype=float))

    def forward_income(self, holdings: dict[str, float]) -> pd.DataFrame:
        """Annual income estimate using trailing-12-month dividends per share."""
        rows = []
        for t, shares in holdings.items():
            divs = self._divs(t)
            dps = ttm_dividends(divs)
            price = self._price(t)
            income = shares * dps
            rows.append({
                "ticker": t, "shares": shares, "price": price, "value": shares * price,
                "ttm_dps": dps, "annual_income": income, "monthly_income": income / 12,
                "yield": dps / price if price else np.nan, "frequency": payment_frequency(divs),
            })
        df = pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
        if not df.empty:
            df["income_share"] = df["annual_income"] / df["annual_income"].sum() if df["annual_income"].sum() else 0
        return df

    def holdings_from_weights(self, weights: pd.Series, portfolio_value: float) -> dict[str, float]:
        return {t: portfolio_value * w / self._price(t) for t, w in weights.items() if w > 0 and self._price(t) > 0}

    def portfolio_yield(self, weights: pd.Series) -> float:
        w = weights / weights.sum()
        return float(sum(w[t] * ttm_dividends(self._divs(t)) / self._price(t) for t in w.index if self._price(t) > 0))

    def yield_on_cost(self, holdings: dict[str, float], cost_basis: dict[str, float]) -> pd.Series:
        """cost_basis: total dollars paid per ticker."""
        return pd.Series({t: holdings[t] * ttm_dividends(self._divs(t)) / cost_basis[t]
                          for t in holdings if cost_basis.get(t)})

    def income_calendar(self, holdings: dict[str, float], start: pd.Timestamp | None = None) -> pd.DataFrame:
        """Projected payments for the next 12 months (rows = month, cols = ticker).

        Each payment made in the trailing 12 months is assumed to recur in the same calendar month.
        """
        start = (pd.Timestamp(start) if start is not None else pd.Timestamp.today()).to_period("M")
        months = pd.period_range(start, periods=12, freq="M")
        cal = pd.DataFrame(0.0, index=months.strftime("%Y-%m"), columns=list(holdings))
        for t, shares in holdings.items():
            divs = self._divs(t)
            if divs.empty:
                continue
            last12 = divs.loc[divs.index[-1] - pd.Timedelta(days=364):]
            for dt, amt in last12.items():
                for m in months:
                    if m.month == dt.month:
                        cal.loc[m.strftime("%Y-%m"), t] += shares * amt
        cal["total"] = cal.sum(axis=1)
        return cal

    def expected_growth(self, weights: pd.Series, years: int = 5) -> float:
        g = []
        w = weights / weights.sum()
        for t in w.index:
            c = dividend_cagr(self._divs(t), years)
            if not np.isnan(c):
                g.append((w[t], c))
        tot = sum(x for x, _ in g)
        return float(sum(x * c for x, c in g) / tot) if tot else 0.0

    @staticmethod
    def project_income(initial_value: float, yield_: float, dividend_growth: float, price_growth: float,
                       years: int = 10, reinvest: bool = True, annual_contribution: float = 0.0) -> pd.DataFrame:
        """Simple annual projection of portfolio value and income."""
        rows, value, y = [], initial_value, yield_
        for yr in range(1, years + 1):
            income = value * y
            value = value * (1 + price_growth) + (income if reinvest else 0) + annual_contribution
            # dividend growth beyond price growth raises the yield on the new value
            y = y * (1 + dividend_growth) / (1 + price_growth)
            rows.append({"year": yr, "income": income, "value": value, "yield": y})
        return pd.DataFrame(rows).set_index("year")

    def payments_per_year(self, t: str) -> int:
        return payments_per_year(payment_frequency(self._divs(t)))
