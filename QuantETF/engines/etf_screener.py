"""Rule-based ETF screening against static metadata and computed market statistics."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

from data.dividend_data import trailing_yield
from engines.risk_engine import annualized_volatility


@dataclass
class ScreenCriteria:
    max_expense_ratio: float | None = None
    min_history_years: float | None = None
    min_avg_dollar_volume: float | None = None
    min_yield: float | None = None
    max_yield: float | None = None
    max_volatility: float | None = None
    categories: list[str] | None = None
    exclude_categories: list[str] | None = None
    asset_classes: list[str] | None = None
    tickers: list[str] | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "ScreenCriteria":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def merged(self, other: dict | None) -> "ScreenCriteria":
        base = {f.name: getattr(self, f.name) for f in fields(self)}
        base.update({k: v for k, v in (other or {}).items() if k in base})
        return ScreenCriteria(**base)


def build_universe_frame(universe_cfg: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(universe_cfg)
    if df.empty:
        return pd.DataFrame(columns=["name", "category", "asset_class", "expense_ratio"])
    return df.drop_duplicates("ticker").set_index("ticker")


class ETFScreener:
    def __init__(self, universe: pd.DataFrame):
        self.universe = universe

    def market_stats(self, prices, dividends: dict | None = None, lookback_days: int = 63) -> pd.DataFrame:
        rows = {}
        rets = prices.returns(total=True)
        for t in self.universe.index:
            if t not in prices.close.columns:
                rows[t] = {"history_years": 0.0, "avg_dollar_volume": np.nan, "yield": np.nan, "volatility": np.nan}
                continue
            px = prices.close[t].dropna()
            vol = prices.volume[t].reindex(px.index).fillna(0)
            dv = (px * vol).iloc[-lookback_days:].mean()
            hist_years = (px.index[-1] - px.index[0]).days / 365.25 if len(px) > 1 else 0.0
            divs = (dividends or {}).get(t)
            rows[t] = {
                "history_years": hist_years,
                "avg_dollar_volume": float(dv),
                "yield": trailing_yield(divs, px) if divs is not None else np.nan,
                "volatility": annualized_volatility(rets[t].iloc[-756:]),
            }
        return pd.DataFrame.from_dict(rows, orient="index")

    def screen(self, criteria: ScreenCriteria, prices=None, dividends: dict | None = None) -> pd.DataFrame:
        """Return universe + stats with `passed` (bool) and `reasons` (why it failed)."""
        df = self.universe.copy()
        if prices is not None:
            df = df.join(self.market_stats(prices, dividends))
        reasons: dict[str, list[str]] = {t: [] for t in df.index}

        def fail(mask: pd.Series, why: str):
            for t in mask[mask].index:
                reasons[t].append(why)

        c = criteria
        if c.tickers:
            fail(~df.index.isin(c.tickers), "not in ticker list")
        if c.categories:
            fail(~df["category"].isin(c.categories), "category")
        if c.exclude_categories:
            fail(df["category"].isin(c.exclude_categories), "excluded category")
        if c.asset_classes:
            fail(~df["asset_class"].isin(c.asset_classes), "asset class")
        if c.max_expense_ratio is not None:
            fail(df["expense_ratio"].fillna(np.inf) > c.max_expense_ratio, "expense ratio")
        if prices is not None:
            if c.min_history_years is not None:
                fail(df["history_years"].fillna(0) < c.min_history_years, "history")
            if c.min_avg_dollar_volume is not None:
                fail(df["avg_dollar_volume"].fillna(0) < c.min_avg_dollar_volume, "liquidity")
            if c.max_volatility is not None:
                fail(df["volatility"].fillna(np.inf) > c.max_volatility, "volatility")
            if c.min_yield is not None:
                fail(df["yield"].fillna(0) < c.min_yield, "yield too low")
            if c.max_yield is not None:
                fail(df["yield"].fillna(0) > c.max_yield, "yield too high")

        df["reasons"] = [", ".join(reasons[t]) for t in df.index]
        df["passed"] = df["reasons"] == ""
        return df

    def passed(self, criteria: ScreenCriteria, prices=None, dividends=None) -> list[str]:
        res = self.screen(criteria, prices, dividends)
        return list(res.index[res["passed"]])
