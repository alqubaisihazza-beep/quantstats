"""Dividend history retrieval and per-ETF dividend statistics."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from data.database import Database


class DividendData:
    def __init__(self, provider, db: Database | None = None):
        self.provider = provider
        self.db = db

    def get_dividends(self, tickers, start=None, end=None, refresh: bool = False) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for t in tickers:
            s = None
            if self.db is not None and not refresh:
                cached = self.db.load_dividends(t, start, end)
                s = cached if not cached.empty else None
            if s is None:
                try:
                    s = self.provider.fetch_dividends(t, start, end)
                except Exception as exc:
                    print(f"[dividend_data] failed to fetch {t}: {exc}", file=sys.stderr)
                    s = pd.Series(dtype=float, name=t)
                if self.db is not None and not s.empty:
                    self.db.save_dividends(t, s)
            out[t] = s.sort_index().rename(t)
        return out


def ttm_dividends(divs: pd.Series, as_of=None) -> float:
    """Sum of dividends paid in the trailing 365 days."""
    if divs is None or divs.empty:
        return 0.0
    as_of = pd.Timestamp(as_of) if as_of is not None else divs.index[-1]
    return float(divs.loc[as_of - pd.Timedelta(days=365) + pd.Timedelta(days=1): as_of].sum())


def trailing_yield(divs: pd.Series, prices: pd.Series, as_of=None) -> float:
    px = prices.dropna()
    if px.empty:
        return np.nan
    as_of = pd.Timestamp(as_of) if as_of is not None else px.index[-1]
    price = px.loc[:as_of].iloc[-1]
    return ttm_dividends(divs, as_of) / price if price > 0 else np.nan


def annual_dividends(divs: pd.Series, complete_years_only: bool = True, as_of=None) -> pd.Series:
    if divs is None or divs.empty:
        return pd.Series(dtype=float)
    yearly = divs.groupby(divs.index.year).sum()
    if complete_years_only:
        last_year = (pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()).year
        yearly = yearly[yearly.index < last_year]
        # drop a partial first year (history starts mid-year)
        first_ts = divs.index[0]
        if first_ts.month > 3 and len(yearly) and yearly.index[0] == first_ts.year:
            yearly = yearly.iloc[1:]
    return yearly


def dividend_cagr(divs: pd.Series, years: int = 5, as_of=None) -> float:
    yearly = annual_dividends(divs, as_of=as_of)
    if len(yearly) < 2:
        return np.nan
    yearly = yearly.iloc[-(years + 1):]
    first, last = yearly.iloc[0], yearly.iloc[-1]
    n = yearly.index[-1] - yearly.index[0]
    if first <= 0 or last <= 0 or n <= 0:
        return np.nan
    return float((last / first) ** (1 / n) - 1)


def dividend_consistency(divs: pd.Series, years: int = 5, as_of=None) -> float:
    """Fraction of year-over-year changes that were non-decreasing."""
    yearly = annual_dividends(divs, as_of=as_of).iloc[-(years + 1):]
    if len(yearly) < 2:
        return np.nan
    changes = yearly.diff().dropna()
    return float((changes >= -1e-9).mean())


def payment_frequency(divs: pd.Series) -> str:
    if divs is None or len(divs) < 3:
        return "unknown" if divs is None or divs.empty else "irregular"
    gap = float(np.median(np.diff(divs.index.values).astype("timedelta64[D]").astype(float)))
    if gap <= 45:
        return "monthly"
    if gap <= 120:
        return "quarterly"
    if gap <= 250:
        return "semiannual"
    return "annual"


def payments_per_year(freq: str) -> int:
    return {"monthly": 12, "quarterly": 4, "semiannual": 2, "annual": 1}.get(freq, 0)


def dividend_summary(div_map: dict[str, pd.Series], close: pd.DataFrame, as_of=None) -> pd.DataFrame:
    rows = []
    for t, s in div_map.items():
        px = close[t] if t in close.columns else pd.Series(dtype=float)
        rows.append({
            "ticker": t,
            "ttm_dividend": ttm_dividends(s, as_of),
            "yield": trailing_yield(s, px, as_of) if not px.empty else np.nan,
            "div_cagr_5y": dividend_cagr(s, 5, as_of),
            "div_consistency": dividend_consistency(s, 5, as_of),
            "frequency": payment_frequency(s),
            "last_payment": s.index[-1].date() if not s.empty else None,
        })
    return pd.DataFrame(rows).set_index("ticker") if rows else pd.DataFrame()
