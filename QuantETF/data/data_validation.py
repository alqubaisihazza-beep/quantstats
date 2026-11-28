"""Price/dividend data quality checks and cleaning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    issues: list[dict] = field(default_factory=list)

    def add(self, ticker: str, check: str, severity: str, detail: str) -> None:
        self.issues.append({"ticker": ticker, "check": check, "severity": severity, "detail": detail})

    @property
    def ok(self) -> bool:
        return not any(i["severity"] == "error" for i in self.issues)

    def errors_for(self, ticker: str) -> list[dict]:
        return [i for i in self.issues if i["ticker"] == ticker and i["severity"] == "error"]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.issues, columns=["ticker", "check", "severity", "detail"])


def validate_prices(
    prices: pd.DataFrame,
    max_daily_move: float = 0.25,
    min_history_days: int = 252,
    stale_days: int = 10,
    max_nan_frac: float = 0.05,
) -> ValidationReport:
    """Validate a wide price frame (index = dates, columns = tickers)."""
    rep = ValidationReport()
    if prices is None or prices.empty:
        rep.add("*", "empty", "error", "no price data")
        return rep
    if not prices.index.is_monotonic_increasing:
        rep.add("*", "index_order", "warning", "index not sorted ascending")
    if prices.index.has_duplicates:
        rep.add("*", "duplicates", "warning", f"{prices.index.duplicated().sum()} duplicate dates")

    for t in prices.columns:
        s = prices[t]
        valid = s.dropna()
        if valid.empty:
            rep.add(t, "empty", "error", "no valid prices")
            continue
        first, last = valid.index[0], valid.index[-1]
        span = s.loc[first:last]
        nan_frac = float(span.isna().mean())
        if nan_frac > max_nan_frac:
            rep.add(t, "missing", "warning", f"{nan_frac:.1%} missing values inside history")
        if (valid <= 0).any():
            rep.add(t, "non_positive", "error", f"{int((valid <= 0).sum())} non-positive prices")
        if len(valid) < min_history_days:
            rep.add(t, "short_history", "warning", f"{len(valid)} days < {min_history_days}")
        rets = valid.pct_change().dropna()
        jumps = rets[rets.abs() > max_daily_move]
        if not jumps.empty:
            rep.add(
                t, "jump", "warning",
                f"{len(jumps)} daily moves > {max_daily_move:.0%} (max {jumps.abs().max():.1%} on {jumps.abs().idxmax().date()})",
            )
        run = _longest_flat_run(valid)
        if run >= stale_days:
            rep.add(t, "stale", "warning", f"price unchanged for {run} consecutive days")
    return rep


def clean_prices(prices: pd.DataFrame, ffill_limit: int = 5) -> pd.DataFrame:
    """Sort, de-duplicate, null out non-positive values, forward-fill short gaps."""
    if prices is None or prices.empty:
        return prices
    df = prices.copy()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.mask(df <= 0)
    df = df.ffill(limit=ffill_limit)
    return df.dropna(axis=1, how="all")


def validate_dividends(divs: pd.Series, ticker: str, prices: pd.Series | None = None,
                       max_single_yield: float = 0.10) -> ValidationReport:
    rep = ValidationReport()
    if divs is None or divs.empty:
        return rep
    if (divs <= 0).any():
        rep.add(ticker, "dividend_non_positive", "error", "non-positive dividend amounts")
    if prices is not None and not prices.dropna().empty:
        px = prices.dropna().reindex(divs.index, method="ffill")
        y = (divs / px).dropna()
        big = y[y > max_single_yield]
        if not big.empty:
            rep.add(ticker, "dividend_outlier", "warning",
                    f"{len(big)} payments > {max_single_yield:.0%} of price (special/return of capital?)")
    return rep


def _longest_flat_run(s: pd.Series) -> int:
    v = s.to_numpy()
    if len(v) < 2:
        return 0
    same = np.isclose(np.diff(v), 0.0)
    best = cur = 0
    for flag in same:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return best
