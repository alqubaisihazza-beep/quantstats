"""Multi-factor ETF scoring: compute metrics, percentile-rank, weighted composite (0-100)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.dividend_data import dividend_cagr, dividend_consistency, trailing_yield
from engines import risk_engine as rk

# True  -> higher value is better; False -> lower is better
DIRECTION = {
    "cagr": True,
    "volatility": False,
    "sharpe": True,
    "sortino": True,
    "max_drawdown": True,      # stored negative; closer to 0 is better
    "momentum": True,
    "yield": True,
    "div_cagr_5y": True,
    "div_consistency": True,
    "expense_ratio": False,
}


def compute_metrics(prices, dividends: dict | None, universe: pd.DataFrame | None = None,
                    rf: float = 0.0, lookback_days: int = 756) -> pd.DataFrame:
    rets = prices.returns(total=True)
    rows = {}
    for t in prices.tickers:
        r = rets[t].dropna().iloc[-lookback_days:]
        divs = (dividends or {}).get(t, pd.Series(dtype=float))
        rows[t] = {
            "cagr": rk.annualized_return(r),
            "volatility": rk.annualized_volatility(r),
            "sharpe": rk.sharpe_ratio(r, rf),
            "sortino": rk.sortino_ratio(r, rf),
            "max_drawdown": rk.max_drawdown(r),
            "momentum": rk.momentum(prices.adj_close[t]),
            "yield": trailing_yield(divs, prices.close[t]) if not divs.empty else 0.0,
            "div_cagr_5y": dividend_cagr(divs, 5, as_of=prices.close.index[-1]),
            "div_consistency": dividend_consistency(divs, 5, as_of=prices.close.index[-1]),
            "expense_ratio": (universe["expense_ratio"].get(t, np.nan)
                              if universe is not None and "expense_ratio" in universe else np.nan),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def score_etfs(metrics: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Percentile-rank each weighted metric (direction-aware) and combine.

    Missing values get a neutral 0.5 rank. Returns metrics + `<m>_rank` + `score` + `rank`,
    sorted by score descending.
    """
    w = {k: float(v) for k, v in weights.items() if k in metrics.columns and v}
    if not w:
        raise ValueError("No scoring weights match available metrics")
    total = sum(abs(v) for v in w.values())
    out = metrics.copy()
    score = pd.Series(0.0, index=metrics.index)
    for m, wt in w.items():
        col = metrics[m].astype(float)
        asc = DIRECTION.get(m, True)
        rank = col.rank(pct=True, ascending=asc) if col.notna().sum() > 1 else pd.Series(0.5, index=col.index)
        rank = rank.fillna(0.5)
        out[f"{m}_rank"] = rank
        score += wt / total * rank
    out["score"] = (score * 100).round(2)
    out = out.sort_values("score", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)
    return out
