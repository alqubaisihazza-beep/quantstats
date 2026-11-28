"""Performance and risk metrics, risk decomposition, limits and stress tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

PERIODS = 252


# --------------------------------------------------------------------------- basic metrics
def annualized_return(returns: pd.Series, periods: int = PERIODS) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    growth = float((1 + r).prod())
    years = len(r) / periods
    return growth ** (1 / years) - 1 if growth > 0 and years > 0 else np.nan


def annualized_volatility(returns: pd.Series, periods: int = PERIODS) -> float:
    r = returns.dropna()
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else np.nan


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods: int = PERIODS) -> float:
    r = returns.dropna() - rf / periods
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if len(r) > 1 and sd > 0 else np.nan


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods: int = PERIODS) -> float:
    r = returns.dropna() - rf / periods
    downside = np.sqrt((np.minimum(r, 0) ** 2).mean())
    return float(r.mean() / downside * np.sqrt(periods)) if len(r) > 1 and downside > 0 else np.nan


def drawdown_series(returns_or_equity: pd.Series, is_equity: bool = False) -> pd.Series:
    eq = returns_or_equity if is_equity else (1 + returns_or_equity.fillna(0)).cumprod()
    return eq / eq.cummax() - 1


def max_drawdown(returns_or_equity: pd.Series, is_equity: bool = False) -> float:
    s = returns_or_equity.dropna()
    return float(drawdown_series(s, is_equity).min()) if not s.empty else np.nan


def calmar_ratio(returns: pd.Series, periods: int = PERIODS) -> float:
    mdd = max_drawdown(returns)
    cagr = annualized_return(returns, periods)
    return float(cagr / abs(mdd)) if mdd and mdd < 0 else np.nan


def momentum(prices: pd.Series, lookback: int = 252, skip: int = 21) -> float:
    """12-1 momentum: return from t-lookback to t-skip."""
    p = prices.dropna()
    if len(p) <= lookback:
        return np.nan
    return float(p.iloc[-1 - skip] / p.iloc[-1 - lookback] - 1)


def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    df = pd.concat([returns, benchmark], axis=1).dropna()
    if len(df) < 2:
        return np.nan
    var_b = df.iloc[:, 1].var()
    return float(df.iloc[:, 0].cov(df.iloc[:, 1]) / var_b) if var_b > 0 else np.nan


# --------------------------------------------------------------------------- tail risk
def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR as a positive loss fraction."""
    r = returns.dropna()
    return float(-np.percentile(r, (1 - confidence) * 100)) if not r.empty else np.nan


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    r = returns.dropna()
    return float(-(r.mean() + norm.ppf(1 - confidence) * r.std(ddof=1))) if len(r) > 1 else np.nan


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Expected shortfall: mean loss beyond the VaR threshold (positive number)."""
    r = returns.dropna()
    if r.empty:
        return np.nan
    cutoff = np.percentile(r, (1 - confidence) * 100)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if not tail.empty else np.nan


# --------------------------------------------------------------------------- portfolio
def portfolio_returns(weights: pd.Series, returns: pd.DataFrame) -> pd.Series:
    """Daily-rebalanced portfolio returns (weights normalized)."""
    w = weights.reindex(returns.columns).fillna(0.0)
    if w.sum() == 0:
        return pd.Series(0.0, index=returns.index)
    w = w / w.sum()
    return returns.fillna(0.0) @ w


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """Fraction of total portfolio variance contributed by each asset (sums to 1)."""
    w = weights.reindex(cov.columns).fillna(0.0).values
    port_var = float(w @ cov.values @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=cov.columns)
    return pd.Series(w * (cov.values @ w) / port_var, index=cov.columns)


def stress_test(weights: pd.Series, asset_classes: pd.Series, scenarios: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Apply instantaneous shocks per asset class; returns portfolio P&L % by scenario."""
    w = weights / weights.sum()
    rows = []
    for name, shocks in scenarios.items():
        cls = asset_classes.reindex(w.index).fillna("equity")
        impact = sum(float(w[t]) * float(shocks.get(cls[t], 0.0)) for t in w.index)
        rows.append({"scenario": name, "portfolio_return": impact})
    return pd.DataFrame(rows).set_index("scenario")


def summary_metrics(returns: pd.Series, rf: float = 0.0, benchmark: pd.Series | None = None,
                    confidence: float = 0.95) -> dict:
    out = {
        "cagr": annualized_return(returns),
        "volatility": annualized_volatility(returns),
        "sharpe": sharpe_ratio(returns, rf),
        "sortino": sortino_ratio(returns, rf),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns),
        "var": var_historical(returns, confidence),
        "cvar": cvar(returns, confidence),
    }
    if benchmark is not None:
        out["beta"] = beta(returns, benchmark)
    return out


class RiskEngine:
    def __init__(self, returns: pd.DataFrame, rf: float = 0.0, confidence: float = 0.95,
                 benchmark: pd.Series | None = None):
        self.returns = returns.dropna(how="all")
        self.rf = rf
        self.confidence = confidence
        self.benchmark = benchmark

    def covariance(self, annualize: bool = True) -> pd.DataFrame:
        c = self.returns.cov()
        return c * PERIODS if annualize else c

    def correlation(self) -> pd.DataFrame:
        return self.returns.corr()

    def portfolio_returns(self, weights: pd.Series) -> pd.Series:
        return portfolio_returns(weights, self.returns)

    def report(self, weights: pd.Series) -> dict:
        pr = self.portfolio_returns(weights)
        m = summary_metrics(pr, self.rf, self.benchmark, self.confidence)
        m["risk_contributions"] = risk_contributions(weights, self.covariance())
        m["drawdown"] = drawdown_series(pr)
        return m

    def check_limits(self, weights: pd.Series, limits: dict) -> list[str]:
        """Return a list of human-readable limit breaches (empty list = OK)."""
        breaches = []
        w = weights / weights.sum()
        mx = limits.get("max_position_weight")
        if mx is not None:
            for t, v in w[w > mx + 1e-9].items():
                breaches.append(f"{t} weight {v:.1%} exceeds max position {mx:.0%}")
        pr = self.portfolio_returns(w)
        vol_lim = limits.get("max_portfolio_volatility")
        if vol_lim is not None:
            v = annualized_volatility(pr)
            if v > vol_lim:
                breaches.append(f"portfolio volatility {v:.1%} exceeds {vol_lim:.0%}")
        dd_lim = limits.get("max_drawdown_limit")
        if dd_lim is not None:
            dd = max_drawdown(pr)
            if dd < dd_lim:
                breaches.append(f"historical max drawdown {dd:.1%} worse than {dd_lim:.0%}")
        return breaches
