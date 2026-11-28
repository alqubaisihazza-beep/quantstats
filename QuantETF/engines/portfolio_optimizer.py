"""Long-only portfolio construction: equal / inverse-vol / min-variance / max-Sharpe / risk parity / score."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PERIODS = 252
METHODS = ("equal_weight", "inverse_volatility", "min_variance", "max_sharpe", "risk_parity", "score_weighted")


def cap_weights(w: pd.Series, max_weight: float, min_weight: float = 0.0, iters: int = 100) -> pd.Series:
    """Iteratively clip to [min, max] and redistribute excess proportionally."""
    w = w.clip(lower=0).astype(float)
    w = w / w.sum() if w.sum() > 0 else pd.Series(1 / len(w), index=w.index)
    max_weight = max(max_weight, 1 / len(w))
    for _ in range(iters):
        over = w > max_weight + 1e-12
        under = w < min_weight - 1e-12
        if not over.any() and not under.any():
            break
        w[over] = max_weight
        w[under] = min_weight
        free = ~(over | under)
        rem = 1 - w[~free].sum()
        if free.any() and w[free].sum() > 0:
            w[free] = w[free] / w[free].sum() * rem
        elif free.any():
            w[free] = rem / free.sum()
    return w / w.sum()


class PortfolioOptimizer:
    def __init__(self, returns: pd.DataFrame, min_weight: float = 0.0, max_weight: float = 1.0,
                 rf: float = 0.0, shrinkage: float = 0.1):
        self.returns = returns.dropna(how="all").dropna(axis=1, how="any")
        if self.returns.shape[1] == 0:
            raise ValueError("No assets with complete return history")
        self.assets = list(self.returns.columns)
        n = len(self.assets)
        self.max_weight = max(max_weight, 1 / n)            # keep problem feasible
        self.min_weight = min(min_weight, 1 / n)
        self.rf = rf
        self.mu = self.returns.mean() * PERIODS
        sample = self.returns.cov() * PERIODS
        target = np.diag(np.diag(sample.values))
        self.cov = pd.DataFrame((1 - shrinkage) * sample.values + shrinkage * target,
                                index=self.assets, columns=self.assets)

    # ------------------------------------------------------------------ helpers
    def _bounds(self):
        return [(self.min_weight, self.max_weight)] * len(self.assets)

    def _solve(self, objective, x0=None) -> pd.Series:
        n = len(self.assets)
        x0 = np.full(n, 1 / n) if x0 is None else x0
        res = minimize(objective, x0, method="SLSQP", bounds=self._bounds(),
                       constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}],
                       options={"maxiter": 500, "ftol": 1e-12})
        x = res.x if res.success else x0
        return self._clean(pd.Series(x, index=self.assets))

    def _clean(self, w: pd.Series) -> pd.Series:
        w = w.clip(lower=0)
        w[w < 1e-4] = 0.0
        return w / w.sum()

    def stats(self, w: pd.Series) -> dict:
        w = w.reindex(self.assets).fillna(0).values
        ret = float(w @ self.mu.values)
        vol = float(np.sqrt(w @ self.cov.values @ w))
        return {"expected_return": ret, "volatility": vol,
                "sharpe": (ret - self.rf) / vol if vol > 0 else np.nan}

    # ------------------------------------------------------------------ methods
    def equal_weight(self) -> pd.Series:
        return pd.Series(1 / len(self.assets), index=self.assets)

    def inverse_volatility(self) -> pd.Series:
        iv = 1 / np.sqrt(np.diag(self.cov.values))
        return cap_weights(pd.Series(iv, index=self.assets), self.max_weight, self.min_weight)

    def min_variance(self) -> pd.Series:
        C = self.cov.values
        return self._solve(lambda x: x @ C @ x)

    def max_sharpe(self) -> pd.Series:
        C, mu, rf = self.cov.values, self.mu.values, self.rf

        def neg_sharpe(x):
            vol = np.sqrt(x @ C @ x)
            return -(x @ mu - rf) / vol if vol > 0 else 1e6

        return self._solve(neg_sharpe)

    def risk_parity(self) -> pd.Series:
        C = self.cov.values
        n = len(self.assets)

        def obj(x):
            port_var = x @ C @ x
            rc = x * (C @ x) / port_var
            return float(((rc - 1 / n) ** 2).sum())

        x0 = self.inverse_volatility().values
        return self._solve(obj, x0)

    def score_weighted(self, scores: pd.Series) -> pd.Series:
        s = scores.reindex(self.assets).fillna(0).clip(lower=0)
        if s.sum() == 0:
            return self.equal_weight()
        return cap_weights(s, self.max_weight, self.min_weight)

    def optimize(self, method: str = "max_sharpe", scores: pd.Series | None = None) -> pd.Series:
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
        if len(self.assets) == 1:
            return pd.Series(1.0, index=self.assets)
        if method == "score_weighted":
            if scores is None:
                raise ValueError("score_weighted requires scores")
            return self.score_weighted(scores)
        return getattr(self, method)()

    def efficient_frontier(self, points: int = 20) -> pd.DataFrame:
        C, mu = self.cov.values, self.mu.values
        lo, hi = float(self.stats(self.min_variance())["expected_return"]), float(mu.max())
        rows = []
        for target in np.linspace(lo, hi, points):
            cons = [{"type": "eq", "fun": lambda x: x.sum() - 1},
                    {"type": "ineq", "fun": lambda x, t=target: x @ mu - t}]
            res = minimize(lambda x: x @ C @ x, np.full(len(mu), 1 / len(mu)), method="SLSQP",
                           bounds=self._bounds(), constraints=cons)
            if res.success:
                rows.append({"expected_return": float(res.x @ mu), "volatility": float(np.sqrt(res.fun))})
        return pd.DataFrame(rows)
