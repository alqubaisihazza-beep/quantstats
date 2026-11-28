"""Strategy pipeline: screen -> score -> select -> optimize (optionally per asset-class bucket)."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pandas as pd

from engines.etf_scoring import compute_metrics, score_etfs
from engines.etf_screener import ETFScreener, ScreenCriteria
from engines.portfolio_optimizer import PortfolioOptimizer


@dataclass
class StrategyResult:
    screen: pd.DataFrame
    scores: pd.DataFrame
    weights: pd.Series
    as_of: pd.Timestamp

    @property
    def selected(self) -> list[str]:
        return list(self.weights.index)


class Strategy:
    name = "base"
    description = ""
    screen: dict = {}
    score_weights: dict = {"sharpe": 1.0}
    optimizer: str = "max_sharpe"
    top_n: int = 5
    min_weight: float = 0.0
    max_weight: float = 0.40
    lookback_days: int = 756
    min_history_days: int = 252
    # Optional: {bucket: {"weight": float, "categories": [...] | "asset_classes": [...], "n": int, "optimizer": str}}
    buckets: dict | None = None

    def __init__(self, universe: pd.DataFrame, overrides: dict | None = None, rf: float = 0.0):
        self.universe = universe
        self.rf = rf
        for attr in ("screen", "score_weights", "buckets"):
            setattr(self, attr, copy.deepcopy(getattr(self, attr)))
        for k, v in (overrides or {}).items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown strategy setting {k!r} for {self.name}")
            setattr(self, k, v)

    # ------------------------------------------------------------------ pipeline
    def candidates(self, prices, dividends) -> tuple[pd.DataFrame, list[str]]:
        screen = ETFScreener(self.universe).screen(ScreenCriteria.from_dict(self.screen), prices, dividends)
        passed = screen.index[screen["passed"]]
        rets = prices.returns().iloc[-self.lookback_days:]
        enough = [t for t in passed if t in rets.columns and rets[t].notna().sum() >= self.min_history_days]
        return screen, enough

    def build(self, prices, dividends: dict | None = None) -> StrategyResult:
        dividends = dividends or {}
        screen, cands = self.candidates(prices, dividends)
        if not cands:
            raise ValueError(f"{self.name}: no ETFs passed the screen")
        metrics = compute_metrics(prices.subset(cands), dividends, self.universe, self.rf, self.lookback_days)
        scores = score_etfs(metrics, self.score_weights)
        weights = self._bucket_weights(prices, scores) if self.buckets else self._optimize(
            list(scores.index[: self.top_n]), prices, scores, self.optimizer, self.max_weight)
        weights = weights[weights > 1e-4].sort_values(ascending=False)
        return StrategyResult(screen, scores, weights / weights.sum(), prices.close.index[-1])

    def _optimize(self, tickers, prices, scores, method, max_weight) -> pd.Series:
        if len(tickers) == 1:
            return pd.Series(1.0, index=tickers)
        rets = prices.returns()[tickers].iloc[-self.lookback_days:].dropna()
        opt = PortfolioOptimizer(rets, self.min_weight, max_weight, self.rf)
        return opt.optimize(method, scores=scores["score"] if "score" in scores else None)

    def _bucket_weights(self, prices, scores) -> pd.Series:
        parts, total_w = [], 0.0
        for name, b in self.buckets.items():
            meta = self.universe.reindex(scores.index)
            mask = pd.Series(True, index=scores.index)
            if b.get("categories"):
                mask &= meta["category"].isin(b["categories"])
            if b.get("asset_classes"):
                mask &= meta["asset_class"].isin(b["asset_classes"])
            picks = list(scores.index[mask][: b.get("n", 1)])
            if not picks:
                continue
            w = self._optimize(picks, prices, scores, b.get("optimizer", self.optimizer), 1.0)
            parts.append(w * b["weight"])
            total_w += b["weight"]
        if not parts:
            raise ValueError(f"{self.name}: no bucket had eligible ETFs")
        combined = pd.concat(parts).groupby(level=0).sum()
        return combined / total_w                                 # redistribute empty buckets

    # ------------------------------------------------------------------ backtest hook
    def target_fn(self, refit_fallback: pd.Series | None = None):
        """Callable for Backtester: rebuilds the portfolio using only data available at each date."""
        state = {"last": refit_fallback}

        def fn(dt, prices_hist, divs_hist):
            try:
                state["last"] = self.build(prices_hist, divs_hist).weights
            except ValueError:
                if state["last"] is None:
                    raise
            return state["last"]

        return fn
