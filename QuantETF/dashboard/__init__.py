"""Streamlit dashboard pages. Each module exposes render(ctx: AppContext)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from data import resolve_path
from data.database import Database
from data.dividend_data import DividendData
from data.market_data import PriceHistory, build_market_data, build_provider
from engines.backtesting import Backtester, BacktestConfig, BacktestResult
from engines.etf_screener import build_universe_frame
from strategies import get_strategy
from strategies.base import StrategyResult


@dataclass
class AppContext:
    cfg: dict
    db: Database
    universe: pd.DataFrame
    prices: PriceHistory
    dividends: dict[str, pd.Series]
    strategy_name: str = "dividend_growth"
    _cache: dict = field(default_factory=dict)

    @property
    def synthetic(self) -> bool:
        return self.cfg.get("data", {}).get("provider") == "synthetic"

    @property
    def rf(self) -> float:
        return float(self.cfg.get("risk_free_rate", 0.0))

    def strategy(self, name: str | None = None) -> StrategyResult:
        name = name or self.strategy_name
        key = ("strategy", name)
        if key not in self._cache:
            self._cache[key] = get_strategy(name, self.universe, self.cfg).build(self.prices, self.dividends)
        return self._cache[key]

    def backtest(self, weights: pd.Series, name: str, **overrides) -> BacktestResult:
        key = ("bt", name, tuple(weights.round(6).items()), tuple(sorted(overrides.items())))
        if key not in self._cache:
            bcfg = BacktestConfig.from_dict({**self.cfg.get("backtest", {}), **overrides})
            bt = Backtester(self.prices, self.dividends, bcfg, benchmark=self.cfg.get("benchmark"), rf=self.rf)
            self._cache[key] = bt.run(weights, name=name)
        return self._cache[key]


def load_context(cfg: dict, start: str | None = None, refresh: bool = False) -> AppContext:
    db = Database(resolve_path(cfg["data"]["database"]))
    universe = build_universe_frame(cfg["universe"])
    tickers = list(universe.index)
    bench = cfg.get("benchmark")
    if bench and bench not in tickers:
        tickers.append(bench)
    start = start or cfg["data"].get("start_date", "2015-01-01")
    prices = build_market_data(cfg, db).get_prices(tickers, start, refresh=refresh)
    dividends = DividendData(build_provider(cfg), db).get_dividends(prices.tickers, start, refresh=refresh)
    return AppContext(cfg, db, universe, prices, dividends)


def pct(x, digits: int = 1) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}%}"
