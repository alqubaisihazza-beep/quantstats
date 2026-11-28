"""Strategy registry."""

from strategies.balanced import BalancedStrategy
from strategies.base import Strategy, StrategyResult
from strategies.dividend_growth import DividendGrowthStrategy
from strategies.high_income import HighIncomeStrategy
from strategies.index_investing import IndexInvestingStrategy

STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (DividendGrowthStrategy, HighIncomeStrategy, BalancedStrategy, IndexInvestingStrategy)
}


def get_strategy(name: str, universe, cfg: dict | None = None) -> Strategy:
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy {name!r}; choose from {list(STRATEGIES)}")
    cfg = cfg or {}
    overrides = (cfg.get("strategies") or {}).get(name) or {}
    return STRATEGIES[name](universe, overrides, rf=cfg.get("risk_free_rate", 0.0))


__all__ = ["STRATEGIES", "Strategy", "StrategyResult", "get_strategy"]
