from strategies.base import Strategy


class BalancedStrategy(Strategy):
    name = "balanced"
    description = "55/35/10 equity/bond/real-asset mix; best-scoring ETFs per sleeve, risk parity inside each sleeve."
    screen = {"max_expense_ratio": 0.0050, "min_history_years": 3}
    score_weights = {"sharpe": 0.35, "max_drawdown": 0.25, "volatility": 0.15, "expense_ratio": 0.15, "yield": 0.10}
    optimizer = "risk_parity"
    buckets = {
        "equity": {"weight": 0.55, "asset_classes": ["equity"], "n": 3},
        "bonds": {"weight": 0.35, "asset_classes": ["bond"], "n": 2},
        "real_assets": {"weight": 0.10, "asset_classes": ["real_estate", "commodity"], "n": 2},
    }
