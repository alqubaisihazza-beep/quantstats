from strategies.base import Strategy


class DividendGrowthStrategy(Strategy):
    name = "dividend_growth"
    description = "Equity ETFs with fast, consistent dividend growth and solid risk-adjusted returns (risk parity)."
    screen = {
        "asset_classes": ["equity"],
        "categories": ["dividend_growth", "high_dividend", "us_large", "us_broad"],
        "max_expense_ratio": 0.0060,
        "min_yield": 0.010,
        "min_history_years": 3,
    }
    score_weights = {
        "div_cagr_5y": 0.30,
        "div_consistency": 0.20,
        "yield": 0.10,
        "sharpe": 0.15,
        "max_drawdown": 0.10,
        "expense_ratio": 0.15,
    }
    optimizer = "risk_parity"
    top_n = 5
    max_weight = 0.35
