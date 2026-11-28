from strategies.base import Strategy


class HighIncomeStrategy(Strategy):
    name = "high_income"
    description = "Maximize current yield across equity income, covered-call, REIT and credit ETFs (inverse-vol)."
    screen = {
        "min_yield": 0.030,
        "max_expense_ratio": 0.0060,
        "min_history_years": 3,
        "exclude_categories": ["bond_treasury_long"],
    }
    score_weights = {
        "yield": 0.40,
        "div_consistency": 0.15,
        "max_drawdown": 0.15,
        "volatility": 0.10,
        "sharpe": 0.10,
        "expense_ratio": 0.10,
    }
    optimizer = "inverse_volatility"
    top_n = 6
    max_weight = 0.30
