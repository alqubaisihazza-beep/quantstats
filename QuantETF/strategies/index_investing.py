from strategies.base import Strategy


class IndexInvestingStrategy(Strategy):
    name = "index_investing"
    description = "Classic low-cost three-fund portfolio: US total market / international / aggregate bonds."
    screen = {"max_expense_ratio": 0.0010, "min_history_years": 3}
    score_weights = {"expense_ratio": 0.70, "sharpe": 0.15, "volatility": 0.15}
    optimizer = "equal_weight"
    buckets = {
        "us_equity": {"weight": 0.60, "categories": ["us_broad", "us_large"], "n": 1},
        "international": {"weight": 0.25, "categories": ["intl_broad", "intl_developed"], "n": 1},
        "bonds": {"weight": 0.15, "categories": ["bond_aggregate"], "n": 1},
    }
