import numpy as np
import pandas as pd
import pytest

from engines import risk_engine as rk
from engines.backtesting import Backtester, BacktestConfig
from engines.dividend_engine import DividendEngine
from engines.etf_scoring import compute_metrics, score_etfs
from engines.etf_screener import ETFScreener, ScreenCriteria
from engines.portfolio_optimizer import PortfolioOptimizer, cap_weights
from engines.rebalancing import Rebalancer, is_calendar_rebalance
from engines.reinvestment import simulate_drip


# ---------------------------------------------------------------- risk
def test_basic_metrics():
    r = pd.Series([0.01] * 252)
    assert abs(rk.annualized_return(r) - (1.01 ** 252 - 1)) < 1e-9
    eq = pd.Series([100, 120, 90, 130.0])
    assert abs(rk.max_drawdown(eq, is_equity=True) - (-0.25)) < 1e-12
    r2 = pd.Series(np.r_[np.full(94, 0.01), np.full(6, -0.05)])
    assert abs(rk.var_historical(r2, 0.95) - 0.05) < 1e-9
    assert rk.cvar(r2, 0.95) >= rk.var_historical(r2, 0.95) - 1e-12


def test_risk_contributions_sum_to_one(toy_returns):
    cov = toy_returns.cov()
    rc = rk.risk_contributions(pd.Series(1 / 3, index=cov.columns), cov)
    assert abs(rc.sum() - 1) < 1e-9 and rc["HIGH"] > rc["LOW"]


def test_stress_test():
    w = pd.Series({"A": 0.6, "B": 0.4})
    cls = pd.Series({"A": "equity", "B": "bond"})
    out = rk.stress_test(w, cls, {"crash": {"equity": -0.3, "bond": 0.05}})
    assert abs(out.loc["crash", "portfolio_return"] - (-0.16)) < 1e-12


def test_check_limits(toy_returns):
    eng = rk.RiskEngine(toy_returns)
    breaches = eng.check_limits(pd.Series({"LOW": 0.1, "MID": 0.1, "HIGH": 0.8}), {"max_position_weight": 0.5})
    assert any("HIGH" in b for b in breaches)


# ---------------------------------------------------------------- optimizer
@pytest.mark.parametrize("method", ["equal_weight", "inverse_volatility", "min_variance", "max_sharpe", "risk_parity"])
def test_optimizer_constraints(toy_returns, method):
    w = PortfolioOptimizer(toy_returns, max_weight=0.6).optimize(method)
    assert abs(w.sum() - 1) < 1e-6
    assert (w >= -1e-9).all() and (w <= 0.6 + 1e-6).all()


def test_min_variance_prefers_low_vol(toy_returns):
    w = PortfolioOptimizer(toy_returns, shrinkage=0).optimize("min_variance")
    assert w["LOW"] > 0.8


def test_risk_parity_equalizes(toy_returns):
    opt = PortfolioOptimizer(toy_returns, shrinkage=0)
    w = opt.optimize("risk_parity")
    rc = rk.risk_contributions(w, opt.cov)
    assert rc.max() - rc.min() < 0.02


def test_cap_weights():
    w = cap_weights(pd.Series({"a": 10, "b": 1, "c": 1.0}), 0.5)
    assert abs(w["a"] - 0.5) < 1e-9 and abs(w.sum() - 1) < 1e-9 and abs(w["b"] - 0.25) < 1e-9


# ---------------------------------------------------------------- screener / scoring
def test_screener(universe, prices, dividends):
    res = ETFScreener(universe).screen(ScreenCriteria(max_expense_ratio=0.001, asset_classes=["bond"]), prices, dividends)
    passed = set(res.index[res["passed"]])
    assert passed == {"BND", "AGG"}
    assert "expense ratio" in res.loc["HYG", "reasons"]
    assert "asset class" in res.loc["VTI", "reasons"]


def test_scoring_direction(universe, prices, dividends):
    m = compute_metrics(prices, dividends, universe)
    s = score_etfs(m, {"expense_ratio": 1.0})
    assert s.index[0] in {"VTI", "VOO", "BND", "AGG"}
    assert s.index[-1] == "DIVO"
    assert s["score"].between(0, 100).all() and list(s["rank"]) == list(range(1, len(s) + 1))


# ---------------------------------------------------------------- rebalancing
def test_calendar_rebalance():
    d = pd.Timestamp
    assert is_calendar_rebalance(d("2024-04-01"), d("2024-03-29"), "quarterly")
    assert not is_calendar_rebalance(d("2024-05-01"), d("2024-04-30"), "quarterly")
    assert is_calendar_rebalance(d("2024-05-01"), d("2024-04-30"), "monthly")
    assert is_calendar_rebalance(d("2025-01-02"), d("2024-12-31"), "annual")


def test_generate_orders():
    reb = Rebalancer(threshold=0.05, min_trade_value=10, allow_fractional=False)
    prices = pd.Series({"A": 100.0, "B": 50.0, "C": 20.0})
    holdings = {"A": 80, "B": 40, "C": 50}          # 8000 / 2000 / 1000 + 0 cash
    target = pd.Series({"A": 0.5, "B": 0.5})        # drop C
    assert reb.needs_rebalance(holdings, prices, target)
    orders = {o.ticker: o for o in reb.generate_orders(holdings, prices, target)}
    assert orders["C"].side == "SELL" and orders["C"].shares == 50
    assert orders["A"].side == "SELL" and orders["A"].shares == 25
    assert orders["B"].side == "BUY" and orders["B"].shares == 70
    spent = sum(o.est_value for o in orders.values() if o.side == "BUY")
    raised = sum(o.est_value for o in orders.values() if o.side == "SELL")
    assert spent <= raised + 1e-9


def test_cash_flow_orders_buy_only():
    reb = Rebalancer(min_trade_value=1, allow_fractional=True)
    prices = pd.Series({"A": 10.0, "B": 10.0})
    orders = reb.cash_flow_orders({"A": 100, "B": 0}, prices, pd.Series({"A": 0.5, "B": 0.5}), cash=500)
    assert len(orders) == 1 and orders[0].ticker == "B" and abs(orders[0].shares - 50) < 1e-9


# ---------------------------------------------------------------- dividends / DRIP
def _flat_prices(tickers, n=300, price=100.0):
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({t: price for t in tickers}, index=idx)


def test_drip_same_policy():
    close = _flat_prices(["A"])
    divs = {"A": pd.Series([1.0, 1.0], index=close.index[[50, 150]])}
    res = simulate_drip(close, divs, {"A": 100})
    assert abs(res.total_income - (100 + 101)) < 1e-9
    assert abs(res.shares["A"].iloc[-1] - (100 * 1.01 * 1.01)) < 1e-9
    assert abs(res.cash.iloc[-1]) < 1e-9


def test_drip_cash_and_underweight():
    close = _flat_prices(["A", "B"])
    divs = {"A": pd.Series([2.0], index=[close.index[10]])}
    cash = simulate_drip(close, divs, {"A": 100, "B": 100}, policy="cash")
    assert cash.cash.iloc[-1] == 200 and cash.shares["A"].iloc[-1] == 100
    uw = simulate_drip(close, divs, {"A": 100, "B": 50}, policy="underweight",
                       target_weights=pd.Series({"A": 0.5, "B": 0.5}))
    assert uw.shares["B"].iloc[-1] == 52 and uw.shares["A"].iloc[-1] == 100


def test_dividend_engine_income():
    close = _flat_prices(["A"], n=400, price=50.0)
    pay = close.index[[30, 163, 226, 289, 352]]
    divs = {"A": pd.Series(0.5, index=pay)}
    eng = DividendEngine(divs, close)
    inc = eng.forward_income({"A": 100})
    assert abs(inc.loc["A", "annual_income"] - 200) < 1e-9         # 4 payments in ttm * 0.5 * 100
    assert abs(inc.loc["A", "yield"] - 0.04) < 1e-9
    cal = eng.income_calendar({"A": 100}, start=close.index[-1])
    assert abs(cal["total"].sum() - 200) < 1e-9
    proj = DividendEngine.project_income(10000, 0.04, 0.05, 0.0, years=2, reinvest=False)
    assert abs(proj.loc[1, "income"] - 400) < 1e-9 and abs(proj.loc[2, "income"] - 420) < 1e-9


# ---------------------------------------------------------------- backtest
class _P:
    """Minimal PriceHistory stand-in."""

    def __init__(self, close):
        self.close = close
        self.adj_close = close

    def slice(self, start=None, end=None):
        return _P(self.close.loc[start:end])


def test_backtest_flat_prices_with_dividends():
    close = _flat_prices(["A", "B"], n=260)
    divs = {"A": pd.Series(1.0, index=close.index[[60, 120, 180, 240]])}
    cfg = BacktestConfig(initial_capital=10000, rebalance="none", transaction_cost_bps=0)
    res = Backtester(_P(close), divs, cfg).run(pd.Series({"A": 0.5, "B": 0.5}))
    # 50 shares of A, DRIP at flat price -> 50 * 1.01^4 shares
    assert abs(res.equity.iloc[-1] - (5000 * 1.01 ** 4 + 5000)) < 1e-6
    assert res.metrics["total_income"] > 200


def test_backtest_costs_and_payout():
    close = _flat_prices(["A"], n=100)
    divs = {"A": pd.Series(1.0, index=[close.index[50]])}
    cfg = BacktestConfig(initial_capital=10000, rebalance="none", transaction_cost_bps=10, reinvest_dividends=False)
    res = Backtester(_P(close), divs, cfg).run(pd.Series({"A": 1.0}))
    assert abs(res.equity.iloc[-1] - 10000 * (1 - 0.001)) < 1e-6        # dividend withdrawn
    assert abs(res.metrics["total_return"] - (0.999 * 1.01 - 1) - 0.0) < 0.002  # TWR counts the payout


def test_backtest_contributions_not_counted_as_return():
    close = _flat_prices(["A"], n=300)
    cfg = BacktestConfig(initial_capital=1000, rebalance="monthly", transaction_cost_bps=0, monthly_contribution=100)
    res = Backtester(_P(close), {}, cfg).run(pd.Series({"A": 1.0}))
    assert abs(res.metrics["total_return"]) < 1e-9
    assert res.metrics["total_contributions"] > 0
    assert abs(res.equity.iloc[-1] - (1000 + res.metrics["total_contributions"])) < 1e-6


def test_backtest_real_universe(prices, dividends, tmp_path):
    cfg = BacktestConfig(rebalance="quarterly")
    res = Backtester(prices, dividends, cfg, benchmark="SPY").run(pd.Series({"VTI": 0.6, "BND": 0.4}), name="t")
    assert res.metrics["n_rebalances"] > 10
    assert {"cagr", "sharpe", "max_drawdown", "beta", "benchmark_cagr"} <= set(res.metrics)
    md, js = res.to_report(tmp_path)
    assert md.exists() and js.exists()
