import os

import pandas as pd
import pytest

from brokers.ibkr import IBKRBroker, order_spec
from brokers.manual_execution import ManualExecutionBroker
from engines.backtesting import Backtester, BacktestConfig
from engines.rebalancing import Order
from strategies import STRATEGIES, get_strategy


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategy_builds(name, cfg, universe, prices, dividends):
    strat = get_strategy(name, universe, cfg)
    res = strat.build(prices, dividends)
    assert abs(res.weights.sum() - 1) < 1e-6
    assert (res.weights > 0).all()
    assert set(res.weights.index) <= set(res.screen.index[res.screen["passed"]])


def test_index_investing_three_funds(cfg, universe, prices, dividends):
    w = get_strategy("index_investing", universe, cfg).build(prices, dividends).weights
    assert len(w) == 3
    assert abs(w.max() - 0.60) < 1e-9
    cats = universe.loc[w.index, "category"]
    assert "bond_aggregate" in set(cats)


def test_balanced_sleeves(cfg, universe, prices, dividends):
    w = get_strategy("balanced", universe, cfg).build(prices, dividends).weights
    by_class = w.groupby(universe.loc[w.index, "asset_class"]).sum()
    assert abs(by_class["equity"] - 0.55) < 1e-6 and abs(by_class["bond"] - 0.35) < 1e-6


def test_high_income_respects_yield(cfg, universe, prices, dividends):
    res = get_strategy("high_income", universe, cfg).build(prices, dividends)
    assert (res.screen.loc[res.weights.index, "yield"] >= 0.03).all()
    assert res.weights.max() <= 0.30 + 1e-6


def test_override_validation(universe):
    with pytest.raises(ValueError):
        STRATEGIES["balanced"](universe, {"nonsense": 1})


def test_dynamic_backtest(cfg, universe, prices, dividends):
    strat = get_strategy("dividend_growth", universe, cfg)
    bt = Backtester(prices, dividends, BacktestConfig(rebalance="annual"))
    res = bt.run(strat.target_fn(), start="2019-01-01", name="dg")
    assert res.metrics["n_rebalances"] >= 4
    assert len(res.trades) > 0


def test_manual_broker_flow(db, tmp_path):
    b = ManualExecutionBroker(db, "acct", tmp_path)
    b.set_cash(1000)
    out = b.submit_orders([Order("VTI", "BUY", 3, 250.0)])
    assert out["csv"].exists() and out["ticket"].exists()
    assert len(b.pending()) == 1
    b.confirm_fill(out["trade_ids"][0], price=249.0)
    assert b.get_positions().loc["VTI", "shares"] == 3
    assert abs(b.get_cash() - (1000 - 747)) < 1e-9
    assert b.pending().empty
    with pytest.raises(ValueError):
        b.confirm_fill(out["trade_ids"][0], price=249.0)


def test_manual_import_csv(db, tmp_path):
    p = tmp_path / "pos.csv"
    p.write_text("ticker,shares,cost_basis\nvti,10,2000\nCASH,500,\n")
    b = ManualExecutionBroker(db, "x", tmp_path)
    assert b.import_positions_csv(p) == 1
    assert b.get_positions().loc["VTI", "shares"] == 10 and b.get_cash() == 500


def test_ibkr_order_spec_and_safety(monkeypatch):
    s = order_spec(Order("VTI", "BUY", 5, 100.0), "LMT", 10)
    assert s["limit_price"] == 100.1 and s["action"] == "BUY"
    assert order_spec(Order("VTI", "SELL", 5, 100.0), "LMT", 10)["limit_price"] == 99.9
    monkeypatch.setenv("QUANTETF_ALLOW_LIVE", "1")
    assert not IBKRBroker(dry_run=True).live_allowed
    assert IBKRBroker(dry_run=False).live_allowed
    monkeypatch.setenv("QUANTETF_ALLOW_LIVE", "0")
    assert not IBKRBroker(dry_run=False).live_allowed
    with pytest.raises(RuntimeError):
        IBKRBroker().get_positions()
