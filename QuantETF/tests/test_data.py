import numpy as np
import pandas as pd

from data.data_validation import clean_prices, validate_prices
from data.database import Database
from data.dividend_data import dividend_cagr, dividend_consistency, payment_frequency, trailing_yield, ttm_dividends
from data.market_data import MarketData, SyntheticProvider


def test_database_roundtrip(db):
    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({"close": [1, 2, 3, 4, 5.0], "adj_close": [1, 2, 3, 4, 5.0], "volume": [10] * 5}, index=idx)
    assert db.save_prices("AAA", df) == 5
    out = db.load_prices("AAA", "2024-01-02")
    assert len(out) == 4 and out["close"].iloc[-1] == 5.0
    db.save_dividends("AAA", pd.Series([0.1, 0.2], index=idx[:2]))
    assert abs(db.load_dividends("AAA").sum() - 0.3) < 1e-12


def test_apply_fill_average_cost(db):
    db.set_cash("a", 1000)
    db.apply_fill("a", "X", "BUY", 10, 50)
    db.apply_fill("a", "X", "BUY", 10, 30)
    h = db.get_holdings("a")
    assert h.loc["X", "shares"] == 20 and h.loc["X", "cost_basis"] == 800
    db.apply_fill("a", "X", "SELL", 5, 60)
    h = db.get_holdings("a")
    assert h.loc["X", "shares"] == 15 and abs(h.loc["X", "cost_basis"] - 600) < 1e-9
    assert abs(db.get_cash("a") - (1000 - 800 + 300)) < 1e-9


def test_validation_flags_problems():
    idx = pd.bdate_range("2024-01-01", periods=300)
    good = pd.Series(np.linspace(100, 120, 300), index=idx)
    bad = good.copy()
    bad.iloc[50] = -1
    jump = good.copy()
    jump.iloc[100:] *= 1.5
    rep = validate_prices(pd.DataFrame({"GOOD": good, "BAD": bad, "JUMP": jump}))
    checks = {(i["ticker"], i["check"]) for i in rep.issues}
    assert ("BAD", "non_positive") in checks
    assert ("JUMP", "jump") in checks
    assert not any(i["ticker"] == "GOOD" for i in rep.issues)
    assert not rep.ok


def test_clean_prices_ffill_and_nonpositive():
    idx = pd.bdate_range("2024-01-01", periods=5)
    df = pd.DataFrame({"A": [1.0, np.nan, 0.0, 4.0, 5.0], "B": [np.nan] * 5}, index=idx)
    out = clean_prices(df, ffill_limit=5)
    assert list(out.columns) == ["A"]
    assert out["A"].tolist() == [1.0, 1.0, 1.0, 4.0, 5.0]


def test_dividend_stats():
    dates = pd.to_datetime([f"{y}-{m:02d}-15" for y in range(2018, 2024) for m in (3, 6, 9, 12)])
    amounts = [0.25 * 1.10 ** (d.year - 2018) for d in dates]
    s = pd.Series(amounts, index=dates)
    assert payment_frequency(s) == "quarterly"
    assert abs(dividend_cagr(s, 5, as_of="2024-06-01") - 0.10) < 1e-9
    assert dividend_consistency(s, 5, as_of="2024-06-01") == 1.0
    assert abs(ttm_dividends(s) - 4 * 0.25 * 1.1 ** 5) < 1e-9
    px = pd.Series(100.0, index=pd.bdate_range("2023-01-01", "2023-12-31"))
    assert abs(trailing_yield(s, px) - ttm_dividends(s, px.index[-1]) / 100) < 1e-12


def test_market_data_caches(db, cfg):
    class Counting(SyntheticProvider):
        calls = 0

        def fetch_prices(self, t, s, e=None):
            Counting.calls += 1
            return super().fetch_prices(t, s, e)

    md = MarketData(Counting(cfg["universe"]), db)
    p1 = md.get_prices(["VTI", "BND"], "2020-01-01", "2022-12-30")
    p2 = md.get_prices(["VTI", "BND"], "2020-01-01", "2022-12-30")
    assert Counting.calls == 2
    pd.testing.assert_frame_equal(p1.close, p2.close, check_freq=False)
    assert p1.adj_close["BND"].pct_change().sum() > p1.close["BND"].pct_change().sum()  # dividends add return
