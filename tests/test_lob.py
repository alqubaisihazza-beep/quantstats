"""
Tests for quantstats.lob (limit order book & imbalance modeling)
"""

import numpy as np
import pandas as pd
import pytest

from quantstats import lob

TICK = 0.01


def simulate_book(n=20000, n_levels=3, seed=7, fill_prob=0.5):
    """
    Queue-depletion book: best queues random-walk, and when one empties that
    quote moves away a tick, widening the spread. A one-tick gap then fills
    from either side at random. Imbalance therefore predicts the next mid
    move, and the book is symmetric between bid and ask.
    """
    rng = np.random.default_rng(seed)
    bid, ask = 10000, 10001  # in ticks
    qb, qa = 50.0, 50.0
    rows = []
    for _ in range(n):
        qb += rng.normal(0, 8)
        qa += rng.normal(0, 8)
        wide = ask - bid > 1
        if qa <= 0:
            ask += 1
            bid += 1 if wide else 0
            qa = rng.uniform(20, 80)
            if wide:
                qb = rng.uniform(5, 30)
        elif qb <= 0:
            bid -= 1
            ask -= 1 if wide else 0
            qb = rng.uniform(20, 80)
            if wide:
                qa = rng.uniform(5, 30)
        elif wide and rng.random() < fill_prob:
            # A new order improves into the gap with a small queue.
            if rng.random() < 0.5:
                bid += 1
                qb = rng.uniform(5, 30)
            else:
                ask -= 1
                qa = rng.uniform(5, 30)
        row = {}
        for lvl in range(1, n_levels + 1):
            row[f"bid_price_{lvl}"] = round((bid - lvl + 1) * TICK, 2)
            row[f"ask_price_{lvl}"] = round((ask + lvl - 1) * TICK, 2)
            row[f"bid_size_{lvl}"] = qb if lvl == 1 else 60.0 + 10 * lvl
            row[f"ask_size_{lvl}"] = qa if lvl == 1 else 60.0 + 10 * lvl
        rows.append(row)
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2024-01-02 09:30", periods=n, freq="100ms")
    return df


@pytest.fixture(scope="module")
def book():
    return simulate_book()


@pytest.fixture
def tiny():
    """Hand-checkable top-of-book frame."""
    return pd.DataFrame(
        {
            "bid": [10.00, 10.00, 10.01, 10.01, 10.00],
            "ask": [10.02, 10.02, 10.02, 10.03, 10.02],
            "bid_size": [100, 120, 30, 30, 50],
            "ask_size": [100, 90, 90, 40, 60],
        }
    )


class TestBookAccess:
    def test_to_book_aliases_and_levels(self, tiny, book):
        out = lob.to_book(tiny)
        assert list(out.columns) == [
            "bid_price_1",
            "bid_size_1",
            "ask_price_1",
            "ask_size_1",
        ]
        assert lob.levels(tiny) == 1
        assert lob.levels(book) == 3

    def test_to_book_custom_mapping(self, tiny):
        raw = tiny.rename(columns={"bid": "bp", "ask": "ap"})
        out = lob.to_book(raw, columns={"bp": "bid_price_1", "ap": "ask_price_1"})
        assert out["ask_price_1"].iloc[0] == 10.02

    def test_to_book_rejects_incomplete(self, tiny):
        with pytest.raises(ValueError, match="level 1"):
            lob.to_book(tiny.drop(columns="ask"))
        bad = tiny.assign(bid_price_2=9.99, bid_size_2=1.0, ask_price_2=10.03)
        with pytest.raises(ValueError, match="level 2"):
            lob.to_book(bad)

    def test_levels_out_of_range(self, tiny):
        with pytest.raises(ValueError):
            lob.depth(tiny, levels=2)


class TestFeatures:
    def test_mid_spread(self, tiny):
        np.testing.assert_allclose(lob.mid_price(tiny).iloc[0], 10.01)
        np.testing.assert_allclose(lob.spread(tiny).iloc[3], 0.02)
        np.testing.assert_allclose(
            lob.spread(tiny, relative=True).iloc[0], 0.02 / 10.01
        )

    def test_weighted_mid_leans_to_thin_side(self, tiny):
        wm = lob.weighted_mid(tiny)
        # Row 2: bid 30 vs ask 90 -> closer to the bid.
        assert wm.iloc[2] < lob.mid_price(tiny).iloc[2]
        np.testing.assert_allclose(wm.iloc[0], 10.01)
        np.testing.assert_allclose(wm.iloc[2], (10.01 * 90 + 10.02 * 30) / 120)

    def test_weighted_mid_empty_queues(self, tiny):
        t = tiny.copy()
        t.loc[0, ["bid_size", "ask_size"]] = 0
        assert lob.weighted_mid(t).iloc[0] == pytest.approx(10.01)

    def test_depth(self, book):
        d = lob.depth(book, levels=2)
        np.testing.assert_allclose(d["bid_depth"], book["bid_size_1"] + 80.0)
        assert lob.depth(book, levels=1, side="ask").name == "ask_depth"
        with pytest.raises(ValueError):
            lob.depth(book, side="mid")


class TestImbalance:
    def test_queue_imbalance(self, tiny):
        qi = lob.queue_imbalance(tiny)
        np.testing.assert_allclose(qi.iloc[1], 30 / 210)
        assert qi.between(-1, 1).all()

    def test_queue_imbalance_empty_is_nan(self, tiny):
        t = tiny.copy()
        t.loc[0, ["bid_size", "ask_size"]] = 0
        assert np.isnan(lob.queue_imbalance(t).iloc[0])

    def test_depth_imbalance_matches_l1(self, book):
        np.testing.assert_allclose(
            lob.depth_imbalance(book, levels=1), lob.queue_imbalance(book)
        )

    def test_depth_imbalance_weights(self, book):
        row = book.iloc[[0]]
        w = [1.0, 0.5, 0.25]
        bid = sum(w[i] * row[f"bid_size_{i + 1}"].iloc[0] for i in range(3))
        ask = sum(w[i] * row[f"ask_size_{i + 1}"].iloc[0] for i in range(3))
        got = lob.depth_imbalance(row, weights=w).iloc[0]
        assert got == pytest.approx((bid - ask) / (bid + ask))
        decayed = lob.depth_imbalance(row, decay=np.log(2)).iloc[0]
        assert decayed == pytest.approx(got)
        with pytest.raises(ValueError):
            lob.depth_imbalance(row, weights=[1.0, 1.0])

    def test_deeper_levels_dilute(self, book):
        l1 = lob.depth_imbalance(book, levels=1).abs().mean()
        l3 = lob.depth_imbalance(book, levels=3).abs().mean()
        assert l3 < l1


class TestOFI:
    def test_ofi_by_hand(self, tiny):
        ofi = lob.order_flow_imbalance(tiny)
        assert np.isnan(ofi.iloc[0])
        # 1: prices unchanged -> dQb - dQa = 20 - (-10) = 30
        assert ofi.iloc[1] == pytest.approx(30)
        # 2: bid up -> +Qb_new = 30; ask unchanged -> -(90 - 90) = 0
        assert ofi.iloc[2] == pytest.approx(30)
        # 3: bid same -> 30 - 30 = 0; ask up -> +Qa_old = 90
        assert ofi.iloc[3] == pytest.approx(90)
        # 4: bid down -> -Qb_old = -30; ask down -> -Qa_new = -60
        assert ofi.iloc[4] == pytest.approx(-90)

    def test_multilevel_shape(self, book):
        ofi = lob.order_flow_imbalance(book, levels=None)
        assert list(ofi.columns) == ["ofi_1", "ofi_2", "ofi_3"]
        np.testing.assert_allclose(
            ofi["ofi_1"].iloc[1:], lob.order_flow_imbalance(book).iloc[1:]
        )

    def test_integrated_ofi(self, book):
        res = lob.integrated_ofi(book, freq="10s")
        assert res.weights.sum() == pytest.approx(1.0)
        assert 0 < res.explained_variance <= 1
        assert len(res.ofi) > 100
        # Same windows by count when the index is not time-based.
        res2 = lob.integrated_ofi(book.reset_index(drop=True), freq=100)
        assert len(res2.ofi) == len(res.ofi)


class TestPriceImpact:
    def test_positive_impact(self, book):
        res = lob.price_impact(book, freq="10s")
        assert res.beta > 0
        assert res.beta_tstat > 5
        assert 0 < res.r2 <= 1
        assert res.nobs == len(res.data)

    def test_matches_ols(self, book):
        res = lob.price_impact(book, freq=500)
        x, y = res.data["ofi"], res.data["dmid"]
        beta, alpha = np.polyfit(x, y, 1)
        assert res.beta == pytest.approx(beta)
        assert res.alpha == pytest.approx(alpha, abs=1e-12)
        assert res.r2 == pytest.approx(np.corrcoef(x, y)[0, 1] ** 2)
        # 19,999 diffs in windows of 500 -> 40 windows.
        assert res.nobs == 40

    def test_normalized_and_errors(self, book):
        res = lob.price_impact(book, freq="10s", levels=2, normalize_depth=True)
        assert res.beta > 0
        with pytest.raises(ValueError, match="DatetimeIndex"):
            lob.price_impact(book.reset_index(drop=True), freq="10s")
        with pytest.raises(ValueError):
            lob.price_impact(book.iloc[:5], freq=10)


class TestImbalanceSignal:
    def test_monotone_signal(self, book):
        sig = lob.imbalance_signal(book, bins=5)
        assert sig["count"].sum() > 0
        filled = sig[sig["count"] > 50]
        assert filled["p_up"].is_monotonic_increasing
        assert filled["mean_move"].iloc[0] < 0 < filled["mean_move"].iloc[-1]
        np.testing.assert_allclose(
            (filled[["p_up", "p_down", "p_flat"]].sum(axis=1)), 1.0
        )
        # Looking at the next *different* mid, there is no flat outcome.
        assert (filled["p_flat"] == 0).all()
        # Symmetric book: P(up | I) ~ P(down | -I).
        np.testing.assert_allclose(
            sig["p_up"].to_numpy(), sig["p_down"].to_numpy()[::-1], atol=0.05
        )

    def test_fixed_horizon(self, book):
        sig = lob.imbalance_signal(book, horizon=5, bins=4, skip_unchanged=False)
        assert (sig["p_flat"] > 0).any()
        with pytest.raises(ValueError):
            lob.imbalance_signal(book, horizon=0)

    def test_custom_signal(self, book):
        sig = lob.imbalance_signal(
            book, imbalance=lob.depth_imbalance(book, decay=0.5), bins=4
        )
        assert len(sig) == 4


@pytest.fixture(scope="module")
def model(book):
    return lob.fit_microprice(book, n_imbalance=6, n_spread=2)


class TestMicroprice:
    def test_tick_inferred(self, model):
        assert model.tick_size == pytest.approx(TICK)
        assert model.nobs > 0

    def test_adjustment_shape_and_symmetry(self, model):
        g = model.adjustment
        assert g.shape == (6, 2)
        # Symmetrised fit: g*(I) = -g*(1 - I).
        np.testing.assert_allclose(g.to_numpy(), -g.to_numpy()[::-1], atol=1e-12)
        col = g[1].to_numpy()
        assert np.all(np.diff(col) > 0)
        # Bounded by half the spread in the 1-tick state.
        assert np.abs(col).max() < TICK / 2 + 1e-12

    def test_predict(self, model, book):
        mp = model.predict(book)
        mid = lob.mid_price(book)
        assert mp.notna().all()
        imb = lob.queue_imbalance(book)
        assert (mp[imb > 0.5] > mid[imb > 0.5]).all()
        assert (mp[imb < -0.5] < mid[imb < -0.5]).all()

    def test_microprice_beats_mid(self, model, book):
        """The micro-price should forecast the next different mid better."""
        mid = lob.mid_price(book)
        changes = mid.diff().fillna(1) != 0
        # Next different mid from each snapshot.
        run = changes.cumsum()
        run_next = mid[changes].shift(-1).to_numpy()
        nxt = pd.Series(run_next[run.to_numpy() - 1], index=mid.index)
        ok = nxt.notna()
        mp = model.predict(book)
        err_mid = ((nxt - mid)[ok] ** 2).mean()
        err_mp = ((nxt - mp)[ok] ** 2).mean()
        assert err_mp < err_mid

    def test_errors(self, book):
        with pytest.raises(ValueError):
            lob.fit_microprice(book, n_imbalance=1)
        with pytest.raises(ValueError, match="no usable"):
            lob.fit_microprice(book, tick_size=1.0)


class TestPlots:
    @pytest.fixture(autouse=True)
    def _agg(self):
        import matplotlib

        matplotlib.use("Agg")

    def test_all_charts(self, book, model, tmp_path):
        from matplotlib.figure import Figure

        from quantstats import plots

        small = book.iloc[:300]
        figs = [
            plots.lob_imbalance_signal(book, bins=6, show=False),
            plots.lob_imbalance_signal(lob.imbalance_signal(book), show=False),
            plots.lob_microprice(model, show=False),
            plots.lob_microprice(model, in_ticks=False, grayscale=True, show=False),
            plots.lob_price_impact(lob.price_impact(book, freq="10s"), show=False),
            plots.lob_depth(book, show=False),
            plots.lob_depth(book, at=book.index[10], show=False),
            plots.lob_timeseries(small, model=model, show=False),
        ]
        assert all(isinstance(f, Figure) for f in figs)
        out = tmp_path / "depth.png"
        plots.lob_depth(book, savefig=str(out), show=False)
        assert out.stat().st_size > 0


class TestFromDepth:
    def test_binance_like(self):
        snaps = [
            {
                "E": 1700000000000 + 100 * k,
                "bids": [["100.01", "5"], ["100.00", "7"], ["99.99", "1"]],
                "asks": [["100.02", "3"], ["100.03", "4"]],
            }
            for k in range(3)
        ]
        book = lob.from_depth(snaps, time_key="E")
        assert lob.levels(book) == 2
        assert book["bid_price_1"].iloc[0] == 100.01
        assert book["ask_size_2"].iloc[0] == 4.0
        assert isinstance(book.index, pd.DatetimeIndex)
        assert book.index[1] - book.index[0] == pd.Timedelta("100ms")

    def test_extra_fields_and_errors(self):
        snaps = [{"bids": [[1.0, 2.0, 3]], "asks": [[1.1, 4.0, 1]]}] * 2
        book = lob.from_depth(snaps, timestamps=[0, 1])
        assert list(book.index) == [0, 1]
        assert lob.queue_imbalance(book).iloc[0] == pytest.approx(-1 / 3)
        with pytest.raises(ValueError):
            lob.from_depth(snaps, levels=2)
        with pytest.raises(ValueError):
            lob.from_depth([{"bids": [], "asks": [[1, 1]]}])


def test_microprice_first_order(model, book):
    g1 = model.predict(book, first_order=True)
    full = model.predict(book)
    mid = lob.mid_price(book)
    assert g1.notna().all()
    imb = lob.queue_imbalance(book)
    # Both forecasts lean the same way as the imbalance.
    assert (g1[imb > 0.5] > mid[imb > 0.5]).all()
    assert not np.allclose(g1, full)
