#!/usr/bin/env python
#
# QuantStats: Portfolio analytics for quants
# https://github.com/ranaroussi/quantstats
#
# Copyright 2019-2025 Ran Aroussi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Limit Order Book (LOB) & Imbalance Modeling

Features and models built from limit order book snapshots:

* Book features: mid price, spread, depth, weighted mid.
* Imbalance: top-of-book queue imbalance and multi-level depth imbalance.
* Order Flow Imbalance (OFI), per level, after Cont, Kukanov & Stoikov (2014),
  and integrated OFI, after Cont, Cucuringu & Zhang (2023).
* Price impact: OLS of mid-price changes on OFI over aggregation windows.
* Imbalance signal: how often the mid moves up or down next, by imbalance
  bucket (Gould & Bonart, 2016).
* Micro-price: Stoikov (2018) Markov-chain estimate of the expected future
  mid given the imbalance and spread.

Book format
-----------
A ``book`` is a DataFrame with one row per snapshot, in time order, and
columns ``bid_price_{n}``, ``bid_size_{n}``, ``ask_price_{n}`` and
``ask_size_{n}`` for levels ``n = 1..L`` (level 1 is the best quote). A
top-of-book-only frame may use ``bid``, ``ask``, ``bid_size`` and
``ask_size`` instead. Use :func:`to_book` to rename other layouts.
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass, field
from typing import Any

import numpy as _np
import pandas as _pd
from scipy import stats as _sstats

__all__ = [
    "to_book",
    "from_depth",
    "levels",
    "mid_price",
    "spread",
    "depth",
    "weighted_mid",
    "queue_imbalance",
    "depth_imbalance",
    "order_flow_imbalance",
    "integrated_ofi",
    "price_impact",
    "imbalance_signal",
    "fit_microprice",
    "MicropriceModel",
]

_SIDES = ("bid", "ask")
_FIELDS = ("price", "size")
_COL_RE = _re.compile(r"^(bid|ask)_(price|size)_(\d+)$")
_L1_ALIASES = {
    "bid": "bid_price_1",
    "ask": "ask_price_1",
    "bid_size": "bid_size_1",
    "ask_size": "ask_size_1",
}


# ======== BOOK ACCESS ========


def to_book(
    data: _pd.DataFrame,
    columns: dict[str, str] | None = None,
) -> _pd.DataFrame:
    """
    Normalise a snapshot DataFrame to the ``{side}_{field}_{level}`` layout.

    Args:
        data: Snapshots, one row each, in time order.
        columns: Optional mapping from the frame's column names to canonical
            names, e.g. ``{"bp1": "bid_price_1", "bq1": "bid_size_1"}``.
            Top-of-book aliases (``bid``, ``ask``, ``bid_size``,
            ``ask_size``) are mapped automatically.

    Returns:
        DataFrame with canonical columns only, as floats, levels ascending.

    Raises:
        ValueError: If level 1 is missing a column, or a level is incomplete.
    """
    df = data.rename(columns=columns or {})
    df = df.rename(columns={k: v for k, v in _L1_ALIASES.items() if k in df})
    found: dict[int, set[str]] = {}
    for col in df.columns:
        m = _COL_RE.match(str(col))
        if m:
            found.setdefault(int(m.group(3)), set()).add(f"{m.group(1)}_{m.group(2)}")
    if 1 not in found:
        raise ValueError(
            "book needs level-1 columns bid_price_1, bid_size_1, "
            "ask_price_1 and ask_size_1 (or bid, bid_size, ask, ask_size)"
        )
    need = {f"{s}_{f}" for s in _SIDES for f in _FIELDS}
    n_levels = 0
    while n_levels + 1 in found:
        missing = need - found[n_levels + 1]
        if missing:
            raise ValueError(
                f"level {n_levels + 1} is missing {sorted(missing)} columns"
            )
        n_levels += 1
    cols = [
        f"{s}_{f}_{n}" for n in range(1, n_levels + 1) for s in _SIDES for f in _FIELDS
    ]
    return df[cols].astype(float)


def from_depth(
    snapshots: list[dict],
    levels: int | None = None,
    timestamps: list | _pd.Index | None = None,
    time_key: str | None = None,
) -> _pd.DataFrame:
    """
    Build a book from exchange depth snapshots.

    Each snapshot is a mapping with ``"bids"`` and ``"asks"``, each a list of
    ``[price, size, ...]`` entries ordered best first, the shape returned by
    most exchange REST and websocket depth feeds (Binance, Coinbase level 2,
    Kraken, OKX, Bybit). Prices and sizes may be strings. Extra fields in an
    entry (order counts, timestamps) are ignored.

    Args:
        snapshots: Depth snapshots in time order.
        levels: Levels to keep; the fewest levels any snapshot has if None.
        timestamps: Index for the result, one per snapshot.
        time_key: Snapshot field holding the time, used when ``timestamps``
            is None. Numbers are read as epoch milliseconds.

    Returns:
        DataFrame in the :mod:`quantstats.lob` book layout.

    Raises:
        ValueError: If a snapshot has no bids or asks, or fewer than
            ``levels`` levels.
    """
    if not snapshots:
        raise ValueError("no snapshots")
    depths = [min(len(s["bids"]), len(s["asks"])) for s in snapshots]
    if min(depths) == 0:
        raise ValueError("every snapshot needs at least one bid and one ask")
    n = min(depths) if levels is None else int(levels)
    if n > min(depths):
        raise ValueError(f"a snapshot has only {min(depths)} levels, asked for {n}")
    cols = [
        f"{side}_{fld}_{i}"
        for i in range(1, n + 1)
        for side in _SIDES
        for fld in _FIELDS
    ]
    arr = _np.empty((len(snapshots), 4 * n))
    for r, snap in enumerate(snapshots):
        bids, asks = snap["bids"], snap["asks"]
        for i in range(n):
            arr[r, 4 * i : 4 * i + 2] = bids[i][:2]
            arr[r, 4 * i + 2 : 4 * i + 4] = asks[i][:2]
    index = None
    if timestamps is not None:
        index = _pd.Index(timestamps)
    elif time_key is not None:
        raw = [s[time_key] for s in snapshots]
        index = _pd.DatetimeIndex(
            _pd.to_datetime(raw, unit="ms")
            if isinstance(raw[0], (int, float))
            else _pd.to_datetime(raw)
        )
    return _pd.DataFrame(arr, columns=cols, index=index)


def levels(book: _pd.DataFrame) -> int:
    """Return the number of complete price levels in ``book``."""
    return len(to_book(book).columns) // 4


def _get(book: _pd.DataFrame, side: str, fld: str, level: int = 1) -> _pd.Series:
    col = f"{side}_{fld}_{level}"
    if col in book:
        return book[col].astype(float)
    alias = {v: k for k, v in _L1_ALIASES.items()}.get(col)
    if alias is not None and alias in book:
        return book[alias].astype(float)
    raise KeyError(f"book has no column {col!r}")


def _resolve_levels(book: _pd.DataFrame, n: int | None) -> int:
    available = levels(book)
    if n is None:
        return available
    if n < 1 or n > available:
        raise ValueError(f"levels must be between 1 and {available}, got {n}")
    return n


# ======== BOOK FEATURES ========


def mid_price(book: _pd.DataFrame) -> _pd.Series:
    """Mid price, ``(best bid + best ask) / 2``."""
    out = (_get(book, "bid", "price") + _get(book, "ask", "price")) / 2
    return out.rename("mid_price")


def spread(book: _pd.DataFrame, relative: bool = False) -> _pd.Series:
    """
    Quoted spread, ``best ask - best bid``.

    Args:
        book: Snapshots.
        relative: Divide by the mid price (a fraction, e.g. 0.0001 = 1bp).
    """
    out = _get(book, "ask", "price") - _get(book, "bid", "price")
    if relative:
        out = out / mid_price(book)
    return out.rename("spread")


def depth(
    book: _pd.DataFrame, levels: int | None = None, side: str = "both"
) -> _pd.Series | _pd.DataFrame:
    """
    Resting size summed over the first ``levels`` levels.

    Args:
        book: Snapshots.
        levels: Levels to sum; all available levels if None.
        side: ``"bid"``, ``"ask"`` or ``"both"`` (DataFrame with both).
    """
    n = _resolve_levels(book, levels)
    sums = {s: sum(_get(book, s, "size", i) for i in range(1, n + 1)) for s in _SIDES}
    if side == "both":
        return _pd.DataFrame({"bid_depth": sums["bid"], "ask_depth": sums["ask"]})
    if side not in _SIDES:
        raise ValueError("side must be 'bid', 'ask' or 'both'")
    return _pd.Series(sums[side], name=f"{side}_depth")


def weighted_mid(book: _pd.DataFrame) -> _pd.Series:
    """
    Size-weighted mid, ``(P_bid * Q_ask + P_ask * Q_bid) / (Q_bid + Q_ask)``.

    Leans toward the side with less resting size, i.e. the side more likely
    to be consumed next. Falls back to the mid when both queues are empty.
    """
    pb, pa = _get(book, "bid", "price"), _get(book, "ask", "price")
    qb, qa = _get(book, "bid", "size"), _get(book, "ask", "size")
    tot = qb + qa
    out = (pb * qa + pa * qb) / tot.where(tot > 0)
    return out.fillna((pb + pa) / 2).rename("weighted_mid")


# ======== IMBALANCE ========


def queue_imbalance(book: _pd.DataFrame) -> _pd.Series:
    """
    Top-of-book queue imbalance, ``(Q_bid - Q_ask) / (Q_bid + Q_ask)``.

    Lies in ``[-1, 1]``; positive means more resting size on the bid. NaN
    when both best queues are empty.
    """
    qb, qa = _get(book, "bid", "size"), _get(book, "ask", "size")
    tot = qb + qa
    return ((qb - qa) / tot.where(tot > 0)).rename("queue_imbalance")


def depth_imbalance(
    book: _pd.DataFrame,
    levels: int | None = None,
    weights: _np.ndarray | list[float] | None = None,
    decay: float | None = None,
) -> _pd.Series:
    """
    Multi-level depth imbalance, ``(sum w_i Qb_i - sum w_i Qa_i) / (sum w_i
    Qb_i + sum w_i Qa_i)``.

    Args:
        book: Snapshots.
        levels: Levels to include; all available levels if None.
        weights: One weight per level. Overrides ``decay``.
        decay: Exponential weights ``exp(-decay * (i - 1))``, so deeper
            levels count less. Equal weights when neither is given.

    Returns:
        Series in ``[-1, 1]``.
    """
    n = _resolve_levels(book, levels)
    if weights is not None:
        w = _np.asarray(weights, dtype=float)
        if w.shape != (n,):
            raise ValueError(f"weights must have length {n}")
    elif decay is not None:
        w = _np.exp(-float(decay) * _np.arange(n))
    else:
        w = _np.ones(n)
    bid = _pd.concat(
        [w[i] * _get(book, "bid", "size", i + 1) for i in range(n)], axis=1
    ).sum(axis=1)
    ask = _pd.concat(
        [w[i] * _get(book, "ask", "size", i + 1) for i in range(n)], axis=1
    ).sum(axis=1)
    tot = bid + ask
    out = (bid - ask) / tot.where(tot > 0)
    return _pd.Series(out, name="depth_imbalance")


# ======== ORDER FLOW IMBALANCE ========


def _level_ofi(book: _pd.DataFrame, level: int) -> _pd.Series:
    pb, qb = _get(book, "bid", "price", level), _get(book, "bid", "size", level)
    pa, qa = _get(book, "ask", "price", level), _get(book, "ask", "size", level)
    pb0, qb0, pa0, qa0 = pb.shift(), qb.shift(), pa.shift(), qa.shift()
    bid_flow = qb.where(pb >= pb0, 0.0) - qb0.where(pb <= pb0, 0.0)
    ask_flow = qa.where(pa <= pa0, 0.0) - qa0.where(pa >= pa0, 0.0)
    e = bid_flow - ask_flow
    e.iloc[0] = _np.nan
    return e


def order_flow_imbalance(
    book: _pd.DataFrame, levels: int | None = 1
) -> _pd.Series | _pd.DataFrame:
    """
    Order Flow Imbalance between consecutive snapshots (Cont, Kukanov &
    Stoikov, 2014).

    At level ``m`` the contribution of snapshot ``n`` is::

        e_n = 1{Pb_n >= Pb_{n-1}} Qb_n - 1{Pb_n <= Pb_{n-1}} Qb_{n-1}
            - 1{Pa_n <= Pa_{n-1}} Qa_n + 1{Pa_n >= Pa_{n-1}} Qa_{n-1}

    Positive values are net buying pressure: bids added or improved, asks
    cancelled, hit or pulled back. Sum over a window to get that window's OFI.

    Args:
        book: Snapshots in time order.
        levels: 1 (default) returns a Series for the best level. A larger
            number, or None for every level, returns a DataFrame with columns
            ``ofi_1..ofi_L``.

    Returns:
        Series or DataFrame; the first row is NaN.
    """
    if levels == 1:
        return _level_ofi(book, 1).rename("ofi")
    n = _resolve_levels(book, levels)
    return _pd.DataFrame({f"ofi_{i}": _level_ofi(book, i) for i in range(1, n + 1)})


def _windows(data: _pd.DataFrame | _pd.Series, freq: str | int) -> Any:
    if isinstance(freq, int):
        if freq < 1:
            raise ValueError("freq must be a positive number of snapshots")
        return data.groupby(_np.arange(len(data)) // freq)
    if not isinstance(data.index, _pd.DatetimeIndex):
        raise ValueError("a time-based freq needs a DatetimeIndex; pass an int")
    return data.groupby(_pd.Grouper(freq=freq))


@dataclass
class IntegratedOFI:
    """Result of :func:`integrated_ofi`."""

    ofi: _pd.Series
    """Integrated OFI, one value per window."""
    weights: _pd.Series
    """First-principal-component weights per level, summing to 1."""
    explained_variance: float
    """Share of the scaled multi-level OFI variance the component explains."""


def integrated_ofi(
    book: _pd.DataFrame, freq: str | int, levels: int | None = None
) -> IntegratedOFI:
    """
    Integrated multi-level OFI (Cont, Cucuringu & Zhang, 2023).

    Per-level OFI is summed over each window, scaled by the average depth at
    the top ``levels`` levels in that window, and projected on its first
    principal component, whose weights are normalised to sum to 1.

    Args:
        book: Snapshots.
        freq: Window: a pandas offset alias (``"1min"``) on a DatetimeIndex,
            or an int number of snapshots.
        levels: Levels to use; all available if None.
    """
    n = _resolve_levels(book, levels)
    ofi = order_flow_imbalance(book, levels=n)
    if isinstance(ofi, _pd.Series):
        ofi = ofi.to_frame("ofi_1")
    d = depth(book, levels=n)
    assert isinstance(d, _pd.DataFrame)
    avg_depth = (d["bid_depth"] + d["ask_depth"]) / (2 * n)
    num = _windows(ofi, freq).sum(min_count=1)
    den = _windows(avg_depth, freq).mean()
    scaled = num.div(den.where(den > 0), axis=0).dropna()
    if len(scaled) < 2:
        raise ValueError("need at least two windows with data")
    cov = _np.cov(scaled.to_numpy(), rowvar=False).reshape(n, n)
    vals, vecs = _np.linalg.eigh(cov)
    v1 = vecs[:, -1]
    if v1.sum() < 0:
        v1 = -v1
    w = v1 / _np.abs(v1).sum()
    total = vals.sum()
    return IntegratedOFI(
        ofi=_pd.Series(scaled.to_numpy() @ w, index=scaled.index, name="ofi_i"),
        weights=_pd.Series(w, index=scaled.columns),
        explained_variance=float(vals[-1] / total) if total > 0 else _np.nan,
    )


# ======== PRICE IMPACT ========


@dataclass
class ImpactResult:
    """Result of :func:`price_impact`."""

    beta: float
    """Mid-price change per unit of OFI (price impact coefficient)."""
    alpha: float
    """Intercept."""
    beta_tstat: float
    alpha_tstat: float
    r2: float
    nobs: int
    data: _pd.DataFrame = field(repr=False)
    """Per-window ``ofi`` and ``dmid`` the regression used."""


def price_impact(
    book: _pd.DataFrame,
    freq: str | int,
    levels: int | None = 1,
    normalize_depth: bool = False,
) -> ImpactResult:
    """
    Linear price impact of order flow: ``dMid_k = alpha + beta * OFI_k``.

    Cont, Kukanov & Stoikov (2014) find this relation is close to linear,
    with ``beta`` roughly inversely proportional to depth.

    Args:
        book: Snapshots.
        freq: Window: a pandas offset alias on a DatetimeIndex, or an int
            number of snapshots.
        levels: Levels whose OFI is summed; 1 for best level only.
        normalize_depth: Divide each window's OFI by its average depth over
            those levels, making ``beta`` comparable across stocks and days.
    """
    n = _resolve_levels(book, levels)
    ofi = order_flow_imbalance(book, levels=n)
    flow = ofi if isinstance(ofi, _pd.Series) else ofi.sum(axis=1, min_count=1)
    mid = mid_price(book)
    dmid = mid.diff()
    frame = _pd.DataFrame({"ofi": flow, "dmid": dmid})
    if normalize_depth:
        d = depth(book, levels=n)
        assert isinstance(d, _pd.DataFrame)
        frame["depth"] = (d["bid_depth"] + d["ask_depth"]) / (2 * n)
    frame = frame.iloc[1:]
    grouped = _windows(frame, freq)
    agg = grouped[["ofi", "dmid"]].sum(min_count=1)
    if normalize_depth:
        dep = grouped["depth"].mean()
        agg["ofi"] = agg["ofi"] / dep.where(dep > 0)
    agg = agg.dropna()
    if len(agg) < 3:
        raise ValueError("need at least three windows to fit price impact")
    x, y = agg["ofi"].to_numpy(), agg["dmid"].to_numpy()
    if _np.ptp(x) == 0:
        raise ValueError("OFI is constant across windows")
    fit = _sstats.linregress(x, y)
    nobs = len(agg)
    resid = y - (fit.intercept + fit.slope * x)  # type: ignore[attr-defined]
    s2 = (resid @ resid) / max(nobs - 2, 1)
    sxx = ((x - x.mean()) ** 2).sum()
    se_a = _np.sqrt(s2 * (1 / nobs + x.mean() ** 2 / sxx))
    se_b = fit.stderr  # type: ignore[attr-defined]
    return ImpactResult(
        beta=float(fit.slope),  # type: ignore[attr-defined]
        alpha=float(fit.intercept),  # type: ignore[attr-defined]
        beta_tstat=float(fit.slope / se_b) if se_b > 0 else _np.inf,  # type: ignore[attr-defined]
        alpha_tstat=float(fit.intercept / se_a) if se_a > 0 else _np.inf,  # type: ignore[attr-defined]
        r2=float(fit.rvalue**2),  # type: ignore[attr-defined]
        nobs=nobs,
        data=agg,
    )


# ======== IMBALANCE SIGNAL ========


def imbalance_signal(
    book: _pd.DataFrame,
    horizon: int = 1,
    bins: int = 10,
    imbalance: _pd.Series | None = None,
    skip_unchanged: bool = True,
) -> _pd.DataFrame:
    """
    Next mid-price move conditional on imbalance (Gould & Bonart, 2016).

    Buckets the imbalance into equal-width bins on ``[-1, 1]`` and tabulates
    what the mid did ``horizon`` steps later.

    Args:
        book: Snapshots.
        horizon: Steps ahead. With ``skip_unchanged`` a step is a mid change,
            so ``horizon=1`` asks about the *next* mid move whenever it comes.
        bins: Number of imbalance buckets.
        imbalance: Signal to bucket; :func:`queue_imbalance` if None. Must
            lie in ``[-1, 1]`` and share ``book``'s index.
        skip_unchanged: Look ahead to the next *different* mid, not the
            snapshot ``horizon`` rows later.

    Returns:
        DataFrame indexed by bucket interval with ``count``, ``p_up``,
        ``p_down``, ``p_flat``, ``mean_move`` (in price units) and
        ``mean_imbalance``.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    imb = queue_imbalance(book) if imbalance is None else imbalance.astype(float)
    mid = mid_price(book)
    if skip_unchanged:
        changes = mid.diff().fillna(1.0) != 0
        # Mid at the start of each run, and the mid `horizon` runs later.
        run = _pd.Series(changes.cumsum())
        run_mid = _np.asarray(mid[changes], dtype=float)
        future = _np.full(len(run_mid), _np.nan)
        if len(run_mid) > horizon:
            future[:-horizon] = run_mid[horizon:]
        future_mid = _pd.Series(future[run.to_numpy() - 1], index=mid.index)
    else:
        future_mid = mid.shift(-horizon)
    move = future_mid - mid
    frame = _pd.DataFrame({"imb": imb, "move": move}).dropna()
    edges = _np.linspace(-1, 1, bins + 1)
    frame["bucket"] = _pd.cut(frame["imb"], edges, include_lowest=True)
    g = frame.groupby("bucket", observed=False)
    out = _pd.DataFrame(
        {
            "count": g.size(),
            "p_up": g["move"].apply(lambda m: (m > 0).mean()),
            "p_down": g["move"].apply(lambda m: (m < 0).mean()),
            "p_flat": g["move"].apply(lambda m: (m == 0).mean()),
            "mean_move": g["move"].mean(),
            "mean_imbalance": g["imb"].mean(),
        }
    )
    out.index.name = "imbalance"
    return out


# ======== MICRO-PRICE ========


@dataclass
class MicropriceModel:
    """
    Fitted Stoikov (2018) micro-price model.

    ``adjustment`` holds ``g*(I, S)``: the expected long-run mid change,
    in price units, given the imbalance bucket ``I`` and spread ``S`` in
    ticks. The micro-price is ``mid + g*(I, S)``.
    """

    adjustment: _pd.DataFrame
    """``g*``: rows are imbalance buckets, columns spread in ticks."""
    tick_size: float
    bin_edges: _np.ndarray = field(repr=False)
    """Bucket edges on ``Q_bid / (Q_bid + Q_ask)`` in ``[0, 1]``."""
    first_order: _pd.DataFrame = field(repr=False)
    """``g1``: expected mid change at the next move only."""
    nobs: int = 0

    def predict(self, book: _pd.DataFrame) -> _pd.Series:
        """
        Micro-price for each snapshot.

        Spreads outside the fitted range, and states never seen in fitting,
        use the mid unchanged.
        """
        mid = mid_price(book)
        imb = _imbalance01(book)
        bucket = _bucket(imb, self.bin_edges)
        ticks = _np.round(spread(book).to_numpy() / self.tick_size)
        adj = _np.zeros(len(mid))
        for s in self.adjustment.columns:
            col = _np.nan_to_num(self.adjustment[s].to_numpy(), nan=0.0)
            mask = (ticks == s) & (bucket >= 0)
            adj[mask] = col[bucket[mask]]
        return (mid + adj).rename("microprice")


def _imbalance01(book: _pd.DataFrame) -> _pd.Series:
    qb, qa = _get(book, "bid", "size"), _get(book, "ask", "size")
    tot = qb + qa
    return qb / tot.where(tot > 0)


def _bucket(imb: _pd.Series, edges: _np.ndarray) -> _np.ndarray:
    vals = imb.to_numpy()
    out = _np.clip(_np.searchsorted(edges, vals, side="right") - 1, 0, len(edges) - 2)
    return _np.where(_np.isnan(vals), -1, out)


def _infer_tick(spr: _pd.Series) -> float:
    pos = _np.asarray(spr[spr > 0], dtype=float)
    if len(pos) == 0:
        raise ValueError("cannot infer tick_size: no positive spreads")
    tick = float(pos.min())
    # Snap away float noise, e.g. 0.010000000000005116 -> 0.01.
    return float(f"{tick:.10g}")


def fit_microprice(
    book: _pd.DataFrame,
    tick_size: float | None = None,
    n_imbalance: int = 10,
    n_spread: int = 2,
    max_iter: int = 6,
    symmetrize: bool = True,
) -> MicropriceModel:
    """
    Fit Stoikov's (2018) micro-price.

    The state is (imbalance bucket, spread in ticks), with imbalance
    ``I = Q_bid / (Q_bid + Q_ask)`` in equal-width buckets on ``[0, 1]``.
    From consecutive snapshots it estimates

    * ``Q``: transitions in which the mid does not move,
    * ``R1``: probability of each mid move ``K`` (half-tick steps up to one
      tick), and ``R2``: the state reached when the mid does move,

    then ``g1 = (I - Q)^-1 R1 K``, ``B = (I - Q)^-1 R2`` and
    ``g* = sum_{k=0}^{max_iter} B^k g1``.

    Snapshots with a spread wider than ``n_spread`` ticks, a one-step mid
    move bigger than one tick, or empty best queues are dropped. The data
    are symmetrised (each observation is mirrored bid-for-ask) so the
    adjustment is antisymmetric in imbalance, as the model assumes.

    Args:
        book: Snapshots, sampled finely enough that most steps move the mid
            by at most one tick.
        tick_size: Price tick; the smallest positive spread if None.
        n_imbalance: Imbalance buckets.
        n_spread: Largest spread, in ticks, kept as a state.
        max_iter: Terms of the ``B^k g1`` series.
        symmetrize: Mirror the data (recommended).

    Returns:
        :class:`MicropriceModel`. States never visited have NaN adjustment.
    """
    if n_imbalance < 2 or n_spread < 1:
        raise ValueError("need n_imbalance >= 2 and n_spread >= 1")
    spr = spread(book)
    tick = _infer_tick(spr) if tick_size is None else float(tick_size)
    edges = _np.linspace(0.0, 1.0, n_imbalance + 1)

    mid = mid_price(book).to_numpy()
    ticks = _np.round(spr.to_numpy() / tick)
    bucket = _bucket(_imbalance01(book), edges)
    valid = (ticks >= 1) & (ticks <= n_spread) & (bucket >= 0)

    # Moves in half ticks; keep steps where both ends are valid states.
    dm = _np.round(_np.diff(mid) / tick * 2)
    keep = valid[:-1] & valid[1:] & (_np.abs(dm) <= 2)
    s0, s1 = ticks[:-1][keep].astype(int), ticks[1:][keep].astype(int)
    i0, i1 = bucket[:-1][keep], bucket[1:][keep]
    dm = dm[keep].astype(int)
    if symmetrize:
        s0, s1 = _np.r_[s0, s0], _np.r_[s1, s1]
        i0 = _np.r_[i0, n_imbalance - 1 - i0]
        i1 = _np.r_[i1, n_imbalance - 1 - i1]
        dm = _np.r_[dm, -dm]
    nobs = int(keep.sum())
    if nobs == 0:
        raise ValueError("no usable transitions; check tick_size and n_spread")

    n_states = n_imbalance * n_spread
    x0 = (s0 - 1) * n_imbalance + i0
    x1 = (s1 - 1) * n_imbalance + i1
    moves = _np.array([-2, -1, 1, 2])  # half ticks
    k_vec = moves * tick / 2

    q = _np.zeros((n_states, n_states))
    r1 = _np.zeros((n_states, len(moves)))
    r2 = _np.zeros((n_states, n_states))
    still = dm == 0
    _np.add.at(q, (x0[still], x1[still]), 1.0)
    _np.add.at(r2, (x0[~still], x1[~still]), 1.0)
    move_idx = _np.searchsorted(moves, dm[~still])
    _np.add.at(r1, (x0[~still], move_idx), 1.0)

    totals = q.sum(axis=1) + r1.sum(axis=1)
    seen = totals > 0
    # Unvisited states absorb with zero move, keeping I - Q invertible.
    safe = _np.where(seen, totals, 1.0)[:, None]
    q, r1, r2 = q / safe, r1 / safe, r2 / safe
    eye = _np.eye(n_states)
    try:
        fundamental = _np.linalg.inv(eye - q)
    except _np.linalg.LinAlgError:
        fundamental = _np.linalg.pinv(eye - q)
    g1 = fundamental @ r1 @ k_vec
    b = fundamental @ r2
    g = g1.copy()
    term = g1.copy()
    for _ in range(max_iter):
        term = b @ term
        g = g + term
    g[~seen] = _np.nan
    g1[~seen] = _np.nan

    labels = _pd.IntervalIndex.from_breaks(edges, closed="left", name="imbalance")
    cols = _pd.Index(range(1, n_spread + 1), name="spread_ticks")

    def _table(v: _np.ndarray) -> _pd.DataFrame:
        return _pd.DataFrame(
            v.reshape(n_spread, n_imbalance).T, index=labels, columns=cols
        )

    return MicropriceModel(
        adjustment=_table(g),
        tick_size=tick,
        bin_edges=edges,
        first_order=_table(g1),
        nobs=nobs,
    )
