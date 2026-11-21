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

"""Charts for :mod:`quantstats.lob` (limit order book & imbalance models)."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as _plt
import numpy as _np
import pandas as _pd

from .. import lob as _lob

_UP = "#4fa487"
_DOWN = "#af4b64"
_LINE = "#348dc1"
_ACCENT = "#fedd78"
_GRAY_UP, _GRAY_DOWN, _GRAY_LINE = "#444444", "#AAAAAA", "#000000"


def _palette(grayscale: bool) -> tuple[str, str, str]:
    if grayscale:
        return _GRAY_UP, _GRAY_DOWN, _GRAY_LINE
    return _UP, _DOWN, _LINE


def _title(ax, title: str, subtitle: str | None, fontname: str) -> None:
    text = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(text, fontname=fontname, fontweight="bold", fontsize=12)


def _finish(fig, savefig: Any, show: bool):
    fig.tight_layout()
    if savefig:
        if isinstance(savefig, dict):
            fig.savefig(**savefig)
        else:
            fig.savefig(savefig)
    if show:
        _plt.show()
    _plt.close(fig)
    return None if show else fig


def _bucket_labels(index: _pd.Index) -> list[str]:
    return [
        f"{max(i.left, -1):+.2f} to {i.right:+.2f}"
        if isinstance(i, _pd.Interval)
        else str(i)
        for i in index
    ]


def lob_imbalance_signal(
    data: _pd.DataFrame,
    horizon: int = 1,
    bins: int = 10,
    title: str = "Next Mid Move by Imbalance",
    figsize: tuple[float, float] = (10, 5),
    grayscale: bool = False,
    fontname: str = "Arial",
    subtitle: bool = True,
    savefig: Any = None,
    show: bool = True,
):
    """
    Bar chart of P(up) and P(down) for the next mid move, by imbalance bucket.

    Args:
        data: An order book (see :mod:`quantstats.lob`) or a table already
            returned by :func:`quantstats.lob.imbalance_signal`.
        horizon, bins: Passed to :func:`quantstats.lob.imbalance_signal`
            when ``data`` is a book.

    Returns:
        The figure when ``show=False``, else None.
    """
    table = (
        data
        if {"p_up", "p_down", "count"}.issubset(data.columns)
        else _lob.imbalance_signal(data, horizon=horizon, bins=bins)
    )
    up, down, line = _palette(grayscale)
    fig, ax = _plt.subplots(figsize=figsize)
    x = _np.arange(len(table))
    w = 0.4
    ax.bar(x - w / 2, table["p_up"], w, color=up, label="P(up)")
    ax.bar(x + w / 2, table["p_down"], w, color=down, label="P(down)")
    ax.axhline(0.5, color="black", linewidth=0.6, linestyle=":")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(_bucket_labels(table.index), rotation=45, ha="right")
    ax.set_xlabel("Imbalance bucket", fontname=fontname)
    ax.set_ylabel("Probability", fontname=fontname, fontweight="bold")
    ax2 = ax.twinx()
    ax2.plot(x, table["mean_move"], color=line, marker="o", label="Mean move")
    ax2.axhline(0, color=line, linewidth=0.5)
    ax2.set_ylabel("Mean move (price)", fontname=fontname, color=line)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    sub = f"{int(table['count'].sum()):,} observations" if subtitle else None
    _title(ax, title, sub, fontname)
    return _finish(fig, savefig, show)


def lob_microprice(
    model: _lob.MicropriceModel,
    in_ticks: bool = True,
    title: str = "Micro-price Adjustment",
    figsize: tuple[float, float] = (10, 5),
    grayscale: bool = False,
    fontname: str = "Arial",
    subtitle: bool = True,
    savefig: Any = None,
    show: bool = True,
):
    """
    Micro-price adjustment ``g*(I, S)`` against imbalance, one line per spread.

    Args:
        model: A fitted :class:`quantstats.lob.MicropriceModel`.
        in_ticks: Show the adjustment in ticks rather than price units.
    """
    adj = model.adjustment / (model.tick_size if in_ticks else 1.0)
    edges = model.bin_edges
    mids = (edges[:-1] + edges[1:]) / 2
    fig, ax = _plt.subplots(figsize=figsize)
    styles = ["-", "--", ":", "-."]
    colors = (
        [_LINE, _DOWN, _UP, _ACCENT] if not grayscale else ["#000000", "#555555"] * 2
    )
    for k, s in enumerate(adj.columns):
        ax.plot(
            mids,
            adj[s],
            marker="o",
            linestyle=styles[k % len(styles)],
            color=colors[k % len(colors)],
            label=f"spread = {s} tick{'s' if s != 1 else ''}",
        )
        if in_ticks:
            ax.axhline(s / 2, color=colors[k % len(colors)], alpha=0.3, linewidth=0.8)
            ax.axhline(-s / 2, color=colors[k % len(colors)], alpha=0.3, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axvline(0.5, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Imbalance  Q_bid / (Q_bid + Q_ask)", fontname=fontname)
    unit = "ticks" if in_ticks else "price"
    ax.set_ylabel(f"Micro-price − mid ({unit})", fontname=fontname, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    sub = (
        f"tick = {model.tick_size:g} | {model.nobs:,} transitions"
        + (" | faint lines: ± half spread" if in_ticks else "")
        if subtitle
        else None
    )
    _title(ax, title, sub, fontname)
    return _finish(fig, savefig, show)


def lob_price_impact(
    result: _lob.ImpactResult,
    title: str = "Order Flow Imbalance vs Mid-price Change",
    figsize: tuple[float, float] = (8, 6),
    grayscale: bool = False,
    fontname: str = "Arial",
    subtitle: bool = True,
    savefig: Any = None,
    show: bool = True,
):
    """Scatter of window OFI against mid change, with the fitted line."""
    _, down, line = _palette(grayscale)
    d = result.data
    fig, ax = _plt.subplots(figsize=figsize)
    ax.scatter(d["ofi"], d["dmid"], s=12, alpha=0.5, color=line, edgecolors="none")
    xs = _np.linspace(d["ofi"].min(), d["ofi"].max(), 50)
    ax.plot(xs, result.alpha + result.beta * xs, color=down, linewidth=2, label="OLS")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("OFI per window", fontname=fontname)
    ax.set_ylabel("Mid-price change", fontname=fontname, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    sub = (
        f"β = {result.beta:.3g} (t = {result.beta_tstat:.1f}) | "
        f"R² = {result.r2:.2f} | n = {result.nobs:,}"
        if subtitle
        else None
    )
    _title(ax, title, sub, fontname)
    return _finish(fig, savefig, show)


def lob_depth(
    book: _pd.DataFrame,
    at: int | Any = -1,
    title: str = "Order Book Depth",
    figsize: tuple[float, float] = (8, 6),
    grayscale: bool = False,
    fontname: str = "Arial",
    subtitle: bool = True,
    savefig: Any = None,
    show: bool = True,
):
    """
    Price ladder of one snapshot: bid and ask size at each level.

    Args:
        book: Order book snapshots.
        at: Row position (int) or index label of the snapshot to draw.
    """
    up, down, _ = _palette(grayscale)
    b = _lob.to_book(book)
    row = b.iloc[at] if isinstance(at, int) else b.loc[at]
    n = len(b.columns) // 4
    bp = [row[f"bid_price_{i}"] for i in range(1, n + 1)]
    bs = [row[f"bid_size_{i}"] for i in range(1, n + 1)]
    ap = [row[f"ask_price_{i}"] for i in range(1, n + 1)]
    asz = [row[f"ask_size_{i}"] for i in range(1, n + 1)]
    gaps = _np.diff(sorted(set(bp + ap)))
    height = 0.8 * (gaps[gaps > 0].min() if (gaps > 0).any() else 1.0)
    fig, ax = _plt.subplots(figsize=figsize)
    ax.barh(bp, [-s for s in bs], height=height, color=up, label="Bid")
    ax.barh(ap, asz, height=height, color=down, label="Ask")
    mid = (bp[0] + ap[0]) / 2
    ax.axhline(mid, color="black", linewidth=0.8, linestyle="--", label="Mid")
    ax.axvline(0, color="black", linewidth=0.6)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(t):g}" for t in ticks])
    ax.set_xlabel("Size  (bid ←  → ask)", fontname=fontname)
    ax.set_ylabel("Price", fontname=fontname, fontweight="bold")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    sub = None
    if subtitle:
        imb = (bs[0] - asz[0]) / (bs[0] + asz[0]) if bs[0] + asz[0] > 0 else _np.nan
        label = row.name if not isinstance(row.name, int) else f"row {row.name}"
        sub = f"{label} | spread = {ap[0] - bp[0]:g} | queue imbalance = {imb:+.2f}"
    _title(ax, title, sub, fontname)
    return _finish(fig, savefig, show)


def lob_timeseries(
    book: _pd.DataFrame,
    model: _lob.MicropriceModel | None = None,
    title: str = "Mid, Weighted Mid & Imbalance",
    figsize: tuple[float, float] = (10, 6),
    grayscale: bool = False,
    fontname: str = "Arial",
    subtitle: bool = True,
    savefig: Any = None,
    show: bool = True,
):
    """
    Best quotes, mid, weighted mid (and micro-price if a model is given)
    above the queue imbalance.
    """
    up, down, line = _palette(grayscale)
    fig, (ax, ax2) = _plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    x = book.index
    ax.step(
        x,
        _lob._get(book, "bid", "price"),
        where="post",
        color=up,
        lw=0.8,
        label="Best bid",
    )
    ax.step(
        x,
        _lob._get(book, "ask", "price"),
        where="post",
        color=down,
        lw=0.8,
        label="Best ask",
    )
    ax.plot(x, _lob.mid_price(book), color="black", lw=0.8, label="Mid")
    ax.plot(
        x, _lob.weighted_mid(book), color=line, lw=1, alpha=0.7, label="Weighted mid"
    )
    if model is not None:
        ax.plot(
            x,
            model.predict(book),
            color=_ACCENT if not grayscale else "#777777",
            lw=1.4,
            label="Micro-price",
        )
    ax.set_ylabel("Price", fontname=fontname, fontweight="bold")
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")
    ax.legend(loc="upper left", fontsize=8, ncol=5)
    ax.grid(True, alpha=0.3)
    imb = _lob.queue_imbalance(book)
    ax2.fill_between(x, 0, imb.where(imb > 0, 0), color=up, step="post")
    ax2.fill_between(x, 0, imb.where(imb < 0, 0), color=down, step="post")
    ax2.set_ylim(-1, 1)
    ax2.set_ylabel("Imbalance", fontname=fontname)
    ax2.grid(True, alpha=0.3)
    sub = f"{len(book):,} snapshots" if subtitle else None
    _title(ax, title, sub, fontname)
    return _finish(fig, savefig, show)
