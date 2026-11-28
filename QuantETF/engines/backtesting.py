"""Daily event-driven backtester with explicit dividends, DRIP, contributions, costs and rebalancing."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Union

import numpy as np
import pandas as pd

from engines import risk_engine as rk
from engines.rebalancing import is_calendar_rebalance

TargetSpec = Union[pd.Series, Callable[[pd.Timestamp, object, dict], pd.Series]]


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    rebalance: str = "quarterly"          # none | monthly | quarterly | annual | threshold
    threshold: float = 0.05
    transaction_cost_bps: float = 5.0
    reinvest_dividends: bool = True
    dividend_tax_rate: float = 0.0
    monthly_contribution: float = 0.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "BacktestConfig":
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series                     # time-weighted (contributions removed)
    weights: pd.DataFrame
    trades: pd.DataFrame
    income: pd.Series
    contributions: pd.Series
    metrics: dict
    config: BacktestConfig
    benchmark_returns: pd.Series | None = None
    name: str = "backtest"
    rebalance_dates: list = field(default_factory=list)

    def to_report(self, output_dir: str | Path = "reports") -> tuple[Path, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"quantetf_backtest_{self.name}_{date.today().isoformat()}"
        md, js = out / f"{stem}.md", out / f"{stem}.json"
        final_w = self.weights.iloc[-1]
        final_w = final_w[final_w > 1e-4].sort_values(ascending=False)
        lines = [
            f"# QuantETF Backtest - {self.name}",
            f"**Generated:** {pd.Timestamp.now():%Y-%m-%d %H:%M}  ",
            f"**Period:** {self.equity.index[0].date()} to {self.equity.index[-1].date()}  ",
            f"**Rebalance:** {self.config.rebalance} | **Costs:** {self.config.transaction_cost_bps} bps | "
            f"**DRIP:** {self.config.reinvest_dividends}",
            "",
            "## Metrics",
            "| Metric | Value |",
            "|---|---|",
            *[f"| {k} | {_fmt(k, v)} |" for k, v in self.metrics.items()],
            "",
            "## Final weights",
            "| Ticker | Weight |",
            "|---|---|",
            *[f"| {t} | {w:.1%} |" for t, w in final_w.items()],
            "",
            "_Past performance does not guarantee future results. Not investment advice._",
        ]
        md.write_text("\n".join(lines), encoding="utf-8")
        payload = {
            "name": self.name,
            "config": asdict(self.config),
            "metrics": {k: (None if v is None or (isinstance(v, float) and np.isnan(v)) else v)
                        for k, v in self.metrics.items()},
            "final_weights": final_w.round(6).to_dict(),
            "equity": {d.strftime("%Y-%m-%d"): round(float(v), 2) for d, v in self.equity.resample("ME").last().items()},
        }
        js.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return md, js

    def tearsheet(self, path: str | Path) -> Path | None:
        """Full HTML tearsheet via quantstats (optional dependency)."""
        try:
            import quantstats as qs
        except ImportError:
            print("quantstats not installed: pip install -r requirements.txt", file=sys.stderr)
            return None
        qs.reports.html(self.returns, benchmark=self.benchmark_returns, output=str(path), title=self.name)
        return Path(path)


def _fmt(k, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if k in {"sharpe", "sortino", "calmar", "beta", "n_rebalances"}:
        return f"{v:.2f}" if isinstance(v, float) else str(v)
    if k in {"final_value", "total_income", "total_contributions", "total_costs"}:
        return f"${v:,.0f}"
    return f"{v:.2%}"


class Backtester:
    def __init__(self, prices, dividends: dict[str, pd.Series] | None = None,
                 config: BacktestConfig | None = None, benchmark: str | None = None, rf: float = 0.0):
        self.prices = prices
        self.dividends = dividends or {}
        self.config = config or BacktestConfig()
        self.benchmark = benchmark
        self.rf = rf

    def _normalize(self, w: pd.Series, p: pd.Series) -> pd.Series:
        w = w[(w > 0) & w.index.isin(p.index)]
        w = w[p.reindex(w.index).notna() & (p.reindex(w.index) > 0)]
        if w.empty:
            raise ValueError("Target weights contain no tradable tickers")
        return w / w.sum()

    def run(self, target: TargetSpec, start=None, end=None, warmup_days: int = 0, name: str = "backtest") -> BacktestResult:
        cfg = self.config
        close = self.prices.close.loc[start:end].ffill()
        dynamic = callable(target)
        if not dynamic:
            target = target[target > 0]
            close = close.dropna(subset=list(target.index))     # start when all assets trade
        if len(close) <= warmup_days + 1:
            raise ValueError("Not enough price history for backtest")
        dates = close.index[warmup_days:]
        tickers = list(close.columns)
        divs = pd.DataFrame({t: self.dividends.get(t, pd.Series(dtype=float)) for t in tickers}).reindex(close.index).fillna(0.0)
        bps = cfg.transaction_cost_bps / 1e4

        def get_target(dt):
            if dynamic:
                hist = self.prices.slice(end=dt)
                dhist = {t: s.loc[:dt] for t, s in self.dividends.items()}
                return target(dt, hist, dhist)
            return target

        shares = pd.Series(0.0, index=tickers)
        cash = float(cfg.initial_capital)
        cur_target = None
        eq, flows, inc, wts, trades, rebal_dates = [], [], [], [], [], []
        total_costs = 0.0
        prev = None

        def trade_to(dt, p, w):
            nonlocal shares, cash, total_costs
            total = float((shares * p.fillna(0)).sum() + cash)
            desired = (w.reindex(tickers).fillna(0.0) * total)
            cur_val = shares * p.fillna(0)
            delta = desired - cur_val
            cost = float(delta.abs().sum() * bps)
            new_shares = (w.reindex(tickers).fillna(0.0) * (total - cost) / p).fillna(0.0)
            for t in tickers:
                d_sh = new_shares[t] - shares[t]
                if pd.notna(p[t]) and abs(d_sh * p[t]) > 0.01:
                    trades.append({"date": dt, "ticker": t, "side": "BUY" if d_sh > 0 else "SELL",
                                   "shares": abs(d_sh), "price": float(p[t]), "value": abs(d_sh * p[t])})
            shares, cash = new_shares, 0.0
            total_costs += cost
            rebal_dates.append(dt)

        def buy_with(dt, p, amount, w):
            """Invest `amount` of cash pro-rata to w (no selling)."""
            nonlocal shares, cash, total_costs
            w = w[w > 0]
            cost = amount * bps
            for t, wt in w.items():
                spend = (amount - cost) * wt
                shares[t] += spend / p[t]
                trades.append({"date": dt, "ticker": t, "side": "BUY", "shares": spend / p[t],
                               "price": float(p[t]), "value": spend})
            cash -= amount
            total_costs += cost

        for dt in dates:
            p = close.loc[dt]
            flow = 0.0
            if prev is None:
                cur_target = self._normalize(get_target(dt), p)
                trade_to(dt, p, cur_target)
                income_today = 0.0
            else:
                # 1. dividends on ex-date
                dps = divs.loc[dt]
                paid = shares * dps * (1 - cfg.dividend_tax_rate)
                income_today = float(paid.sum())
                cash += income_today
                # 2. contributions on first trading day of each month
                if cfg.monthly_contribution and dt.month != prev.month:
                    cash += cfg.monthly_contribution
                    flow = cfg.monthly_contribution
                # 3. rebalance?
                rebalance = is_calendar_rebalance(dt, prev, cfg.rebalance)
                if cfg.rebalance == "threshold":
                    vals = shares * p.fillna(0)
                    tot = vals.sum() + cash
                    drift = (vals / tot - cur_target.reindex(tickers).fillna(0)).abs().max()
                    rebalance = drift > cfg.threshold
                if not cfg.reinvest_dividends and income_today > 0:
                    # dividends are paid out to the investor (an outflow), not reinvested
                    cash -= income_today
                    flow -= income_today
                if rebalance:
                    cur_target = self._normalize(get_target(dt), p)
                    trade_to(dt, p, cur_target)
                else:
                    if cfg.reinvest_dividends and income_today > 0:
                        for t, amt in paid[paid > 0].items():
                            buy_with(dt, p, float(amt), pd.Series({t: 1.0}))
                    if cfg.monthly_contribution and dt.month != prev.month:
                        buy_with(dt, p, cfg.monthly_contribution, cur_target)
            value = float((shares * p.fillna(0)).sum() + cash)
            eq.append(value)
            flows.append(flow)
            inc.append(income_today)
            wts.append((shares * p.fillna(0)) / value if value > 0 else shares * 0)
            prev = dt

        equity = pd.Series(eq, index=dates, name=name)
        flows_s = pd.Series(flows, index=dates)
        rets = ((equity - flows_s) / equity.shift(1) - 1).iloc[1:]
        bench = None
        if self.benchmark and self.benchmark in self.prices.adj_close.columns:
            bench = self.prices.adj_close[self.benchmark].reindex(dates).pct_change(fill_method=None).iloc[1:]

        weights = pd.DataFrame(wts, index=dates)
        trades_df = pd.DataFrame(trades, columns=["date", "ticker", "side", "shares", "price", "value"])
        traded = trades_df.groupby("date")["value"].sum().reindex(dates).fillna(0) / equity
        years = max(len(rets) / 252, 1e-9)

        metrics = rk.summary_metrics(rets, self.rf, bench)
        metrics.update({
            "total_return": float((1 + rets).prod() - 1),
            "final_value": float(equity.iloc[-1]),
            "total_income": float(sum(inc)),
            "total_contributions": float(flows_s[flows_s > 0].sum()),
            "total_costs": total_costs,
            "annual_turnover": float(traded.iloc[1:].sum() / years),
            "n_rebalances": len(rebal_dates) - 1,
        })
        if bench is not None:
            metrics["benchmark_cagr"] = rk.annualized_return(bench)
            metrics["benchmark_max_drawdown"] = rk.max_drawdown(bench)
        return BacktestResult(equity, rets, weights, trades_df, pd.Series(inc, index=dates), flows_s,
                              metrics, cfg, bench, name, rebal_dates)
