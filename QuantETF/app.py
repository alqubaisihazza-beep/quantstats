"""QuantETF entry point.

Dashboard:  streamlit run app.py
CLI:        python app.py {screen,build,backtest,rebalance,fill} --help
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from data import ROOT, load_config
from strategies import STRATEGIES

PAGES = ["Overview", "Screener", "Portfolio", "Risk", "Dividends"]


# =========================================================================== dashboard
def run_dashboard() -> None:
    import streamlit as st

    from dashboard import dividends, load_context, overview, portfolio, risk, screener

    st.set_page_config(page_title="QuantETF", layout="wide")
    cfg = load_config()

    with st.sidebar:
        st.title("QuantETF")
        page = st.radio("Page", PAGES)
        strat = st.selectbox("Strategy", list(STRATEGIES), format_func=lambda s: s.replace("_", " ").title())
        st.caption(STRATEGIES[strat].description)
        provider = st.selectbox("Data provider", ["yfinance", "synthetic"],
                                index=0 if cfg["data"].get("provider") == "yfinance" else 1)
        start = st.date_input("History start", pd.Timestamp(cfg["data"].get("start_date", "2015-01-01")))
        refresh = st.button("Refresh market data")

    cfg["data"]["provider"] = provider
    if provider == "synthetic":
        cfg["data"]["database"] = "data/quantetf_synthetic.db"     # never mix fake and real data

    @st.cache_resource(show_spinner="Loading market data...")
    def _ctx(provider: str, start: str, nonce: int):
        return load_context(cfg, start, refresh=nonce > 0)

    if refresh:
        st.session_state["nonce"] = st.session_state.get("nonce", 0) + 1
    ctx = _ctx(provider, str(start), st.session_state.get("nonce", 0))
    ctx.strategy_name = strat

    if ctx.synthetic:
        st.warning("Synthetic demo data - NOT real market prices.")
    if ctx.prices.close.empty:
        st.error("No market data loaded. Check your network or switch the provider to 'synthetic'.")
        return
    if ctx.prices.validation is not None and ctx.prices.validation.issues:
        with st.sidebar.expander(f"Data quality ({len(ctx.prices.validation.issues)} issues)"):
            st.dataframe(ctx.prices.validation.to_frame(), width="stretch")
    st.caption(f"Data through {ctx.prices.close.index[-1].date()} | Not investment advice.")

    pages = {"Overview": overview, "Screener": screener, "Portfolio": portfolio, "Risk": risk, "Dividends": dividends}
    try:
        pages[page].render(ctx)
    except ValueError as exc:
        st.error(str(exc))


# =========================================================================== CLI
def _context(args):
    from dashboard import load_context

    cfg = load_config(args.config)
    if args.synthetic:
        cfg["data"]["provider"] = "synthetic"
        cfg["data"]["database"] = "data/quantetf_synthetic.db"
    ctx = load_context(cfg, args.start, refresh=args.refresh)
    if ctx.prices.close.empty:
        sys.exit("error: no market data loaded (network down? try --synthetic)")
    return ctx


def cmd_screen(args) -> None:
    from engines.etf_scoring import compute_metrics, score_etfs
    from engines.etf_screener import ETFScreener, ScreenCriteria

    ctx = _context(args)
    screen = ETFScreener(ctx.universe).screen(ScreenCriteria.from_dict(ctx.cfg.get("screener")), ctx.prices, ctx.dividends)
    passed = [t for t in screen.index[screen["passed"]] if t in ctx.prices.tickers]
    scored = score_etfs(compute_metrics(ctx.prices.subset(passed), ctx.dividends, ctx.universe, ctx.rf),
                        ctx.cfg["scoring"])
    cols = ["score", "cagr", "volatility", "sharpe", "max_drawdown", "yield", "expense_ratio"]
    print(scored[cols].round(4).to_string())
    rejected = screen.loc[~screen["passed"], "reasons"]
    if not rejected.empty:
        print("\nRejected:\n" + rejected.to_string())


def cmd_build(args) -> None:
    ctx = _context(args)
    res = ctx.strategy(args.strategy)
    print(f"{args.strategy} as of {res.as_of.date()}")
    print(res.weights.map("{:.2%}".format).to_string())


def cmd_backtest(args) -> None:
    from engines.backtesting import Backtester, BacktestConfig
    from strategies import get_strategy

    ctx = _context(args)
    bcfg = BacktestConfig.from_dict(ctx.cfg.get("backtest"))
    if args.rebalance:
        bcfg.rebalance = args.rebalance
    bt = Backtester(ctx.prices, ctx.dividends, bcfg, benchmark=ctx.cfg.get("benchmark"), rf=ctx.rf)
    if args.dynamic:
        strat = get_strategy(args.strategy, ctx.universe, ctx.cfg)
        start = ctx.prices.close.index[0] + pd.DateOffset(years=3)          # warm-up for scoring
        res = bt.run(strat.target_fn(), start=start, name=f"{args.strategy}_walkforward")
    else:
        res = bt.run(ctx.strategy(args.strategy).weights, name=args.strategy)
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.metrics.items()}
    print(json.dumps(out, indent=2))
    md, js = res.to_report(ROOT / "reports")
    print(f"\nreport: {md}\n        {js}")
    if args.tearsheet:
        path = res.tearsheet(ROOT / "reports" / f"{res.name}_tearsheet.html")
        if path:
            print(f"tearsheet: {path}")


def cmd_rebalance(args) -> None:
    from brokers.manual_execution import ManualExecutionBroker
    from engines.rebalancing import Rebalancer

    ctx = _context(args)
    target = ctx.strategy(args.strategy).weights
    last = ctx.prices.close.ffill().iloc[-1]
    broker_name = args.broker or ctx.cfg.get("brokers", {}).get("default", "manual")
    reb = Rebalancer.from_config(ctx.cfg)

    if broker_name == "ibkr":
        from brokers.ibkr import IBKRBroker

        with IBKRBroker.from_config(ctx.cfg) as ib:
            pos = ib.get_positions()
            holdings, cash = pos["shares"].to_dict(), ib.get_cash()
            orders = (reb.cash_flow_orders if args.buy_only else reb.generate_orders)(holdings, last, target, cash)
            for r in ib.submit_orders(orders):
                print(r)
        return

    acct = ctx.cfg.get("brokers", {}).get("manual", {}).get("account", "manual")
    broker = ManualExecutionBroker(ctx.db, acct, ROOT / "reports")
    if args.positions:
        broker.import_positions_csv(args.positions)
    holdings = broker.get_positions()["shares"].to_dict()
    cash = float(args.cash) if args.cash is not None else broker.get_cash()
    print(reb.drift(holdings, last, target, cash).round(4).to_string())
    orders = (reb.cash_flow_orders if args.buy_only else reb.generate_orders)(holdings, last, target, cash)
    if not orders:
        print("\nNo orders needed.")
        return
    print()
    for o in orders:
        print(f"{o.side:4} {o.shares:>10g} {o.ticker:6} @ ~${o.est_price:,.2f}  (${o.est_value:,.2f})")
    if args.submit:
        out = broker.submit_orders(orders, note=args.strategy)
        print(f"\nticket: {out['ticket']}")


def cmd_fill(args) -> None:
    from brokers.manual_execution import ManualExecutionBroker
    from data import resolve_path
    from data.database import Database

    cfg = load_config(args.config)
    db = Database(resolve_path(cfg["data"]["database"]))
    acct = cfg.get("brokers", {}).get("manual", {}).get("account", "manual")
    ManualExecutionBroker(db, acct).confirm_fill(args.trade_id, args.price, args.shares)
    print(f"trade {args.trade_id} filled at {args.price}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="quantetf", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=None)
    common.add_argument("--start", default=None, help="history start date (default from config)")
    common.add_argument("--synthetic", action="store_true", help="use offline synthetic demo data")
    common.add_argument("--refresh", action="store_true", help="re-download data, ignore cache")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("screen", parents=[common], help="screen + score the universe").set_defaults(fn=cmd_screen)

    b = sub.add_parser("build", parents=[common], help="target weights for a strategy")
    b.add_argument("--strategy", choices=list(STRATEGIES), default="dividend_growth")
    b.set_defaults(fn=cmd_build)

    bt = sub.add_parser("backtest", parents=[common], help="backtest a strategy and write a report")
    bt.add_argument("--strategy", choices=list(STRATEGIES), default="dividend_growth")
    bt.add_argument("--dynamic", action="store_true", help="walk-forward: re-select ETFs at each rebalance")
    bt.add_argument("--rebalance", choices=["none", "monthly", "quarterly", "annual", "threshold"])
    bt.add_argument("--tearsheet", action="store_true", help="also write a quantstats HTML tearsheet")
    bt.set_defaults(fn=cmd_backtest)

    r = sub.add_parser("rebalance", parents=[common], help="compute rebalance orders")
    r.add_argument("--strategy", choices=list(STRATEGIES), default="dividend_growth")
    r.add_argument("--broker", choices=["manual", "ibkr"])
    r.add_argument("--positions", help="CSV of current positions (ticker,shares[,cost_basis]; CASH row for cash)")
    r.add_argument("--cash", type=float, help="override cash balance")
    r.add_argument("--buy-only", action="store_true", help="invest cash into underweights, no sells")
    r.add_argument("--submit", action="store_true", help="manual broker: record orders and write a ticket")
    r.set_defaults(fn=cmd_rebalance)

    f = sub.add_parser("fill", help="confirm a manual order fill")
    f.add_argument("trade_id", type=int)
    f.add_argument("--price", type=float, required=True)
    f.add_argument("--shares", type=float)
    f.add_argument("--config", default=None)
    f.set_defaults(fn=cmd_fill)

    args = p.parse_args(argv)
    args.fn(args)


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime import exists

        return exists()
    except Exception:
        return False


if _in_streamlit():
    run_dashboard()
elif __name__ == "__main__":
    main()
