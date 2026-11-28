import pandas as pd
import plotly.express as px
import streamlit as st

from brokers.manual_execution import ManualExecutionBroker
from dashboard import AppContext, pct
from data import ROOT
from engines.portfolio_optimizer import METHODS, PortfolioOptimizer
from engines.rebalancing import Rebalancer


def render(ctx: AppContext) -> None:
    st.header("Portfolio")
    res = ctx.strategy()
    tickers = res.selected

    # ------------------------------------------------------------ optimizer comparison
    st.subheader("Optimizer comparison (selected ETFs)")
    rets = ctx.prices.returns()[tickers].iloc[-756:].dropna()
    opt = PortfolioOptimizer(rets, max_weight=ctx.cfg.get("risk", {}).get("max_position_weight", 1.0), rf=ctx.rf)
    rows, all_w = [], {}
    for m in METHODS:
        try:
            w = opt.optimize(m, scores=res.scores["score"])
        except ValueError:
            continue
        all_w[m] = w
        rows.append({"method": m, **opt.stats(w)})
    all_w["strategy"] = res.weights
    rows.append({"method": "strategy", **opt.stats(res.weights)})
    stats = pd.DataFrame(rows).set_index("method")
    st.dataframe(stats.style.format({"expected_return": "{:.2%}", "volatility": "{:.2%}", "sharpe": "{:.2f}"}),
                 width="stretch")
    wdf = pd.DataFrame(all_w).fillna(0)
    st.plotly_chart(px.bar(wdf.T, barmode="stack", labels={"value": "weight", "index": "method"}),
                    width="stretch")
    choice = st.selectbox("Weights to trade", list(all_w), index=list(all_w).index("strategy"))
    target = all_w[choice]

    # ------------------------------------------------------------ holdings + rebalance
    st.subheader("Holdings & rebalance")
    acct = ctx.cfg.get("brokers", {}).get("manual", {}).get("account", "manual")
    broker = ManualExecutionBroker(ctx.db, acct, ROOT / "reports")
    held = broker.get_positions()
    base = pd.DataFrame({"shares": 0.0}, index=sorted(set(target.index) | set(held.index)))
    base.loc[held.index, "shares"] = held["shares"]
    edited = st.data_editor(base.rename_axis("ticker").reset_index(), num_rows="dynamic", key="holdings")
    cash = st.number_input("Cash ($)", 0.0, 1e9, float(broker.get_cash() or ctx.cfg["backtest"]["initial_capital"]))
    if st.button("Save holdings"):
        for _, r in edited.iterrows():
            if r["ticker"]:
                prev_cb = float(held.loc[r["ticker"], "cost_basis"]) if r["ticker"] in held.index else 0.0
                ctx.db.set_holding(acct, str(r["ticker"]).upper(), float(r["shares"] or 0), prev_cb)
        broker.set_cash(cash)
        st.success("Saved")

    holdings = {str(r["ticker"]).upper(): float(r["shares"] or 0) for _, r in edited.iterrows() if r["ticker"]}
    last = ctx.prices.close.ffill().iloc[-1]
    missing = [t for t in holdings if t not in last.index]
    if missing:
        st.warning(f"No prices for {missing}; ignored.")
        holdings = {t: s for t, s in holdings.items() if t in last.index}

    reb = Rebalancer.from_config(ctx.cfg)
    drift = reb.drift(holdings, last, target, cash)
    st.dataframe(drift.style.format({"current_weight": "{:.1%}", "target_weight": "{:.1%}", "drift": "{:+.1%}",
                                     "value": "${:,.0f}"}), width="stretch")
    mode = st.radio("Order mode", ["Full rebalance", "Buy-only with cash"], horizontal=True)
    orders = (reb.generate_orders(holdings, last, target, cash) if mode == "Full rebalance"
              else reb.cash_flow_orders(holdings, last, target, cash))
    st.write(f"Needs rebalance (> {pct(reb.threshold, 0)} drift): **{reb.needs_rebalance(holdings, last, target, cash)}**")
    if orders:
        st.dataframe(pd.DataFrame([o.to_dict() for o in orders]), width="stretch")
        if st.button("Create manual order ticket"):
            out = broker.submit_orders(orders, note=f"{ctx.strategy_name}/{choice}")
            st.success(f"Ticket written: {out['ticket'].name}")
    else:
        st.info("No orders needed.")

    pending = broker.pending()
    if not pending.empty:
        st.subheader("Pending manual orders")
        st.dataframe(pending, width="stretch")
        with st.form("fill"):
            c = st.columns(3)
            tid = c[0].selectbox("Trade id", pending["id"].tolist())
            price = c[1].number_input("Fill price", 0.0, 1e6, float(pending.set_index("id").loc[tid, "price"]))
            if c[2].form_submit_button("Confirm fill"):
                broker.confirm_fill(int(tid), price)
                st.success(f"Trade {tid} filled")
