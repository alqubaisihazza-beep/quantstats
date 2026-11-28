import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import AppContext, pct
from engines.risk_engine import RiskEngine, stress_test


def render(ctx: AppContext) -> None:
    st.header("Risk")
    res = ctx.strategy()
    w = res.weights
    rcfg = ctx.cfg.get("risk", {})
    conf = float(rcfg.get("var_confidence", 0.95))
    lookback = st.select_slider("Lookback (trading days)", [252, 504, 756, 1260], value=756)
    rets = ctx.prices.returns()[list(w.index)].iloc[-lookback:].dropna()
    bench = ctx.cfg.get("benchmark")
    bench_r = ctx.prices.returns()[bench].reindex(rets.index) if bench in ctx.prices.tickers else None
    eng = RiskEngine(rets, ctx.rf, conf, bench_r)
    rep = eng.report(w)

    c = st.columns(6)
    c[0].metric("Volatility", pct(rep["volatility"]))
    c[1].metric(f"1-day VaR {conf:.0%}", pct(rep["var"], 2))
    c[2].metric(f"1-day CVaR {conf:.0%}", pct(rep["cvar"], 2))
    c[3].metric("Max drawdown", pct(rep["max_drawdown"]))
    c[4].metric("Sortino", f"{rep['sortino']:.2f}")
    c[5].metric("Beta", f"{rep.get('beta', np.nan):.2f}")

    breaches = eng.check_limits(w, rcfg)
    if breaches:
        for b in breaches:
            st.error(b)
    else:
        st.success("All configured risk limits satisfied.")

    left, right = st.columns(2)
    with left:
        st.subheader("Risk contribution vs weight")
        rc = pd.DataFrame({"weight": w, "risk_contribution": rep["risk_contributions"]})
        st.plotly_chart(px.bar(rc, barmode="group"), width="stretch")
    with right:
        st.subheader("Correlation")
        st.plotly_chart(px.imshow(eng.correlation(), text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r"),
                        width="stretch")

    st.subheader("Rolling 63-day volatility")
    pr = eng.portfolio_returns(w)
    st.plotly_chart(px.line(pr.rolling(63).std() * np.sqrt(252), labels={"value": "ann. vol", "index": ""})
                    .update_layout(showlegend=False, height=260), width="stretch")

    st.subheader("Stress scenarios")
    scen = rcfg.get("stress_scenarios", {})
    if scen:
        st_df = stress_test(w, ctx.universe["asset_class"], scen)
        pv = st.number_input("Portfolio value ($)", 0.0, 1e9, float(ctx.cfg["backtest"]["initial_capital"]))
        st_df["pnl_$"] = st_df["portfolio_return"] * pv
        st.dataframe(st_df.style.format({"portfolio_return": "{:+.1%}", "pnl_$": "${:,.0f}"}), width="stretch")
