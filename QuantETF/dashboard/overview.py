import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import AppContext, pct
from engines.dividend_engine import DividendEngine
from engines.risk_engine import drawdown_series


def render(ctx: AppContext) -> None:
    st.header("Overview")
    res = ctx.strategy()
    bt = ctx.backtest(res.weights, ctx.strategy_name)
    m = bt.metrics
    yld = DividendEngine(ctx.dividends, ctx.prices.close).portfolio_yield(res.weights)

    c = st.columns(5)
    c[0].metric("CAGR", pct(m["cagr"]), f"bench {pct(m.get('benchmark_cagr'))}")
    c[1].metric("Volatility", pct(m["volatility"]))
    c[2].metric("Sharpe", f"{m['sharpe']:.2f}")
    c[3].metric("Max drawdown", pct(m["max_drawdown"]), f"bench {pct(m.get('benchmark_max_drawdown'))}",
                delta_color="off")
    c[4].metric("Portfolio yield (TTM)", pct(yld, 2))

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Target allocation")
        w = res.weights.rename("weight").reset_index().rename(columns={"index": "ticker"})
        st.plotly_chart(px.pie(w, names="ticker", values="weight", hole=0.45), width="stretch")
    with right:
        st.subheader("Growth of capital (backtest, static target weights)")
        fig = go.Figure()
        fig.add_scatter(x=bt.equity.index, y=bt.equity, name=ctx.strategy_name)
        if bt.benchmark_returns is not None:
            bench_eq = bt.equity.iloc[0] * (1 + bt.benchmark_returns).cumprod()
            fig.add_scatter(x=bench_eq.index, y=bench_eq, name=ctx.cfg.get("benchmark"), line={"dash": "dot"})
        fig.update_layout(yaxis_title="$", height=380, margin={"t": 10})
        st.plotly_chart(fig, width="stretch")

    st.subheader("Drawdown")
    dd = drawdown_series(bt.returns)
    st.plotly_chart(px.area(dd, labels={"value": "drawdown", "index": ""}).update_layout(showlegend=False, height=250),
                    width="stretch")

    with st.expander("Backtest details"):
        st.dataframe(pd.Series(m, name="value").astype(object), width="stretch")
        st.caption("Static weights backtest uses today's selection over the full history (look-ahead in ETF "
                   "selection). Use `python app.py backtest --dynamic` for a walk-forward test.")
        if st.button("Save report to reports/"):
            from data import ROOT
            md, js = bt.to_report(ROOT / "reports")
            st.success(f"Saved {md.name} and {js.name}")
