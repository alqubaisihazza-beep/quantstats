import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import AppContext, pct
from data.dividend_data import dividend_summary
from engines.dividend_engine import DividendEngine
from engines.reinvestment import simulate_drip


def render(ctx: AppContext) -> None:
    st.header("Dividends & Income")
    res = ctx.strategy()
    w = res.weights
    eng = DividendEngine(ctx.dividends, ctx.prices.close)

    value = st.number_input("Portfolio value ($)", 1000.0, 1e9, float(ctx.cfg["backtest"]["initial_capital"]), step=1000.0)
    holdings = eng.holdings_from_weights(w, value)
    inc = eng.forward_income(holdings)
    total = inc["annual_income"].sum()
    growth = eng.expected_growth(w)

    c = st.columns(4)
    c[0].metric("Est. annual income", f"${total:,.0f}")
    c[1].metric("Est. monthly income", f"${total / 12:,.0f}")
    c[2].metric("Portfolio yield", pct(total / value, 2))
    c[3].metric("Wtd. 5y dividend growth", pct(growth))

    st.dataframe(inc.style.format({"shares": "{:,.2f}", "price": "${:.2f}", "value": "${:,.0f}", "ttm_dps": "${:.3f}",
                                   "annual_income": "${:,.0f}", "monthly_income": "${:,.0f}", "yield": "{:.2%}",
                                   "income_share": "{:.1%}"}), width="stretch")

    st.subheader("Projected income calendar (next 12 months)")
    cal = eng.income_calendar(holdings)
    st.plotly_chart(px.bar(cal.drop(columns="total"), labels={"value": "$", "index": "month"}), width="stretch")

    st.subheader("Income projection")
    c = st.columns(3)
    years = c[0].slider("Years", 1, 40, 20)
    pg = c[1].number_input("Price growth %/yr", -10.0, 20.0, 4.0) / 100
    contrib = c[2].number_input("Annual contribution $", 0.0, 1e7, 0.0, step=1000.0)
    y0 = total / value if value else 0
    drip = DividendEngine.project_income(value, y0, growth, pg, years, True, contrib)
    nodrip = DividendEngine.project_income(value, y0, growth, pg, years, False, contrib)
    fig = go.Figure()
    fig.add_bar(x=drip.index, y=drip["income"], name="income (DRIP)")
    fig.add_bar(x=nodrip.index, y=nodrip["income"], name="income (no DRIP)")
    fig.update_layout(barmode="group", xaxis_title="year", yaxis_title="$ / year")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Historical DRIP simulation")
    years_back = st.slider("Years of history", 1, 10, 5)
    close = ctx.prices.close[list(w.index)].dropna()
    close = close.loc[close.index[-1] - pd.DateOffset(years=years_back):]
    start_px = close.iloc[0]
    shares0 = {t: value * w[t] / start_px[t] for t in w.index}
    sims = {p: simulate_drip(close, ctx.dividends, shares0, policy=p, target_weights=w)
            for p in ("same", "underweight", "cash")}
    fig2 = go.Figure()
    for p, r in sims.items():
        fig2.add_scatter(x=r.value.index, y=r.value, name=f"{p} (income ${r.total_income:,.0f})")
    st.plotly_chart(fig2, width="stretch")

    with st.expander("Per-ETF dividend statistics"):
        summ = dividend_summary({t: ctx.dividends.get(t) for t in ctx.prices.tickers if t in ctx.dividends},
                                ctx.prices.close)
        st.dataframe(summ.style.format({"ttm_dividend": "${:.3f}", "yield": "{:.2%}", "div_cagr_5y": "{:.2%}",
                                        "div_consistency": "{:.0%}"}, na_rep="n/a"), width="stretch")
