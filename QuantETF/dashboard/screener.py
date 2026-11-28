import plotly.express as px
import streamlit as st

from dashboard import AppContext
from engines.etf_scoring import DIRECTION, compute_metrics, score_etfs
from engines.etf_screener import ETFScreener, ScreenCriteria


def render(ctx: AppContext) -> None:
    st.header("ETF Screener")
    defaults = ctx.cfg.get("screener", {})
    uni = ctx.universe

    with st.form("screen"):
        c = st.columns(4)
        max_er = c[0].number_input("Max expense ratio (%)", 0.0, 2.0, float(defaults.get("max_expense_ratio", 0.006)) * 100,
                                   step=0.01) / 100
        min_yield = c[1].number_input("Min yield (%)", 0.0, 20.0, 0.0, step=0.25) / 100
        min_hist = c[2].number_input("Min history (years)", 0.0, 30.0, float(defaults.get("min_history_years", 3)))
        min_dv = c[3].number_input("Min avg $ volume (M)", 0.0, 10_000.0,
                                   float(defaults.get("min_avg_dollar_volume", 5e6)) / 1e6) * 1e6
        c2 = st.columns(2)
        classes = c2[0].multiselect("Asset classes", sorted(uni["asset_class"].unique()))
        cats = c2[1].multiselect("Categories", sorted(uni["category"].unique()))

        st.markdown("**Scoring weights**")
        sw = ctx.cfg.get("scoring", {})
        cols = st.columns(5)
        weights = {m: cols[i % 5].slider(m, 0.0, 1.0, float(sw.get(m, 0.0)), 0.05) for i, m in enumerate(DIRECTION)}
        submitted = st.form_submit_button("Run screen")

    crit = ScreenCriteria(max_expense_ratio=max_er, min_yield=min_yield or None, min_history_years=min_hist,
                          min_avg_dollar_volume=min_dv, asset_classes=classes or None, categories=cats or None)
    screen = ETFScreener(uni).screen(crit, ctx.prices, ctx.dividends)
    passed = [t for t in screen.index[screen["passed"]] if t in ctx.prices.tickers]
    st.write(f"**{len(passed)} / {len(screen)}** ETFs passed." + ("" if submitted else " (defaults)"))

    if passed and any(weights.values()):
        metrics = compute_metrics(ctx.prices.subset(passed), ctx.dividends, uni, ctx.rf)
        scored = score_etfs(metrics, weights).join(uni[["name", "category"]])
        cols = ["rank", "score", "name", "category", "cagr", "volatility", "sharpe", "max_drawdown",
                "yield", "div_cagr_5y", "expense_ratio"]
        fmt = {k: "{:.2%}" for k in ["cagr", "volatility", "max_drawdown", "yield", "div_cagr_5y", "expense_ratio"]}
        fmt.update({"sharpe": "{:.2f}", "score": "{:.1f}"})
        st.dataframe(scored[cols].style.format(fmt, na_rep="n/a"), width="stretch")
        fig = px.scatter(scored.reset_index(), x="volatility", y="cagr", size="score", color="category",
                         hover_name="index", title="Risk / return of passing ETFs")
        st.plotly_chart(fig, width="stretch")

    with st.expander("Rejected ETFs"):
        st.dataframe(screen.loc[~screen["passed"], ["name", "category", "reasons"]], width="stretch")
