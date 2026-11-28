# QuantETF

ETF research and portfolio tool. It screens and scores ETFs, builds portfolios from four strategies, runs risk and dividend-income analysis, backtests with DRIP and rebalancing, and produces orders for manual or Interactive Brokers execution. It includes a Streamlit dashboard and a CLI.

> Research tool, not investment advice. The expense ratios in `config.yaml` are approximate, so check them with the fund issuer.

## Setup

```bash
cd QuantETF
pip install -r requirements.txt
cp .env.example .env          # only needed for IBKR
```

## Dashboard

```bash
streamlit run app.py
```

| Page | What it shows |
|---|---|
| Overview | Strategy KPIs, target allocation, growth of capital vs the benchmark, drawdown |
| Screener | Interactive filters, adjustable scoring weights, ranked table, risk/return scatter |
| Portfolio | All optimizers compared side by side, holdings editor, drift, rebalance orders, manual order tickets and fills |
| Risk | VaR/CVaR, beta, limit checks, risk contribution vs weight, correlation, rolling volatility, stress scenarios |
| Dividends | Forward income, 12-month income calendar, DRIP vs no-DRIP projection, historical DRIP policies |

Choose `synthetic` as the data provider in the sidebar to run offline on generated data. That data is not real prices.

## CLI

```bash
python app.py screen                                    # screen + score universe
python app.py build --strategy balanced                 # target weights
python app.py backtest --strategy dividend_growth       # static-weight backtest -> reports/
python app.py backtest --strategy dividend_growth --dynamic --tearsheet   # walk-forward + quantstats HTML
python app.py rebalance --strategy index_investing --positions my.csv --submit   # manual order ticket
python app.py rebalance --strategy high_income --buy-only --cash 5000
python app.py fill 12 --price 251.30                    # confirm a manual fill
python app.py rebalance --broker ibkr                   # IBKR (what-if preview by default)
```

Add `--synthetic` to any command to use offline data.

## Layout

```
data/        market_data (yfinance | synthetic, SQLite cache), dividend_data, data_validation, database
engines/     etf_screener, etf_scoring, portfolio_optimizer, risk_engine, dividend_engine,
             reinvestment (DRIP), backtesting, rebalancing
strategies/  dividend_growth, high_income, balanced (55/35/10 sleeves), index_investing (3-fund)
brokers/     manual_execution (CSV/markdown tickets + fill tracking), ibkr (ib_insync)
dashboard/   one Streamlit page per module
tests/       pytest suite (offline, synthetic data)
reports/     generated reports and order tickets (gitignored)
```

## Design notes

- **Pipeline:** screen → score (percentile rank per factor, weighted to a 0–100 score) → pick the top N, or the best per sleeve → optimize. Optimizers are equal weight, inverse volatility, min variance, max Sharpe, risk parity and score weighted. All are long-only with position caps and a shrunk covariance matrix.
- **Backtester:** works day by day on unadjusted closes and pays dividends as explicit cash on ex-dates. That lets it reinvest dividends (DRIP), pay them out, or tax them. Monthly contributions are excluded from performance through time-weighted returns. Rebalancing can be calendar-based (monthly, quarterly, annual) or triggered by drift past a threshold, with transaction costs in bps.
- **Static vs walk-forward backtests:** a static backtest picks ETFs with today's data, so it has look-ahead bias. `--dynamic` reruns the full strategy at each rebalance using only data available at that date. Both use today's ETF universe, so survivorship bias remains.
- **Strategy settings:** you can override any attribute (`top_n`, `max_weight`, `optimizer`, `screen`, `score_weights`, `buckets`) per strategy under `strategies:` in `config.yaml`.
- **IBKR safety:** `dry_run: true` (the default) only previews orders with what-if checks. Real orders need both `dry_run: false` and `QUANTETF_ALLOW_LIVE=1`. Start with a paper account (port 7497).

## Tests

```bash
python -m pytest
```
