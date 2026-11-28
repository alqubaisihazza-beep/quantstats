"""Market data providers (yfinance / synthetic) with SQLite caching."""

from __future__ import annotations

import sys
import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data.data_validation import ValidationReport, clean_prices, validate_prices
from data.database import Database

PRICE_COLS = ["close", "adj_close", "volume"]


@dataclass
class PriceHistory:
    """Wide frames (index = dates, columns = tickers).

    close     : traded price (dividends NOT included)
    adj_close : total-return price (dividends reinvested)
    volume    : shares traded
    """

    close: pd.DataFrame
    adj_close: pd.DataFrame
    volume: pd.DataFrame
    validation: ValidationReport | None = None

    @property
    def tickers(self) -> list[str]:
        return list(self.close.columns)

    def returns(self, total: bool = True) -> pd.DataFrame:
        px = self.adj_close if total else self.close
        return px.pct_change(fill_method=None).iloc[1:]

    def slice(self, start=None, end=None) -> "PriceHistory":
        sl = slice(start, end)
        return PriceHistory(self.close.loc[sl], self.adj_close.loc[sl], self.volume.loc[sl], self.validation)

    def subset(self, tickers) -> "PriceHistory":
        tickers = [t for t in tickers if t in self.close.columns]
        return PriceHistory(self.close[tickers], self.adj_close[tickers], self.volume[tickers], self.validation)


# --------------------------------------------------------------------------- providers
class YFinanceProvider:
    name = "yfinance"

    def fetch_prices(self, ticker: str, start, end=None) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:  # pragma: no cover
            print("yfinance is not installed: pip install -r requirements.txt", file=sys.stderr)
            raise
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, actions=False,
                         progress=False, timeout=30)
        if df is None or df.empty:
            return pd.DataFrame(columns=PRICE_COLS)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        out = pd.DataFrame(
            {"close": df["Close"], "adj_close": df.get("Adj Close", df["Close"]), "volume": df["Volume"]}
        )
        out.index = pd.to_datetime(out.index).tz_localize(None)
        return out

    def fetch_dividends(self, ticker: str, start=None, end=None) -> pd.Series:
        import yfinance as yf

        s = yf.Ticker(ticker).dividends
        if s is None or s.empty:
            return pd.Series(dtype=float, name=ticker)
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        return s.loc[start:end].rename(ticker)


class SyntheticProvider:
    """Deterministic fake data for offline demos and tests. NOT real market data."""

    name = "synthetic"

    # (annual drift, annual vol, dividend yield, payments/year) by asset class/category
    PROFILES = {
        "equity": (0.08, 0.17, 0.015, 4),
        "dividend_growth": (0.08, 0.15, 0.028, 4),
        "high_dividend": (0.065, 0.15, 0.040, 4),
        "covered_call": (0.05, 0.11, 0.075, 12),
        "bond": (0.025, 0.06, 0.035, 12),
        "real_estate": (0.06, 0.20, 0.038, 4),
        "commodity": (0.05, 0.15, 0.0, 0),
    }

    def __init__(self, universe: list[dict] | None = None, seed: int = 7):
        self.meta = {u["ticker"]: u for u in (universe or [])}
        self.seed = seed

    def _profile(self, ticker):
        m = self.meta.get(ticker, {})
        return self.PROFILES.get(m.get("category"), self.PROFILES.get(m.get("asset_class"), self.PROFILES["equity"]))

    def _rng(self, ticker):
        return np.random.default_rng(zlib.crc32(ticker.encode()) + self.seed)

    def _generate(self, ticker, start, end):
        mu, vol, yld, freq = self._profile(ticker)
        idx = pd.bdate_range(start, end or pd.Timestamp.today().normalize())
        rng = self._rng(ticker)
        # one shared "market" factor so assets are correlated
        mkt = np.random.default_rng(self.seed).standard_normal(len(idx))
        idio = rng.standard_normal(len(idx))
        dt = 1 / 252
        z = 0.7 * mkt + np.sqrt(1 - 0.49) * idio
        price_ret = (mu - yld - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
        close = 50 * np.exp(np.cumsum(price_ret))
        divs = pd.Series(dtype=float)
        if freq:
            step = max(1, 252 // freq)
            pay_pos = np.arange(step // 2, len(idx), step)
            growth = 1.0 + 0.05 * (np.arange(len(pay_pos)) / freq)
            amounts = close[pay_pos] * yld / freq * growth / growth.mean()
            divs = pd.Series(amounts, index=idx[pay_pos])
        close_s = pd.Series(close, index=idx)
        tr = close_s.pct_change().fillna(0) + (divs.reindex(idx).fillna(0) / close_s.shift(1)).fillna(0)
        adj = (1 + tr).cumprod()
        adj = adj / adj.iloc[-1] * close_s.iloc[-1]
        vol_sh = rng.uniform(2e5, 5e6) * np.exp(0.3 * rng.standard_normal(len(idx)))
        prices = pd.DataFrame({"close": close_s, "adj_close": adj, "volume": vol_sh}, index=idx)
        return prices, divs.rename(ticker)

    def fetch_prices(self, ticker, start, end=None):
        return self._generate(ticker, start, end)[0]

    def fetch_dividends(self, ticker, start=None, end=None):
        return self._generate(ticker, start or "2010-01-01", end)[1].loc[start:end]


def build_provider(cfg: dict):
    name = cfg.get("data", {}).get("provider", "yfinance")
    if name == "synthetic":
        return SyntheticProvider(cfg.get("universe", []))
    if name == "yfinance":
        return YFinanceProvider()
    raise ValueError(f"Unknown data provider: {name}")


# --------------------------------------------------------------------------- service
class MarketData:
    def __init__(self, provider, db: Database | None = None, ffill_limit: int = 5, validation: dict | None = None):
        self.provider = provider
        self.db = db
        self.ffill_limit = ffill_limit
        self.validation = validation or {}

    def _cached(self, ticker, start, end) -> pd.DataFrame | None:
        if self.db is None:
            return None
        df = self.db.load_prices(ticker, start, end)
        if df.empty:
            return None
        want_start = pd.Timestamp(start)
        want_end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        fresh_end = df.index[-1] >= want_end - pd.tseries.offsets.BDay(2)
        covers_start = df.index[0] <= want_start + pd.Timedelta(days=10)
        return df if fresh_end and covers_start else None

    def get_prices(self, tickers, start, end=None, refresh: bool = False) -> PriceHistory:
        frames: dict[str, pd.DataFrame] = {}
        for t in tickers:
            df = None if refresh else self._cached(t, start, end)
            if df is None:
                try:
                    df = self.provider.fetch_prices(t, start, end)
                except Exception as exc:  # network / provider errors -> skip ticker
                    print(f"[market_data] failed to fetch {t}: {exc}", file=sys.stderr)
                    continue
                if self.db is not None and not df.empty:
                    self.db.save_prices(t, df)
            if df is not None and not df.empty:
                frames[t] = df
        if not frames:
            empty = pd.DataFrame()
            return PriceHistory(empty, empty, empty, ValidationReport([{"ticker": "*", "check": "empty",
                                                                        "severity": "error", "detail": "no data"}]))
        wide = {c: pd.DataFrame({t: f[c] for t, f in frames.items()}) for c in PRICE_COLS}
        report = validate_prices(
            wide["adj_close"],
            max_daily_move=self.validation.get("max_daily_move", 0.25),
            min_history_days=self.validation.get("min_history_days", 252),
        )
        close = clean_prices(wide["close"], self.ffill_limit)
        adj = clean_prices(wide["adj_close"], self.ffill_limit)
        keep = [t for t in close.columns if t in adj.columns and not report.errors_for(t)]
        return PriceHistory(close[keep], adj[keep], wide["volume"].reindex(close.index)[keep].fillna(0), report)


def build_market_data(cfg: dict, db: Database | None = None) -> MarketData:
    dcfg = cfg.get("data", {})
    return MarketData(build_provider(cfg), db, dcfg.get("ffill_limit", 5), dcfg.get("validation"))
