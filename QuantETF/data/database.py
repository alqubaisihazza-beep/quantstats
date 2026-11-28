"""SQLite persistence for prices, dividends, holdings and trades."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL, date TEXT NOT NULL,
    close REAL, adj_close REAL, volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS dividends (
    ticker TEXT NOT NULL, date TEXT NOT NULL, amount REAL NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS holdings (
    account TEXT NOT NULL, ticker TEXT NOT NULL,
    shares REAL NOT NULL, cost_basis REAL NOT NULL DEFAULT 0,
    updated_at TEXT,
    PRIMARY KEY (account, ticker)
);
CREATE TABLE IF NOT EXISTS cash (
    account TEXT PRIMARY KEY, amount REAL NOT NULL, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account TEXT, ticker TEXT, side TEXT, shares REAL, price REAL,
    status TEXT, broker TEXT, created_at TEXT, filled_at TEXT, note TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------------------------------------------------------------- prices
    def save_prices(self, ticker: str, df: pd.DataFrame) -> int:
        """df: DatetimeIndex with columns close, adj_close, volume."""
        if df is None or df.empty:
            return 0
        rows = [
            (ticker, idx.strftime("%Y-%m-%d"), _f(r.get("close")), _f(r.get("adj_close")), _f(r.get("volume")))
            for idx, r in df.iterrows()
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO prices (ticker, date, close, adj_close, volume) VALUES (?,?,?,?,?)", rows
        )
        self.conn.commit()
        return len(rows)

    def load_prices(self, ticker: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        q, params = "SELECT date, close, adj_close, volume FROM prices WHERE ticker = ?", [ticker]
        if start:
            q, params = q + " AND date >= ?", params + [str(pd.Timestamp(start).date())]
        if end:
            q, params = q + " AND date <= ?", params + [str(pd.Timestamp(end).date())]
        df = pd.read_sql_query(q + " ORDER BY date", self.conn, params=params)
        if df.empty:
            return pd.DataFrame(columns=["close", "adj_close", "volume"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").rename_axis(None)

    # ------------------------------------------------------------- dividends
    def save_dividends(self, ticker: str, series: pd.Series) -> int:
        if series is None or series.empty:
            return 0
        rows = [(ticker, pd.Timestamp(i).strftime("%Y-%m-%d"), float(v)) for i, v in series.items()]
        self.conn.executemany("INSERT OR REPLACE INTO dividends (ticker, date, amount) VALUES (?,?,?)", rows)
        self.conn.commit()
        return len(rows)

    def load_dividends(self, ticker: str, start: str | None = None, end: str | None = None) -> pd.Series:
        q, params = "SELECT date, amount FROM dividends WHERE ticker = ?", [ticker]
        if start:
            q, params = q + " AND date >= ?", params + [str(pd.Timestamp(start).date())]
        if end:
            q, params = q + " AND date <= ?", params + [str(pd.Timestamp(end).date())]
        df = pd.read_sql_query(q + " ORDER BY date", self.conn, params=params)
        if df.empty:
            return pd.Series(dtype=float, name=ticker)
        return pd.Series(df["amount"].values, index=pd.to_datetime(df["date"]), name=ticker)

    # -------------------------------------------------------------- holdings
    def get_holdings(self, account: str = "manual") -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT ticker, shares, cost_basis FROM holdings WHERE account = ? AND shares != 0 ORDER BY ticker",
            self.conn,
            params=[account],
        )
        return df.set_index("ticker")

    def set_holding(self, account: str, ticker: str, shares: float, cost_basis: float = 0.0) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO holdings (account, ticker, shares, cost_basis, updated_at) VALUES (?,?,?,?,?)",
            (account, ticker, float(shares), float(cost_basis), _now()),
        )
        self.conn.commit()

    def apply_fill(self, account: str, ticker: str, side: str, shares: float, price: float) -> None:
        """Update holdings (average cost) and cash for a filled trade."""
        h = self.get_holdings(account)
        cur_sh = float(h.loc[ticker, "shares"]) if ticker in h.index else 0.0
        cur_cb = float(h.loc[ticker, "cost_basis"]) if ticker in h.index else 0.0
        if side.upper() == "BUY":
            new_sh = cur_sh + shares
            new_cb = cur_cb + shares * price
            cash_delta = -shares * price
        else:
            if shares > cur_sh + 1e-9:
                raise ValueError(f"Cannot sell {shares} {ticker}; only {cur_sh} held")
            new_sh = cur_sh - shares
            new_cb = cur_cb * (new_sh / cur_sh) if cur_sh else 0.0
            cash_delta = shares * price
        self.set_holding(account, ticker, new_sh, new_cb)
        self.set_cash(account, self.get_cash(account) + cash_delta)

    def get_cash(self, account: str = "manual") -> float:
        row = self.conn.execute("SELECT amount FROM cash WHERE account = ?", (account,)).fetchone()
        return float(row[0]) if row else 0.0

    def set_cash(self, account: str, amount: float) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cash (account, amount, updated_at) VALUES (?,?,?)", (account, float(amount), _now())
        )
        self.conn.commit()

    # ---------------------------------------------------------------- trades
    def record_trade(
        self, account: str, ticker: str, side: str, shares: float, price: float,
        status: str = "pending", broker: str = "manual", note: str = "",
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO trades (account, ticker, side, shares, price, status, broker, created_at, note) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (account, ticker, side.upper(), float(shares), float(price), status, broker, _now(), note),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_trade(self, trade_id: int, **fields) -> None:
        allowed = {"status", "price", "shares", "filled_at", "note"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return
        cols = ", ".join(f"{k} = ?" for k in sets)
        self.conn.execute(f"UPDATE trades SET {cols} WHERE id = ?", (*sets.values(), trade_id))
        self.conn.commit()

    def get_trades(self, account: str | None = None, status: str | None = None) -> pd.DataFrame:
        q, params = "SELECT * FROM trades WHERE 1=1", []
        if account:
            q, params = q + " AND account = ?", params + [account]
        if status:
            q, params = q + " AND status = ?", params + [status]
        return pd.read_sql_query(q + " ORDER BY id", self.conn, params=params)


def _f(v):
    try:
        return None if v is None or pd.isna(v) else float(v)
    except (TypeError, ValueError):
        return None
