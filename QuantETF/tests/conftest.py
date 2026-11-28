import numpy as np
import pandas as pd
import pytest

from data import load_config
from data.database import Database
from data.dividend_data import DividendData
from data.market_data import MarketData, SyntheticProvider
from engines.etf_screener import build_universe_frame


@pytest.fixture(scope="session")
def cfg():
    c = load_config()
    c["data"]["provider"] = "synthetic"
    return c


@pytest.fixture(scope="session")
def universe(cfg):
    return build_universe_frame(cfg["universe"])


@pytest.fixture(scope="session")
def provider(cfg):
    return SyntheticProvider(cfg["universe"])


@pytest.fixture(scope="session")
def prices(provider, universe):
    md = MarketData(provider, Database(":memory:"))
    return md.get_prices(list(universe.index), "2016-01-01", "2024-12-31")


@pytest.fixture(scope="session")
def dividends(provider, universe):
    return DividendData(provider, Database(":memory:")).get_dividends(list(universe.index), "2016-01-01", "2024-12-31")


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()


@pytest.fixture
def toy_returns():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=750)
    data = {
        "LOW": rng.normal(0.0002, 0.003, len(idx)),
        "MID": rng.normal(0.0004, 0.010, len(idx)),
        "HIGH": rng.normal(0.0006, 0.020, len(idx)),
    }
    return pd.DataFrame(data, index=idx)
