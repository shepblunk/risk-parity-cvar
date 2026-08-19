"""
Download adjusted-close price data for the project universe.

Design intent
-------------
This module should be the ONLY place that talks to an external data
source (yfinance). Everything downstream works on a plain pandas
DataFrame of prices, so the data source can be swapped later (e.g. for
a paid vendor) without touching any risk/optimization code.

TODO
----
- Implement `download_prices` using `yfinance.download`.
- Cache raw downloads to `data/raw/prices_raw.parquet` so re-runs are
  fast and offline-friendly.
- Handle tickers with different inception dates gracefully (log which
  tickers were dropped/truncated rather than silently ffilling long gaps).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from riskparity.config import Universe

logger = logging.getLogger(__name__)


def download_prices(universe: Universe, cache_path: str | None = "data/raw/prices_raw.parquet") -> pd.DataFrame:
    """Download daily adjusted close prices for ``universe.tickers``.

    Parameters
    ----------
    universe : Universe
        Defines tickers and date range.
    cache_path : str, optional
        If given and the file exists, load from cache instead of
        re-downloading. If given and the file does not exist, download
        then write to this path.

    Returns
    -------
    pd.DataFrame
        Index = trading dates, columns = tickers, values = adjusted close.
    """
    cache_file = Path(cache_path) if cache_path is not None else None
    if cache_file is not None and cache_file.exists():
        return pd.read_parquet(cache_file)

    raw = yf.download(
        list(universe.tickers),
        start=universe.start_date,
        end=universe.end_date,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for tickers {universe.tickers}")

    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=universe.tickers[0])
    prices = prices.reindex(columns=list(universe.tickers))

    dropped = [t for t in universe.tickers if prices[t].isna().all()]
    if dropped:
        logger.warning("Tickers with no data returned, dropped: %s", dropped)
        prices = prices.drop(columns=dropped)

    inception = prices.apply(lambda col: col.first_valid_index())
    late_starters = inception[inception > prices.index.min()]
    if not late_starters.empty:
        logger.info("Tickers with later inception dates than the universe start: %s", late_starters.to_dict())

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_file)
    
    return prices

print(f"Download.py WORKING *******")