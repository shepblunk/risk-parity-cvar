"""
Clean raw price data into an analysis-ready return series.

TODO
----
- Align all tickers on a common trading calendar (inner join on dates,
  or explicit handling of holidays per exchange).
- Compute log returns (preferred for risk work: additive over time,
  better behaved for CVaR estimation) via `compute_log_returns`.
- Flag / document any data quality issues (long NaN gaps, stock
  splits not adjusted, etc.) -- this is exactly the kind of "boring but
  essential" step a risk desk cares about, so it's worth a short note
  in the README about what you checked.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Forward-fill across short gaps only (e.g. a single-day halt/holiday
# mismatch across exchanges); a longer NaN run means the ticker wasn't
# trading yet (pre-inception) or data is genuinely missing, and should be
# handled by the dropna alignment step below rather than papered over
# with a stale price.
MAX_FFILL_DAYS = 5


def clean_prices(raw_prices: pd.DataFrame) -> pd.DataFrame:
    """Align tickers, forward-fill small gaps, drop rows that are still
    incomplete afterwards. Returns a clean price DataFrame.
    """
    prices = raw_prices.sort_index().ffill(limit=MAX_FFILL_DAYS)

    before = len(prices)
    prices = prices.dropna(how="any")
    after = len(prices)
    if after < before:
        logger.info(
            "clean_prices: dropped %d/%d rows still containing NaNs after "
            "forward-fill (limit=%d days) -- likely pre-inception rows for "
            "a late-starting ticker",
            before - after, before, MAX_FFILL_DAYS,
        )
    if prices.empty:
        raise ValueError("clean_prices: no rows remain after alignment/cleaning")

    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns: r_t = ln(P_t / P_{t-1})."""
    return np.log(prices).diff().dropna(how="all")
