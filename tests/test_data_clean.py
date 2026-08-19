"""
Tests for riskparity.data.clean.
"""

import numpy as np
import pandas as pd
import pytest

from riskparity.data.clean import clean_prices, compute_log_returns


def test_clean_prices_forward_fills_short_gap():
    """A single-day NaN gap (e.g. a data-vendor hiccup) should be
    forward-filled rather than causing the row to be dropped.
    """
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    raw = pd.DataFrame({"A": [10.0, np.nan, 12.0, 13.0, 14.0], "B": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=dates)

    cleaned = clean_prices(raw)

    assert len(cleaned) == 5
    assert cleaned.loc[dates[1], "A"] == pytest.approx(10.0)


def test_clean_prices_drops_pre_inception_rows():
    """A ticker with a long leading NaN run (late inception) should have
    those rows dropped -- forward-fill cannot invent a price that didn't
    exist yet, and the row must not survive with a fabricated value.
    """
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    a = pd.Series(np.linspace(10, 19, 10), index=dates)
    b = pd.Series(np.linspace(1, 10, 10), index=dates)
    b.iloc[:7] = np.nan  # B only starts trading on day 8
    raw = pd.DataFrame({"A": a, "B": b})

    cleaned = clean_prices(raw)

    assert len(cleaned) == 3
    assert cleaned.index.min() == dates[7]
    assert cleaned.isna().sum().sum() == 0


def test_clean_prices_all_nan_raises():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    raw = pd.DataFrame({"A": [np.nan, np.nan, np.nan]}, index=dates)

    with pytest.raises(ValueError):
        clean_prices(raw)


def test_compute_log_returns_matches_manual_diff():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"A": [100.0, 105.0, 103.0, 110.0]}, index=dates)

    returns = compute_log_returns(prices)

    expected = np.log(prices).diff().dropna()
    assert len(returns) == len(prices) - 1
    pd.testing.assert_frame_equal(returns, expected)


def test_compute_log_returns_round_trip_on_flat_price():
    """A constant price series has zero log return every day."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"A": [50.0] * 5}, index=dates)

    returns = compute_log_returns(prices)

    assert returns["A"].abs().max() == pytest.approx(0.0, abs=1e-12)