"""
Tests for riskparity.risk.volatility.
"""

import numpy as np
import pandas as pd
import pytest

from riskparity.risk.volatility import (
    TRADING_DAYS_PER_YEAR,
    portfolio_volatility,
    sample_covariance,
    shrunk_covariance,
)


def _synthetic_returns(n_days=300, n_assets=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    data = rng.normal(scale=0.01, size=(n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=[f"A{i}" for i in range(n_assets)])


def test_sample_covariance_is_annualized():
    returns = _synthetic_returns()
    cov = sample_covariance(returns, lookback_days=252)

    manual_daily_cov = returns.tail(252).cov()
    pd.testing.assert_frame_equal(cov, manual_daily_cov * TRADING_DAYS_PER_YEAR)


def test_sample_covariance_uses_only_the_lookback_window():
    returns = _synthetic_returns(n_days=300)
    cov_short = sample_covariance(returns, lookback_days=50)

    manual = returns.tail(50).cov() * TRADING_DAYS_PER_YEAR
    pd.testing.assert_frame_equal(cov_short, manual)


def test_sample_covariance_too_few_observations_raises():
    returns = _synthetic_returns(n_days=1)
    with pytest.raises(ValueError):
        sample_covariance(returns, lookback_days=252)


def test_shrunk_covariance_zero_shrinkage_equals_sample():
    returns = _synthetic_returns()
    sample = sample_covariance(returns, lookback_days=252)
    shrunk = shrunk_covariance(returns, lookback_days=252, shrinkage=0.0)

    pd.testing.assert_frame_equal(shrunk, sample)


def test_shrunk_covariance_full_shrinkage_is_diagonal():
    returns = _synthetic_returns()
    shrunk = shrunk_covariance(returns, lookback_days=252, shrinkage=1.0)

    off_diagonal = shrunk.values - np.diag(np.diag(shrunk.values))
    assert np.allclose(off_diagonal, 0.0)


def test_shrunk_covariance_preserves_variances_regardless_of_intensity():
    """The diagonal (variances) should be untouched by shrinkage -- only
    the off-diagonal covariances/correlations are pulled toward 0.
    """
    returns = _synthetic_returns()
    sample = sample_covariance(returns, lookback_days=252)
    shrunk = shrunk_covariance(returns, lookback_days=252, shrinkage=0.5)

    np.testing.assert_allclose(np.diag(shrunk.values), np.diag(sample.values))


def test_shrunk_covariance_invalid_intensity_raises():
    returns = _synthetic_returns()
    with pytest.raises(ValueError):
        shrunk_covariance(returns, lookback_days=252, shrinkage=1.5)


def test_portfolio_volatility_matches_direct_formula():
    rng = np.random.default_rng(1)
    n = 4
    a = rng.normal(size=(n, n))
    cov = a @ a.T + np.eye(n) * 0.01
    weights = rng.dirichlet(np.ones(n))

    result = portfolio_volatility(weights, cov)
    expected = np.sqrt(weights @ cov @ weights)

    assert result == pytest.approx(expected, rel=1e-10)


def test_portfolio_volatility_concentrated_matches_single_asset_vol():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    weights = np.array([1.0, 0.0])

    assert portfolio_volatility(weights, cov) == pytest.approx(np.sqrt(cov[0, 0]))