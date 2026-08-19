"""
Tests for riskparity.optimization.erc.
"""

import numpy as np
import pytest

from riskparity.optimization.erc import solve_erc
from riskparity.risk.contributions import volatility_risk_contributions


def _random_psd_cov(n, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + np.eye(n) * 0.01


def test_erc_weights_sum_to_one_and_long_only():
    cov = _random_psd_cov(5, seed=0)

    weights = solve_erc(cov)

    assert weights.sum() == pytest.approx(1.0, rel=1e-6)
    assert (weights >= 0).all()


def test_erc_contributions_are_equal_at_optimum():
    cov = _random_psd_cov(4, seed=1)

    weights = solve_erc(cov)
    result = volatility_risk_contributions(weights, cov)

    np.testing.assert_allclose(result.contribution, result.contribution[0], atol=1e-6)


def test_erc_matches_closed_form_two_uncorrelated_assets():
    """Same known case as
    tests/test_contributions.py::test_volatility_contributions_two_uncorrelated_assets_known_case,
    but here the solver itself must discover the 1/sigma_i weights
    rather than being handed them.
    """
    sigma1, sigma2 = 0.10, 0.20
    cov = np.array([[sigma1**2, 0.0], [0.0, sigma2**2]])

    weights = solve_erc(cov)

    raw = np.array([1 / sigma1, 1 / sigma2])
    expected = raw / raw.sum()
    np.testing.assert_allclose(weights, expected, atol=1e-5)


def test_erc_equal_vols_equal_correlation_gives_equal_weights():
    """Symmetric case: identical variances and identical pairwise
    correlations for every asset must give equal weights by symmetry.
    """
    n = 4
    sigma, rho = 0.15, 0.3
    cov = np.full((n, n), rho * sigma**2)
    np.fill_diagonal(cov, sigma**2)

    weights = solve_erc(cov)

    np.testing.assert_allclose(weights, np.full(n, 1.0 / n), atol=1e-5)


def test_erc_respects_unequal_risk_budgets():
    sigma1, sigma2 = 0.10, 0.20
    cov = np.array([[sigma1**2, 0.0], [0.0, sigma2**2]])
    budgets = np.array([0.75, 0.25])

    weights = solve_erc(cov, risk_budgets=budgets)
    result = volatility_risk_contributions(weights, cov)

    np.testing.assert_allclose(result.pct_contribution, budgets, atol=1e-5)


def test_erc_invalid_risk_budgets_raises():
    cov = np.eye(3) * 0.04
    with pytest.raises(ValueError):
        solve_erc(cov, risk_budgets=np.array([0.5, 0.6, 0.1]))  # does not sum to 1