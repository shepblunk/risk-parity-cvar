"""
Tests for riskparity.risk.contributions.

These are the tests that matter most for this project: they verify the
mathematical invariant (Euler decomposition) that everything else is
built on, not just "does the code run".
"""

import numpy as np
import pytest

from riskparity.risk.contributions import volatility_risk_contributions


def test_volatility_contributions_sum_to_total_risk():
    """Euler's identity: sum_i RC_i == sigma_p(w), for ANY weights."""
    rng = np.random.default_rng(0)
    n = 5
    a = rng.normal(size=(n, n))
    cov = a @ a.T + np.eye(n) * 0.01  # random PSD covariance
    weights = rng.dirichlet(np.ones(n))  # random weights summing to 1

    result = volatility_risk_contributions(weights, cov)

    assert result.contribution.sum() == pytest.approx(result.total_risk, rel=1e-8)
    assert result.pct_contribution.sum() == pytest.approx(1.0, rel=1e-8)


def test_volatility_contributions_two_uncorrelated_assets_known_case():
    """Closed-form sanity check: for 2 uncorrelated assets, at the ERC
    solution weights must be inversely proportional to volatility
    (w_i proportional to 1/sigma_i). Here we don't solve ERC yet (that's
    optimization/erc.py) -- we just check that AT that known-analytical
    weight vector, contributions are indeed equal.
    """
    sigma1, sigma2 = 0.10, 0.20
    cov = np.array([[sigma1**2, 0.0], [0.0, sigma2**2]])

    # ERC closed form for 2 uncorrelated assets: w_i ~ 1/sigma_i
    raw = np.array([1 / sigma1, 1 / sigma2])
    weights = raw / raw.sum()

    result = volatility_risk_contributions(weights, cov)

    assert result.contribution[0] == pytest.approx(result.contribution[1], rel=1e-6)


def test_volatility_contributions_concentrated_portfolio():
    """A 100%-in-one-asset portfolio should have all risk contribution
    on that asset and zero elsewhere -- a trivial but useful edge case.
    """
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    weights = np.array([1.0, 0.0])

    result = volatility_risk_contributions(weights, cov)

    assert result.contribution[1] == pytest.approx(0.0, abs=1e-10)
    assert result.total_risk == pytest.approx(np.sqrt(cov[0, 0]), rel=1e-8)


# TODO once cvar_risk_contributions is implemented:
# - test that sum_i RC_i ~= CVaR_alpha(portfolio) within a documented tolerance
#   (the scenario estimator is NOT exact the way the vol one is -- pick a
#   tolerance and justify it, don't silently loosen `approx` until it passes)
# - test on a known fat-tailed synthetic distribution (e.g. Student-t
#   simulated returns) that CVaR contributions differ meaningfully from
#   volatility contributions for a skewed asset
