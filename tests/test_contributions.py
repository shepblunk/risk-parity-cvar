"""
Tests for riskparity.risk.contributions.

These are the tests that matter most for this project: they verify the
mathematical invariant (Euler decomposition) that everything else is
built on, not just "does the code run".
"""

import numpy as np
import pytest

from riskparity.risk.contributions import cvar_risk_contributions, volatility_risk_contributions

CONFIDENCE = 0.975


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


def test_cvar_contributions_sum_to_total_risk():
    """Euler's identity for CVaR: sum_i RC_i == CVaR_alpha(w). Unlike the
    module docstring's general "approximately exact" caveat for the
    scenario estimator, this is exactly exact here because marginal and
    total_risk are computed from the SAME tail mask -- see the reasoning
    in cvar_risk_contributions's own docstring.
    """
    rng = np.random.default_rng(0)
    n_assets, n_scenarios = 5, 2000
    asset_returns = rng.normal(0.0, 0.01, size=(n_assets, n_scenarios))
    weights = rng.dirichlet(np.ones(n_assets))

    result = cvar_risk_contributions(weights, asset_returns, CONFIDENCE)

    assert result.contribution.sum() == pytest.approx(result.total_risk, rel=1e-8)
    assert result.pct_contribution.sum() == pytest.approx(1.0, rel=1e-8)


def test_cvar_contributions_concentrated_portfolio():
    """A 100%-in-one-asset portfolio has all CVaR contribution on that
    asset and zero elsewhere -- same edge case already checked for
    volatility_risk_contributions above.
    """
    rng = np.random.default_rng(1)
    asset_returns = rng.normal(0.0, 0.01, size=(2, 1000))
    weights = np.array([1.0, 0.0])

    result = cvar_risk_contributions(weights, asset_returns, CONFIDENCE)

    assert result.contribution[1] == pytest.approx(0.0, abs=1e-12)
    assert result.total_risk == pytest.approx(result.contribution[0], rel=1e-8)


def test_cvar_contributions_differ_from_volatility_on_skewed_asset():
    """Central empirical claim of the project: an asset with a fat left
    tail (strong negative skew) but similar overall volatility to a
    symmetric asset should be assigned a LARGER share of risk by
    Component CVaR than by volatility-based risk contribution, since
    volatility alone cannot see the asymmetry.
    """
    rng = np.random.default_rng(2)
    n = 8000

    symmetric = rng.normal(0.0005, 0.010, size=n)
    # Skewed asset: same rough vol scale as `symmetric`, but with a fat
    # left tail from occasional large negative jumps (same construction
    # as tests/test_tail_risk.py's negative-skew test).
    normal_part = rng.normal(0.0013, 0.0085, size=n)
    jump_mask = rng.random(n) < 0.03
    jumps = np.where(jump_mask, rng.normal(-0.04, 0.01, size=n), 0.0)
    skewed = normal_part + jumps

    asset_returns = np.vstack([symmetric, skewed])
    weights = np.array([0.5, 0.5])

    cov = np.cov(asset_returns)
    vol_result = volatility_risk_contributions(weights, cov)
    cvar_result = cvar_risk_contributions(weights, asset_returns, CONFIDENCE)

    skewed_pct_rc_vol = vol_result.pct_contribution[1]
    skewed_pct_rc_cvar = cvar_result.pct_contribution[1]

    assert skewed_pct_rc_cvar > skewed_pct_rc_vol + 0.05
