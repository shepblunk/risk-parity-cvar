"""
Tests for riskparity.risk.tail_risk.

Convention (see module docstring): both VaR_alpha and CVaR_alpha are
expressed as positive loss numbers,
    VaR_alpha(R)  = -quantile(R, 1 - alpha)
    CVaR_alpha(R) = -E[R | R <= -VaR_alpha(R)]

These tests check invariants that must hold for ANY reasonable
estimator (ordering, monotonicity in confidence, a trivial no-variation
edge case), plus a closed-form comparison against the analytical
Gaussian VaR/CVaR formula for a large i.i.d. normal sample -- useful
because both `historical_cvar` and `cornish_fisher_cvar` should agree
with it (approximately) when the underlying data really is Gaussian
(zero skew/kurtosis), even though they're computed differently.
"""

import numpy as np
import pytest
from scipy import stats

from riskparity.risk.tail_risk import (
    cornish_fisher_cvar,
    historical_cvar,
    historical_var,
)

CONFIDENCE = 0.975


def _gaussian_returns(n, mu=0.0005, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    return pd_series(rng.normal(mu, sigma, size=n))


def pd_series(values):
    import pandas as pd

    return pd.Series(values)


def _analytical_gaussian_var_cvar(mu, sigma, confidence):
    """Closed-form VaR/CVaR for R ~ N(mu, sigma^2), same sign convention
    as the module under test (positive loss numbers).
    """
    z = stats.norm.ppf(1 - confidence)  # negative, e.g. ~ -1.96 at 97.5%
    var = -(mu + sigma * z)
    cvar = -mu + sigma * stats.norm.pdf(z) / (1 - confidence)
    return var, cvar


def test_cvar_is_at_least_var():
    """CVaR is the average loss IN the tail beyond VaR, so by
    construction it can never be smaller than VaR itself -- true for any
    return series with variation, regardless of distribution shape.
    """
    returns = _gaussian_returns(1000)
    var = historical_var(returns, CONFIDENCE)
    cvar = historical_cvar(returns, CONFIDENCE)
    assert cvar >= var - 1e-12


def test_var_cvar_increase_with_confidence():
    """A higher confidence level looks further into the tail, so both
    VaR and CVaR must be (weakly) larger at alpha=0.99 than at
    alpha=0.90 for the same return series.
    """
    returns = _gaussian_returns(1000)
    var_low = historical_var(returns, 0.90)
    var_high = historical_var(returns, 0.99)
    cvar_low = historical_cvar(returns, 0.90)
    cvar_high = historical_cvar(returns, 0.99)

    assert var_high >= var_low
    assert cvar_high >= cvar_low


def test_historical_var_cvar_constant_returns_edge_case():
    """A return series with zero variation (every day the same return
    r) has no real 'tail' -- VaR and CVaR should both reduce to exactly
    -r, the trivial case, analogous to the concentrated-portfolio edge
    case already used for volatility_risk_contributions.
    """
    r = -0.002
    returns = pd_series(np.full(200, r))

    assert historical_var(returns, CONFIDENCE) == pytest.approx(-r, abs=1e-10)
    assert historical_cvar(returns, CONFIDENCE) == pytest.approx(-r, abs=1e-10)


def test_historical_var_cvar_match_gaussian_closed_form():
    """For a large i.i.d. Gaussian sample, the non-parametric historical
    estimator should approximately recover the analytical Gaussian
    VaR/CVaR formula. Tolerance is loose (rel=0.15) and must stay
    documented rather than tightened until it happens to pass: with
    alpha=0.975 and n=5000, the CVaR tail averages only ~125
    observations, so sampling noise on the estimate is real, not a bug.
    """
    mu, sigma, n = 0.0005, 0.01, 5000
    returns = _gaussian_returns(n, mu=mu, sigma=sigma, seed=1)
    expected_var, expected_cvar = _analytical_gaussian_var_cvar(mu, sigma, CONFIDENCE)

    assert historical_var(returns, CONFIDENCE) == pytest.approx(expected_var, rel=0.15)
    assert historical_cvar(returns, CONFIDENCE) == pytest.approx(expected_cvar, rel=0.15)


def test_cornish_fisher_matches_gaussian_closed_form_when_no_skew_kurtosis():
    """Cornish-Fisher adjusts the Gaussian quantile for sample skew and
    excess kurtosis; for genuinely Gaussian data both are ~0, so the
    adjustment should vanish and cornish_fisher_cvar should land close
    to the same analytical Gaussian CVaR as the historical estimator
    above. Same tolerance rationale (finite-sample noise on sample
    skew/kurtosis, not just on the quantile itself).
    """
    mu, sigma, n = 0.0005, 0.01, 5000
    returns = _gaussian_returns(n, mu=mu, sigma=sigma, seed=2)
    _, expected_cvar = _analytical_gaussian_var_cvar(mu, sigma, CONFIDENCE)

    assert cornish_fisher_cvar(returns, CONFIDENCE) == pytest.approx(expected_cvar, rel=0.15)


def test_cornish_fisher_reacts_to_negative_skew():
    """The whole point of Cornish-Fisher over the plain Gaussian
    assumption: for a left-skewed (fat left tail) series, it should
    report a LARGER CVaR than the analytical symmetric-Gaussian formula
    at the same mean/vol -- this is the qualitative behavior the
    CVaR-parity story (Section on Component CVaR) depends on down the
    line.
    """
    n = 5000
    rng = np.random.default_rng(3)
    # Negatively-skewed synthetic returns: mostly small positive noise,
    # occasional larger negative jumps -- same rough vol scale as the
    # other tests, but with a fat left tail.
    normal_part = rng.normal(0.001, 0.008, size=n)
    jump_mask = rng.random(n) < 0.03
    jumps = np.where(jump_mask, rng.normal(-0.04, 0.01, size=n), 0.0)
    returns = pd_series(normal_part + jumps)

    mu, sigma = returns.mean(), returns.std()
    _, symmetric_cvar = _analytical_gaussian_var_cvar(mu, sigma, CONFIDENCE)

    assert cornish_fisher_cvar(returns, CONFIDENCE) > symmetric_cvar


def test_invalid_confidence_raises():
    """Confidence must be a probability in (0, 1); values outside that
    range are a caller error, not silently clamped -- consistent with
    the eager-validation style used elsewhere (e.g. solve_erc's
    risk_budgets checks, shrunk_covariance's shrinkage-range check).
    """
    returns = _gaussian_returns(100)
    with pytest.raises(ValueError):
        historical_var(returns, 1.5)
    with pytest.raises(ValueError):
        historical_cvar(returns, 0.0)
