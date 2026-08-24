"""
Tests for riskparity.optimization.cvar_parity.

CVaR-parity's contribution-equality tolerance is deliberately looser
than vol-ERC's (tests/test_erc.py uses 1e-6): the scenario-based
Component CVaR estimator (risk/contributions.py::cvar_risk_contributions)
is noisier than the closed-form volatility case -- it depends on a
finite, discrete tail set of scenarios rather than a smooth quadratic
form -- so demanding vol-ERC-level precision here would either never be
met or would require artificially inflating max_iter/learning_rate
until a particular random seed happens to pass. 0.02 (2 percentage
points of risk budget) is used instead, chosen empirically from runs
with a few thousand scenarios rather than picked to make an already-
passing test look tighter than it is.
"""

import numpy as np
import pandas as pd
import pytest

from riskparity.optimization.cvar_parity import solve_cvar_parity
from riskparity.optimization.erc import solve_erc
from riskparity.risk.contributions import cvar_risk_contributions

CONFIDENCE = 0.975
CONTRIBUTION_TOL = 0.02  # see module docstring


def _synthetic_returns(n_scenarios, n_assets, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0003, 0.01, size=(n_scenarios, n_assets)))


def test_cvar_parity_weights_sum_to_one_and_long_only():
    asset_returns = _synthetic_returns(3000, 4, seed=0)
    weights, _ = solve_cvar_parity(asset_returns, CONFIDENCE)

    assert weights.sum() == pytest.approx(1.0, rel=1e-8)
    assert np.all(weights >= 0.0)


def test_cvar_parity_converges():
    """Convergence is not guaranteed for this non-convex problem
    (see the module docstring in optimization/cvar_parity.py), so this
    only checks that the dispersion objective substantially improves
    over the run -- not that it hits OptimizationParams.tol exactly, and
    not that every single step is non-increasing (a strict monotonicity
    check would be too strong a claim for a noisy, non-convex cyclical
    update -- if it were ever violated badly, that's a finding to
    surface, not paper over with a weaker synthetic test).
    """
    asset_returns = _synthetic_returns(3000, 4, seed=1)
    _, history = solve_cvar_parity(asset_returns, CONFIDENCE)

    assert len(history) > 1
    assert all(np.isfinite(history))
    assert history[-1] < history[0] * 0.1


def test_cvar_parity_contributions_equal_at_optimum():
    """At the returned weights, Component CVaR contributions should be
    close to the (default, equal) risk budgets -- within
    CONTRIBUTION_TOL, not exactly, per the module docstring's rationale.
    """
    asset_returns = _synthetic_returns(4000, 4, seed=2)
    weights, _ = solve_cvar_parity(asset_returns, CONFIDENCE)

    result = cvar_risk_contributions(weights, asset_returns.to_numpy().T, CONFIDENCE)
    target = np.full(4, 0.25)

    assert np.max(np.abs(result.pct_contribution - target)) < CONTRIBUTION_TOL


def test_cvar_parity_honors_unequal_risk_budgets():
    """A 75/25 budget on a 2-asset universe should be reflected in the
    achieved Component CVaR contributions, not just in the raw weights
    -- mirrors tests/test_erc.py's equivalent check for vol-ERC.
    """
    asset_returns = _synthetic_returns(4000, 2, seed=3)
    budgets = np.array([0.75, 0.25])
    weights, _ = solve_cvar_parity(asset_returns, CONFIDENCE, risk_budgets=budgets)

    result = cvar_risk_contributions(weights, asset_returns.to_numpy().T, CONFIDENCE)

    assert np.max(np.abs(result.pct_contribution - budgets)) < CONTRIBUTION_TOL


def test_cvar_parity_rejects_invalid_risk_budgets():
    asset_returns = _synthetic_returns(500, 2, seed=4)
    with pytest.raises(ValueError):
        solve_cvar_parity(asset_returns, CONFIDENCE, risk_budgets=np.array([0.6, 0.6]))


def test_cvar_parity_differs_from_vol_erc_on_skewed_universe():
    """The central empirical claim of the project: given two assets with
    matched volatility but one carrying a strong negative skew (fat left
    tail), vol-ERC -- blind to anything beyond variance -- allocates
    them equally, while CVaR-parity allocates LESS to the skewed asset,
    because its Component CVaR contribution is higher than its
    volatility contribution at equal weight.
    """
    rng = np.random.default_rng(42)
    n = 6000

    symmetric = rng.normal(0.0005, 0.01, size=n)
    normal_part = rng.normal(0.0013, 0.0085, size=n)
    jump_mask = rng.random(n) < 0.03
    jumps = np.where(jump_mask, rng.normal(-0.04, 0.01, size=n), 0.0)
    skewed_raw = normal_part + jumps
    # Rescale to match `symmetric`'s std exactly, so any weight
    # difference below is attributable to skew, not a residual vol gap.
    skewed = (skewed_raw - skewed_raw.mean()) / skewed_raw.std() * symmetric.std() + symmetric.mean()

    asset_returns = pd.DataFrame({"symmetric": symmetric, "skewed": skewed})
    cov = np.cov(np.vstack([symmetric, skewed]))

    erc_weights = solve_erc(cov)
    cvar_weights, _ = solve_cvar_parity(asset_returns, CONFIDENCE)

    skewed_idx = 1
    assert cvar_weights[skewed_idx] < erc_weights[skewed_idx] - 0.05

