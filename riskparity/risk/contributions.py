"""
Euler (marginal) risk-contribution decomposition.

This is the mathematical backbone of the whole project: for any risk
measure rho(w) that is positively homogeneous of degree 1 in the
weights w (true for volatility and for CVaR of a portfolio return),
Euler's theorem gives an EXACT decomposition:

    rho(w) = sum_i  w_i * d(rho)/d(w_i)  = sum_i RC_i

where:
    MRC_i = d(rho)/d(w_i)     marginal risk contribution of asset i
    RC_i  = w_i * MRC_i       risk contribution of asset i (sums to rho(w))
    RC_i / rho(w)             percentage risk contribution (sums to 1)

Equal Risk Contribution (ERC) / risk parity asks for weights such that
RC_i is the same for every asset.

Implemented here
-----------------
- `volatility_risk_contributions`: closed form, fully implemented below
  as the reference pattern for the rest of the codebase (docstring
  style, type hints, the "contributions sum to total risk" invariant
  you should unit-test).
- `cvar_risk_contributions`: TODO -- CVaR has no closed form in terms of
  (w, Sigma) the way volatility does. Two standard approaches:
    (a) analytical/semi-parametric formula under a distributional
        assumption (see Boudt, Peterson & Carl (2013), "Component VaR
        and CVaR"), which gives MRC_i in terms of conditional moments;
    (b) Monte Carlo / scenario-based: MRC_i ~= average of asset i's
        return over the scenarios that fall in the tail beyond VaR.
        Simpler to implement and audit, noisier -- pick one, document
        the trade-off, maybe implement both and compare.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from riskparity.risk.tail_risk import historical_var


@dataclass
class RiskContributions:
    """Container for a risk decomposition result."""

    total_risk: float
    marginal: np.ndarray  # MRC_i
    contribution: np.ndarray  # RC_i, sums to total_risk
    pct_contribution: np.ndarray  # RC_i / total_risk, sums to 1


def volatility_risk_contributions(weights: np.ndarray, cov: np.ndarray) -> RiskContributions:
    """Exact Euler decomposition of portfolio volatility.

    sigma_p(w) = sqrt(w' Sigma w)
    MRC_i      = (Sigma w)_i / sigma_p(w)
    RC_i       = w_i * MRC_i

    This is fully implemented (not a stub) -- use it as the template
    for `cvar_risk_contributions` below: same signature shape, same
    invariant (contributions sum to total risk), same return type.
    """
    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)

    portfolio_variance = weights @ cov @ weights
    sigma_p = np.sqrt(portfolio_variance)

    marginal = (cov @ weights) / sigma_p
    contribution = weights * marginal
    pct_contribution = contribution / sigma_p

    return RiskContributions(
        total_risk=float(sigma_p),
        marginal=marginal,
        contribution=contribution,
        pct_contribution=pct_contribution,
    )


def cvar_risk_contributions(
    weights: np.ndarray,
    asset_returns: np.ndarray,
    confidence: float,
) -> RiskContributions:
    """Euler decomposition of portfolio CVaR (Component CVaR).

    Scenario-based estimator:
      1. Compute portfolio returns R_t = w' r_t for each historical (or
         simulated) scenario t.
      2. Identify the tail set T = {t : R_t <= -VaR_alpha(R)}.
      3. MRC_i = -mean(r_{i,t} for t in T)   (avg return of asset i in
         the tail scenarios, with a sign flip since we express risk as
         a positive loss number).
      4. RC_i = w_i * MRC_i. Given this shared tail definition,
         sum_i RC_i is exactly -mean(R_t for t in T), i.e. exactly the
         scenario-based total_risk returned alongside it (not merely
         approximate) -- see tests/test_contributions.py.

    `asset_returns` is an (n_assets, n_scenarios) array of scenario
    asset returns (e.g. historical daily returns transposed). VaR is
    computed via `riskparity.risk.tail_risk.historical_var` rather than
    re-deriving the quantile here, so both modules share one definition
    of the tail threshold.
    """
    weights = np.asarray(weights, dtype=float)
    asset_returns = np.asarray(asset_returns, dtype=float)

    portfolio_returns = weights @ asset_returns  # R_t, shape (n_scenarios,)
    var_alpha = historical_var(portfolio_returns, confidence)
    tail_mask = portfolio_returns <= -var_alpha
    if not np.any(tail_mask):
        raise ValueError(
            "cvar_risk_contributions: no scenarios fall in the tail at "
            f"confidence={confidence} -- too few observations or too "
            "extreme a confidence level for this scenario set"
        )

    tail_asset_returns = asset_returns[:, tail_mask]
    marginal = -np.mean(tail_asset_returns, axis=1)
    contribution = weights * marginal
    total_risk = -np.mean(portfolio_returns[tail_mask])

    return RiskContributions(
        total_risk=float(total_risk),
        marginal=marginal,
        contribution=contribution,
        pct_contribution=contribution / total_risk,
    )