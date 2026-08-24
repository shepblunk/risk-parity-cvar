"""
CVaR-based risk parity ("tail risk parity") solver.

This is the hard, differentiating part of the project: unlike
volatility, CVaR has no closed-form quadratic structure in w, so the
equal-risk-budget problem is generally NON-convex when framed directly
on Component CVaR computed from historical scenarios.

Suggested approach (documented so you choose deliberately, not by
accident)
----------------------------------------------------------------------
1. Baseline / sanity check: Rockafellar-Uryasev (2000) show that CVaR
   itself *can* be written as a convex (LP-representable) function of w
   via an auxiliary variable, which is enough to do CVaR
   *minimization*. But equalizing CVaR *contributions* is a different,
   harder problem than minimizing total CVaR -- don't conflate the two
   in the README.
2. Practical solver: sequential/cyclical coordinate approach analogous
   to Roncalli's ERC algorithm, but recomputing Component CVaR (scenario
   based, see risk/contributions.py) at each iteration and nudging
   weights toward equalizing RC_i. This is not guaranteed globally
   convergent -- track objective (dispersion of RC_i) across iterations
   and report it, rather than silently assuming convergence.
3. Alternative: successive convex approximation (SCA), linearizing the
   scenario-based CVaR contribution around the current iterate and
   solving a small QP/LP each step.

Start with (2): it reuses `cvar_risk_contributions` directly and is the
most defensible to explain in an interview ("I used a cyclical update
analogous to the vol-ERC algorithm, and empirically validated
convergence -- here's the convergence plot").

Implementation notes
---------------------
- `asset_returns` follows the same (n_scenarios, n_assets) convention as
  everywhere else in the codebase (e.g. `clean_prices`/
  `compute_log_returns`: dates as rows, tickers as columns). It is
  transposed once, internally, to the (n_assets, n_scenarios) shape
  `cvar_risk_contributions` expects -- callers should not pre-transpose.
- The per-iteration step multiplies each weight by a factor pulling its
  Component CVaR contribution toward its budgeted target
  `b_i * CVaR_alpha(w)`, then re-floors at a tiny positive value and
  renormalizes to sum to 1 (mirrors the `w >= 1e-12` floor in
  `optimization/erc.py`). The step size is
  `OptimizationParams.cvar_parity_learning_rate` rather than a hard-coded
  constant.
- Convergence is checked against `OptimizationParams.tol`, the same
  field `solve_erc` uses -- but nothing here assumes that tolerance is
  actually reachable. The scenario-based CVaR estimator is noisier than
  the closed-form volatility case, so it would not be surprising for
  this loop to run to `max_iter` without dispersion dropping below
  `tol`; `convergence_history` is returned precisely so that can be
  inspected and reported honestly (see `tests/test_cvar_parity.py` and
  `docs/notes.tex`), not assumed away.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from riskparity.config import OptimizationParams
from riskparity.risk.contributions import cvar_risk_contributions

logger = logging.getLogger(__name__)


def solve_cvar_parity(
    asset_returns: pd.DataFrame,
    confidence: float,
    risk_budgets: np.ndarray | None = None,
    params: OptimizationParams = OptimizationParams(),
) -> tuple[np.ndarray, list[float]]:
    """Return (weights, convergence_history) where convergence_history
    is the dispersion-of-contributions objective at each iteration.
    """
    asset_returns = np.asarray(asset_returns, dtype=float)  # (n_scenarios, n_assets)
    n_assets = asset_returns.shape[1]
    scenario_returns = asset_returns.T  # (n_assets, n_scenarios), cvar_risk_contributions's convention

    if risk_budgets is None:
        risk_budgets = np.full(n_assets, 1.0 / n_assets)
    else:
        risk_budgets = np.asarray(risk_budgets, dtype=float)
        if risk_budgets.shape != (n_assets,):
            raise ValueError(f"risk_budgets must have shape ({n_assets},), got {risk_budgets.shape}")
        if np.any(risk_budgets <= 0) or not np.isclose(risk_budgets.sum(), 1.0):
            raise ValueError("risk_budgets must be strictly positive and sum to 1")

    weights = np.full(n_assets, 1.0 / n_assets)
    convergence_history: list[float] = []

    for _ in range(params.max_iter):
        result = cvar_risk_contributions(weights, scenario_returns, confidence)
        dispersion = float(np.sum((result.pct_contribution - risk_budgets) ** 2))
        convergence_history.append(dispersion)

        if dispersion < params.tol:
            break
        
        adjustment_factor = 1-params.cvar_parity_learning_rate*((result.pct_contribution/risk_budgets)-1)
        weights = weights * adjustment_factor
        weights = np.maximum(weights, 1e-8)
        weights = weights / weights.sum()

    if convergence_history[-1] >= params.tol:
        logger.warning(
            "solve_cvar_parity: did not converge after %d iterations "
            "(final dispersion=%.2e, exceeding OptimizationParams.tol=%.2e) "
            "-- CVaR-parity is not guaranteed to converge; inspect the "
            "returned convergence_history rather than assuming these "
            "weights are balanced",
            len(convergence_history), convergence_history[-1], params.tol,
        )

    return weights, convergence_history