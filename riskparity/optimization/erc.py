"""
Volatility-based Equal Risk Contribution (ERC) solver.

Formulation
-----------
Find w >= 0, sum(w) = 1, minimizing the dispersion of risk
contributions:

    min_w  sum_i sum_j ( RC_i(w) - RC_j(w) )^2

or equivalently (Maillard, Roncalli & Teiletche, 2010) solve the convex
proxy problem:

    min_w  0.5 * w' Sigma w - sum_i b_i * ln(w_i)     s.t. w >= 0

for equal budgets b_i = 1/N, then renormalize w to sum to 1. This proxy
is convex and has a unique solution, which is what makes vol-ERC
tractable compared to CVaR-ERC below.

Implementation choice
----------------------
Solved via the log-barrier convex reformulation with `cvxpy`, not via
SLSQP on the direct dispersion objective. `cvxpy` is picked because the
reformulation above is provably convex with a unique minimizer (up to
scale) for any risk-budget vector, so a convex solver either finds that
global optimum or reports infeasibility/failure -- it cannot silently
settle for a local optimum the way a general-purpose local solver like
SLSQP could on a hand-rolled dispersion objective. The trade-off is that
this doesn't reuse machinery that would carry over to the CVaR-parity
solver (`optimization/cvar_parity.py`), which has no equivalent convex
reformulation and needs its own cyclical update regardless of what is
chosen here.

Unequal risk budgets `b` (`risk_budgets`, defaulting to 1/N) are
supported directly: the reformulation above is convex for *any* strictly
positive budget vector summing to 1, not just equal budgets, so no extra
solver logic is needed for the general case.
"""

from __future__ import annotations

import logging

import cvxpy as cp
import numpy as np

from riskparity.config import OptimizationParams
from riskparity.risk.contributions import volatility_risk_contributions

logger = logging.getLogger(__name__)


def solve_erc(
    cov: np.ndarray,
    risk_budgets: np.ndarray | None = None,
    params: OptimizationParams = OptimizationParams(),
) -> np.ndarray:
    """Return weights w (long-only, sum to 1) with equal (or budgeted)
    volatility risk contributions.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]

    if risk_budgets is None:
        risk_budgets = np.full(n, 1.0 / n)
    else:
        risk_budgets = np.asarray(risk_budgets, dtype=float)
        if risk_budgets.shape != (n,):
            raise ValueError(f"risk_budgets must have shape ({n},), got {risk_budgets.shape}")
        if np.any(risk_budgets <= 0) or not np.isclose(risk_budgets.sum(), 1.0):
            raise ValueError("risk_budgets must be strictly positive and sum to 1")

    w = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(w, cov) - risk_budgets @ cp.log(w))
    # w >= tiny positive floor rather than w >= 0: cp.log's domain already
    # forces w > 0 implicitly, but an explicit floor keeps the solver away
    # from the boundary numerically.
    problem = cp.Problem(objective, [w >= 1e-12])
    # cvxpy's automatically-chosen default solver for an exponential-cone
    # problem (needed for the log term) only reaches ~1e-5/1e-6 precision
    # on this problem with its default tolerances -- not tight enough to
    # satisfy OptimizationParams.tol = 1e-8 downstream. CLARABEL with
    # explicit tight tolerances reaches ~1e-10, so it is requested
    # explicitly rather than left to cvxpy's default choice.
    solver_tol = params.tol * 1e-2
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=solver_tol, tol_gap_rel=solver_tol, tol_feas=solver_tol)

    if w.value is None or problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"solve_erc: solver failed to converge (status={problem.status})")

    raw_weights = np.asarray(w.value).flatten()
    weights = raw_weights / raw_weights.sum()

    # The convex reformulation's optimum is the ERC solution in theory,
    # but the numerical solver only reaches it to floating-point/solver
    # tolerance -- verify that explicitly against OptimizationParams.tol
    # rather than assuming the reformulation's exactness carries over.
    achieved = volatility_risk_contributions(weights, cov).pct_contribution
    dispersion = float(np.max(np.abs(achieved - risk_budgets)))
    if dispersion > params.tol:
        logger.warning(
            "solve_erc: risk contributions deviate from target budgets by "
            "up to %.2e, exceeding OptimizationParams.tol=%.2e (solver "
            "status=%s)",
            dispersion, params.tol, problem.status,
        )

    return weights
