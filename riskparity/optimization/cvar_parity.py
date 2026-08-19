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

TODO
----
- Implement `solve_cvar_parity` using the cyclical update described
  above.
- Track and return the convergence history (dispersion of RC_i per
  iteration) -- this becomes a diagnostic plot in results/figures/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskparity.config import OptimizationParams


def solve_cvar_parity(
    asset_returns: pd.DataFrame,
    confidence: float,
    risk_budgets: np.ndarray | None = None,
    params: OptimizationParams = OptimizationParams(),
) -> tuple[np.ndarray, list[float]]:
    """Return (weights, convergence_history) where convergence_history
    is the dispersion-of-contributions objective at each iteration.
    """
    raise NotImplementedError
