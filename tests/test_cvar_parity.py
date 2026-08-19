"""
Tests for riskparity.optimization.cvar_parity.

TODO once solve_cvar_parity is implemented:
- test_cvar_parity_weights_sum_to_one_and_long_only
- test_cvar_parity_converges: convergence_history should be
  (roughly) monotonically decreasing -- if it isn't, that's a real
  finding to discuss in the README, not a bug to hide.
- test_cvar_parity_contributions_equal_at_optimum: within a looser
  tolerance than the vol-ERC test, since the scenario-based Component
  CVaR estimator is itself noisy -- document the tolerance choice.
- test_cvar_parity_differs_from_vol_erc_on_skewed_universe: construct a
  synthetic return set where one asset has strong negative skew (fat
  left tail) with the same volatility as another asset, and verify
  CVaR-parity allocates less to it than vol-ERC does. This is the
  central empirical claim of the project -- it deserves its own test.
"""

import pytest


@pytest.mark.skip(reason="solve_cvar_parity not implemented yet")
def test_cvar_parity_contributions_equal_at_optimum():
    ...
