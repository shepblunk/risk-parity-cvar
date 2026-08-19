"""
Compute comparison metrics across strategies (Equal Weight, Min-Var,
vol-ERC, CVaR-parity).

TODO
----
- `performance_table`: Sharpe, annualized return/vol, max drawdown,
  Calmar, turnover -- one row per strategy.
- `diversification_ratio`: weighted avg of individual vols / portfolio
  vol -- classic risk-parity diversification metric.
- `risk_contribution_stability`: std-dev of pct risk contributions
  across the backtest, per asset per strategy -- this is the number
  that should make CVaR-parity's story ("stays balanced under stress")
  either hold up or not. Report it honestly either way.
"""

from __future__ import annotations

import pandas as pd

from riskparity.backtest.engine import BacktestResult


def performance_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    raise NotImplementedError


def risk_contribution_stability(results: dict[str, BacktestResult]) -> pd.DataFrame:
    raise NotImplementedError
