"""
Rolling backtest engine.

Design
------
At each rebalance date t (per `BacktestParams.rebalance_freq`):
  1. Use only data available up to t (no look-ahead) to estimate
     covariance / return scenarios.
  2. Solve for weights under each strategy being compared (Equal
     Weight, Min-Variance, vol-ERC, CVaR-parity).
  3. Hold those weights until the next rebalance date, accruing
     `BacktestParams.transaction_cost_bps` on turnover at each
     rebalance.
  4. Record: realized portfolio returns, ex-ante risk contributions at
     the rebalance date, and realized weights drift between rebalances.

TODO
----
- Implement `run_backtest(returns, strategy_fn, params) -> BacktestResult`
  where `strategy_fn(window_returns) -> weights` is one of the
  optimizers from riskparity.optimization, so the engine is agnostic to
  which strategy it's running.
- Guard against look-ahead bias explicitly: the window passed to
  `strategy_fn` must strictly end before the rebalance date.
- Track risk-contribution DRIFT between rebalances (using weights fixed
  at t, contributions recomputed at t+1, t+2, ...) -- this is the
  "does the portfolio stay balanced between rebalances" metric that
  matters in practice and is easy to forget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from riskparity.config import BacktestParams


@dataclass
class BacktestResult:
    returns: pd.Series  # realized portfolio returns, daily
    weights_history: pd.DataFrame  # index = rebalance dates
    turnover: pd.Series  # per rebalance date
    risk_contributions_history: pd.DataFrame  # pct RC per asset, per rebalance date


def run_backtest(
    returns: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame], np.ndarray],
    params: BacktestParams = BacktestParams(),
) -> BacktestResult:
    raise NotImplementedError
