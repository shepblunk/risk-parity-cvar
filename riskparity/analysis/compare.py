"""
Compute comparison metrics across strategies (Equal Weight, Min-Var,
vol-ERC, CVaR-parity).

Implementation notes
---------------------
- `performance_table` and `risk_contribution_stability` are implemented
  below, operating purely on the `BacktestResult`s already produced by
  `riskparity.backtest.engine.run_backtest` -- no re-fetching of prices
  or re-solving of any optimizer here, keeping this module a thin
  aggregation layer over backtest output.
- `diversification_ratio` (mentioned in the original module TODO, and
  in docs/notes.tex's "Comparison Metrics" section) is deliberately NOT
  implemented here: DR(w) = sum_i(w_i * sigma_i) / sigma_p(w) needs
  each asset's *individual* volatility and the portfolio volatility at
  each rebalance date, and `BacktestResult` currently only stores
  percentage risk *contributions* (RC_i / sigma_p), not the underlying
  sigma_i or covariance matrix. Adding it would mean extending
  `BacktestResult`'s schema in `backtest/engine.py` -- a bigger change
  than filling in this module -- so it is flagged here rather than
  worked around with a silent re-derivation that could drift out of
  sync with the engine's actual covariance estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from riskparity.backtest.engine import BacktestResult
from riskparity.risk.volatility import TRADING_DAYS_PER_YEAR


def performance_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """One row per strategy: annualized return, annualized volatility,
    Sharpe ratio, max drawdown, Calmar ratio, average turnover.

    Sharpe assumes a 0% risk-free rate: this project only sources free
    ETF price data (no risk-free/cash-rate series), so introducing a
    nonzero rate would mean adding a data dependency purely for this
    one ratio -- documented here rather than silently baked in. Max
    drawdown (and Calmar's denominator) are reported as positive loss
    magnitudes, matching this project's risk-as-positive-number
    convention (see `risk/tail_risk.py`'s CVaR sign flip), not the
    negative-number convention some libraries use.
    """
    rows = {}
    for name, result in results.items():
        r = result.returns
        n_days = len(r)

        wealth = (1.0 + r).cumprod()
        cumulative_return = wealth.iloc[-1] - 1.0
        annualized_return = (1.0 + cumulative_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0
        annualized_vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
        drawdown = wealth / wealth.cummax() - 1.0
        max_drawdown = -drawdown.min()  # positive magnitude

        # A strategy with genuinely zero variance/drawdown (e.g. constant
        # daily returns) does not land on annualized_vol/max_drawdown ==
        # 0.0 exactly -- float64 summation noise leaves a residual on the
        # order of 1e-19, which is still ">0" but not a meaningful
        # denominator. 1e-12 (matching the numerical floor already used in
        # optimization/erc.py) safely separates that noise from any real
        # market volatility/drawdown, which is orders of magnitude larger.
        EPS = 1e-12
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > EPS else np.nan
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > EPS else np.nan

        rows[name] = {
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "avg_turnover": result.turnover.mean(),
        }

    return pd.DataFrame.from_dict(rows, orient="index")


def risk_contribution_stability(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Per strategy, per asset: standard deviation of the percentage
    risk contribution across every rebalance date in the backtest.

    This is the number that determines whether a strategy's "stays
    balanced over time" story actually holds up: a low std means
    %RC_i barely moved from its target across the whole backtest; a
    high one means the portfolio drifted meaningfully out of balance
    between rebalances, however good it looked at any single date.
    Reported honestly regardless of which strategy comes out ahead.
    """
    return pd.DataFrame(
        {name: result.risk_contributions_history.std(axis=0, ddof=1) for name, result in results.items()}
    ).T
