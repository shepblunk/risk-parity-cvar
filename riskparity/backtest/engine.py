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

Implementation notes
---------------------
- `strategy_fn` is a black box `window_returns -> weights`: it may wrap
  `solve_erc`, `solve_cvar_parity`, an equal-weight rule, anything. The
  engine itself never calls a specific optimizer, so it stays agnostic
  to which strategy it is backtesting -- the caller (e.g.
  `scripts/run_pipeline.py`) builds the closure, deciding e.g. how much
  of `window` a CVaR strategy should look back over.
- No-look-ahead is enforced structurally: the window passed to
  `strategy_fn` at rebalance date t is `returns.iloc[:loc]` where `loc`
  is t's position in the index -- i.e. it strictly EXCLUDES the return
  realized on t itself (and everything after). `strategy_fn` cannot see
  the return it is about to be judged on.
- Rebalance dates are the last trading date in each
  `BacktestParams.rebalance_freq` period actually present in `returns`
  (via `pandas.Series.resample(...).last()`), not a fixed calendar grid
  -- this avoids picking non-trading days when `returns`'s index has
  holidays removed.
- A rebalance is skipped (not attempted) until at least
  `RiskParams.vol_lookback_days` of history is available, since the
  ex-ante risk-contribution diagnostic below needs a non-degenerate
  covariance estimate; `run_backtest` raises if no rebalance date ever
  clears that bar (e.g. `returns` is shorter than one lookback window).
- Between rebalances, weights are NOT re-targeted daily -- they drift
  with realized asset returns (buy-and-hold), which is what makes
  turnover (and risk-contribution drift, the "does the portfolio stay
  balanced between rebalances" question flagged above) a meaningful,
  non-trivial quantity to look at instead of being zero by construction.
- `BacktestParams.transaction_cost_bps` is charged once per rebalance,
  on that day's return, proportional to L1 turnover
  `sum(|target_weights - weights_held_just_before_rebalancing|)`. The
  very first rebalance is charged against an implicit all-cash starting
  position (`current_weights = 0`), i.e. the cost of funding the
  portfolio is included rather than treated as free.
- Ex-ante risk contributions are reported using
  `volatility_risk_contributions` for EVERY strategy, including
  CVaR-parity ones -- deliberately, not because volatility is "the"
  right risk measure. Using one common lens for every strategy is what
  makes it possible to ask the project's central question in the
  backtest, not just at a single static date: does a portfolio built to
  equalize Component CVaR end up with unequal *volatility* contributions
  in practice, and how does that gap evolve over time? A CVaR-lens
  version of this diagnostic is a natural extension but is not computed
  here, since the engine has no reliable way to know a scenario window a
  black-box `strategy_fn` would consider appropriate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from riskparity.config import BacktestParams, RISK_PARAMS
from riskparity.risk.contributions import volatility_risk_contributions
from riskparity.risk.volatility import shrunk_covariance

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    returns: pd.Series  # realized portfolio returns, daily
    weights_history: pd.DataFrame  # index = rebalance dates
    turnover: pd.Series  # per rebalance date
    risk_contributions_history: pd.DataFrame  # pct RC per asset, per rebalance date


def _rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading date in each `freq` period actually present in `index`.

    Using the LAST date in each period (e.g. the last trading day of
    each month for freq="ME") means a rebalance uses the most recent
    information available in that period, and periods with no trading
    days in `index` (a resample bin that happens to be empty) are
    dropped rather than producing a spurious rebalance date.
    """
    last_per_period = index.to_series().resample(freq).last().dropna()
    return pd.DatetimeIndex(last_per_period)


def run_backtest(
    returns: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame], np.ndarray],
    params: BacktestParams = BacktestParams(),
) -> BacktestResult:
    """Roll `strategy_fn` forward over `returns`, rebalancing on
    `params.rebalance_freq`, and return realized performance alongside
    ex-ante diagnostics. See the module docstring for the exact rules
    (no-look-ahead, weight drift, transaction costs, risk-contribution
    lens) this implements.
    """
    returns = returns.sort_index()
    dates = returns.index
    tickers = returns.columns
    n_assets = len(tickers)

    rebalance_dates = _rebalance_dates(dates, params.rebalance_freq)

    weights_rows: dict[pd.Timestamp, np.ndarray] = {}
    turnover_rows: dict[pd.Timestamp, float] = {}
    risk_contribution_rows: dict[pd.Timestamp, np.ndarray] = {}
    daily_returns = pd.Series(0.0, index=dates)

    current_weights = np.zeros(n_assets)  # implicit all-cash position pre-first-rebalance

    for i, rebal_date in enumerate(rebalance_dates):
        loc = dates.get_loc(rebal_date)
        window = returns.iloc[:loc]  # strictly before rebal_date -- no look-ahead

        if len(window) < RISK_PARAMS.vol_lookback_days:
            logger.info(
                "run_backtest: skipping rebalance at %s, only %d/%d days "
                "of history available",
                rebal_date.date(), len(window), RISK_PARAMS.vol_lookback_days,
            )
            continue

        target_weights = np.asarray(strategy_fn(window), dtype=float)
        turnover = float(np.abs(target_weights - current_weights).sum())

        cov = shrunk_covariance(window, RISK_PARAMS.vol_lookback_days, RISK_PARAMS.covariance_shrinkage)
        risk_contribution_rows[rebal_date] = volatility_risk_contributions(
            target_weights, cov.to_numpy()
        ).pct_contribution

        weights_rows[rebal_date] = target_weights
        turnover_rows[rebal_date] = turnover

        # Hold target_weights from rebal_date up to (but not including) the
        # next rebalance date, drifting them with realized returns each day.
        next_loc = dates.get_loc(rebalance_dates[i + 1]) if i + 1 < len(rebalance_dates) else len(dates)
        held_weights = target_weights.copy()
        for t_loc in range(loc, next_loc):
            day_returns = returns.iloc[t_loc].to_numpy()
            gross_return = float(held_weights @ day_returns)
            cost = turnover * params.transaction_cost_bps / 1e4 if t_loc == loc else 0.0
            daily_returns.iloc[t_loc] = gross_return - cost

            held_weights = held_weights * (1.0 + day_returns)
            held_weights = held_weights / held_weights.sum()

        current_weights = held_weights

    if not weights_rows:
        raise ValueError(
            "run_backtest: no rebalance date had "
            f"RiskParams.vol_lookback_days={RISK_PARAMS.vol_lookback_days} days of "
            "history available -- pass a longer `returns` history or lower the lookback"
        )

    first_rebalance = next(iter(weights_rows))
    return BacktestResult(
        returns=daily_returns.loc[first_rebalance:],
        weights_history=pd.DataFrame.from_dict(weights_rows, orient="index", columns=tickers),
        turnover=pd.Series(turnover_rows),
        risk_contributions_history=pd.DataFrame.from_dict(risk_contribution_rows, orient="index", columns=tickers),
    )
