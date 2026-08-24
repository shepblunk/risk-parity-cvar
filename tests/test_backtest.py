"""
Tests for riskparity.backtest.engine.

Most checks here are exact/structural (turnover accounting, the
rebalance-day return formula, no-look-ahead) rather than statistical,
because `run_backtest` is bookkeeping around a `strategy_fn` black box,
not an estimator with sampling noise -- unlike
tests/test_cvar_parity.py, there is no scenario-based tolerance to
justify here.
"""

import numpy as np
import pandas as pd
import pytest

from riskparity.backtest.engine import _rebalance_dates, run_backtest
from riskparity.config import BacktestParams, RISK_PARAMS

TICKERS = ["A", "B", "C"]


def _synthetic_returns(n_days: int, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    data = rng.normal(0.0003, 0.01, size=(n_days, n_assets))
    return pd.DataFrame(data, index=dates, columns=TICKERS[:n_assets])


def _equal_weight_strategy(window: pd.DataFrame) -> np.ndarray:
    n = window.shape[1]
    return np.full(n, 1.0 / n)


# --- _rebalance_dates ------------------------------------------------


def test_rebalance_dates_are_last_trading_day_of_each_month():
    dates = pd.bdate_range("2020-01-01", "2020-03-31")

    rebal = _rebalance_dates(dates, "ME")

    assert list(rebal) == [dates[dates.month == m].max() for m in (1, 2, 3)]


def test_rebalance_dates_drops_periods_with_no_trading_days():
    """A resample bin with no data (e.g. a month entirely absent from
    `dates`) must not produce a spurious NaT-derived rebalance date."""
    jan = pd.bdate_range("2020-01-01", "2020-01-31")
    mar = pd.bdate_range("2020-03-01", "2020-03-31")  # February is missing entirely
    dates = jan.append(mar)

    rebal = _rebalance_dates(dates, "ME")

    assert list(rebal) == [jan.max(), mar.max()]


# --- run_backtest: warm-up / validation -------------------------------


def test_run_backtest_raises_if_no_rebalance_has_enough_history():
    returns = _synthetic_returns(RISK_PARAMS.vol_lookback_days - 10)

    with pytest.raises(ValueError):
        run_backtest(returns, _equal_weight_strategy)


def test_run_backtest_skips_rebalances_before_lookback_window_is_full():
    returns = _synthetic_returns(RISK_PARAMS.vol_lookback_days + 100, seed=5)
    all_rebal_dates = _rebalance_dates(returns.index, BacktestParams().rebalance_freq)
    expected = [d for d in all_rebal_dates if returns.index.get_loc(d) >= RISK_PARAMS.vol_lookback_days]

    result = run_backtest(returns, _equal_weight_strategy)

    assert list(result.weights_history.index) == expected


# --- run_backtest: structural invariants -------------------------------


def test_run_backtest_weights_history_long_only_sum_to_one():
    returns = _synthetic_returns(RISK_PARAMS.vol_lookback_days + 400)

    result = run_backtest(returns, _equal_weight_strategy)

    assert not result.weights_history.empty
    np.testing.assert_allclose(result.weights_history.sum(axis=1), 1.0, atol=1e-10)
    assert (result.weights_history >= 0).all().all()


def test_run_backtest_risk_contributions_sum_to_one_per_rebalance():
    returns = _synthetic_returns(RISK_PARAMS.vol_lookback_days + 400)

    result = run_backtest(returns, _equal_weight_strategy)

    np.testing.assert_allclose(result.risk_contributions_history.sum(axis=1), 1.0, atol=1e-8)


def test_run_backtest_returns_index_starts_at_first_rebalance():
    returns = _synthetic_returns(RISK_PARAMS.vol_lookback_days + 60)

    result = run_backtest(returns, _equal_weight_strategy)

    assert result.returns.index.min() == result.weights_history.index[0]


# --- run_backtest: no look-ahead ---------------------------------------


def test_run_backtest_strategy_never_sees_data_on_or_after_rebalance_date():
    seen_windows = []

    def spy_strategy(window: pd.DataFrame) -> np.ndarray:
        seen_windows.append(window)
        return np.full(window.shape[1], 1.0 / window.shape[1])

    returns = _synthetic_returns(RISK_PARAMS.vol_lookback_days + 120)
    result = run_backtest(returns, spy_strategy)

    rebal_dates = result.weights_history.index
    assert len(seen_windows) == len(rebal_dates)
    for window, rebal_date in zip(seen_windows, rebal_dates):
        assert window.index.max() < rebal_date


# --- run_backtest: turnover / transaction cost accounting --------------


def test_run_backtest_first_turnover_equals_full_allocation_from_cash():
    """The first rebalance starts from an implicit all-cash position
    (current_weights = 0), so turnover must equal sum(target_weights),
    i.e. 1.0 for a fully-invested long-only strategy."""
    returns = _synthetic_returns(RISK_PARAMS.vol_lookback_days + 60)

    result = run_backtest(returns, _equal_weight_strategy)

    assert result.turnover.iloc[0] == pytest.approx(1.0, abs=1e-8)


def test_run_backtest_rebalance_day_return_matches_formula():
    """On the rebalance day itself weights have not drifted yet
    (held_weights == target_weights), so the realized net return must
    equal the textbook formula: w . r_t - turnover * bps / 1e4 -- exactly,
    not approximately."""
    returns = _synthetic_returns(RISK_PARAMS.vol_lookback_days + 40, seed=3)
    params = BacktestParams(transaction_cost_bps=25.0)

    result = run_backtest(returns, _equal_weight_strategy, params=params)

    rebal_date = result.weights_history.index[1]
    target_weights = result.weights_history.loc[rebal_date].to_numpy()
    turnover = result.turnover.loc[rebal_date]
    day_returns = returns.loc[rebal_date].to_numpy()

    expected = target_weights @ day_returns - turnover * params.transaction_cost_bps / 1e4
    assert result.returns.loc[rebal_date] == pytest.approx(expected, abs=1e-10)


def test_run_backtest_transaction_costs_only_bite_on_rebalance_days():
    """Comparing a zero-cost run against a costly run isolates the
    effect of `transaction_cost_bps`: returns must be identical on every
    non-rebalance day and strictly lower for the costly run on every
    rebalance day (turnover is essentially never exactly 0 with
    continuous random returns, so the inequality is not flaky)."""
    returns = _synthetic_returns(RISK_PARAMS.vol_lookback_days + 90, seed=7)
    no_cost = BacktestParams(transaction_cost_bps=0.0)
    with_cost = BacktestParams(transaction_cost_bps=50.0)

    result_no_cost = run_backtest(returns, _equal_weight_strategy, params=no_cost)
    result_with_cost = run_backtest(returns, _equal_weight_strategy, params=with_cost)

    rebal_dates = result_no_cost.weights_history.index
    non_rebal_mask = ~result_no_cost.returns.index.isin(rebal_dates)

    np.testing.assert_allclose(
        result_no_cost.returns[non_rebal_mask],
        result_with_cost.returns[non_rebal_mask],
        atol=1e-12,
    )
    assert (result_with_cost.returns[rebal_dates] < result_no_cost.returns[rebal_dates]).all()
