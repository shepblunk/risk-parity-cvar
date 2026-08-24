"""
Tests for riskparity.analysis.compare.

`performance_table` and `risk_contribution_stability` are pure
aggregations over `BacktestResult` -- built directly here (bypassing
`run_backtest`) with small, hand-checkable numbers, so every assertion
below is an exact formula check, not a statistical/tolerance one.
"""

import numpy as np
import pandas as pd
import pytest

from riskparity.analysis.compare import performance_table, risk_contribution_stability
from riskparity.backtest.engine import BacktestResult
from riskparity.risk.volatility import TRADING_DAYS_PER_YEAR


def _dates(n):
    return pd.bdate_range("2020-01-01", periods=n)


def _result(returns: pd.Series, turnover: pd.Series, risk_contributions: pd.DataFrame) -> BacktestResult:
    return BacktestResult(
        returns=returns,
        weights_history=pd.DataFrame(index=risk_contributions.index, columns=risk_contributions.columns),
        turnover=turnover,
        risk_contributions_history=risk_contributions,
    )


# --- performance_table --------------------------------------------------


def test_performance_table_annualized_return_matches_formula_over_one_year():
    """With exactly TRADING_DAYS_PER_YEAR observations, the annualization
    exponent is 1, so annualized_return must equal the raw cumulative
    return exactly."""
    n = TRADING_DAYS_PER_YEAR
    r = pd.Series(0.001, index=_dates(n))
    turnover = pd.Series([1.0])
    rc = pd.DataFrame(0.5, index=[_dates(1)[0]], columns=["A", "B"])

    table = performance_table({"strat": _result(r, turnover, rc)})

    expected = (1.001) ** n - 1.0
    assert table.loc["strat", "annualized_return"] == pytest.approx(expected, rel=1e-10)


def test_performance_table_zero_vol_gives_nan_sharpe_and_calmar():
    r = pd.Series(0.001, index=_dates(10))
    turnover = pd.Series([0.5])
    rc = pd.DataFrame(0.5, index=[_dates(1)[0]], columns=["A", "B"])

    table = performance_table({"strat": _result(r, turnover, rc)})

    assert table.loc["strat", "annualized_volatility"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(table.loc["strat", "sharpe_ratio"])
    assert table.loc["strat", "max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(table.loc["strat", "calmar_ratio"])


def test_performance_table_max_drawdown_matches_hand_computed_value():
    """Flat, then a 10% drop, then flat again: peak-to-trough is exactly
    10%, and stays there since the price never recovers."""
    r = pd.Series([0.0, -0.10, 0.0], index=_dates(3))
    turnover = pd.Series([0.2])
    rc = pd.DataFrame(0.5, index=[_dates(1)[0]], columns=["A", "B"])

    table = performance_table({"strat": _result(r, turnover, rc)})

    assert table.loc["strat", "max_drawdown"] == pytest.approx(0.10, abs=1e-12)


def test_performance_table_avg_turnover_is_plain_mean():
    r = pd.Series(0.0005, index=_dates(5))
    turnover = pd.Series([1.0, 0.2, 0.3, 0.1])
    rc = pd.DataFrame(0.5, index=[_dates(1)[0]], columns=["A", "B"])

    table = performance_table({"strat": _result(r, turnover, rc)})

    assert table.loc["strat", "avg_turnover"] == pytest.approx(turnover.mean())


def test_performance_table_one_row_per_strategy():
    r = pd.Series(0.0003, index=_dates(20))
    turnover = pd.Series([0.4])
    rc = pd.DataFrame(0.5, index=[_dates(1)[0]], columns=["A", "B"])
    results = {"vol_erc": _result(r, turnover, rc), "cvar_parity": _result(r, turnover, rc)}

    table = performance_table(results)

    assert list(table.index) == ["vol_erc", "cvar_parity"]
    assert set(table.columns) == {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "calmar_ratio",
        "avg_turnover",
    }


# --- risk_contribution_stability -----------------------------------------


def test_risk_contribution_stability_matches_manual_std():
    dates = _dates(4)
    rc = pd.DataFrame(
        {"A": [0.5, 0.6, 0.4, 0.55], "B": [0.5, 0.4, 0.6, 0.45]},
        index=dates,
    )
    result = _result(pd.Series(0.0, index=_dates(1)), pd.Series([0.0]), rc)

    stability = risk_contribution_stability({"strat": result})

    np.testing.assert_allclose(stability.loc["strat", "A"], rc["A"].std(ddof=1))
    np.testing.assert_allclose(stability.loc["strat", "B"], rc["B"].std(ddof=1))


def test_risk_contribution_stability_lower_for_more_stable_strategy():
    """A strategy whose %RC barely moves across rebalances must show a
    lower std than one that swings widely -- the "stays balanced over
    time" signal this function exists to produce."""
    dates = _dates(4)
    stable_rc = pd.DataFrame({"A": [0.50, 0.51, 0.49, 0.50], "B": [0.50, 0.49, 0.51, 0.50]}, index=dates)
    volatile_rc = pd.DataFrame({"A": [0.30, 0.70, 0.35, 0.65], "B": [0.70, 0.30, 0.65, 0.35]}, index=dates)
    dummy_returns = pd.Series(0.0, index=_dates(1))
    dummy_turnover = pd.Series([0.0])

    results = {
        "stable": _result(dummy_returns, dummy_turnover, stable_rc),
        "volatile": _result(dummy_returns, dummy_turnover, volatile_rc),
    }

    stability = risk_contribution_stability(results)

    assert (stability.loc["stable"] < stability.loc["volatile"]).all()


def test_risk_contribution_stability_index_and_columns():
    dates = _dates(3)
    rc = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=dates)
    dummy_returns = pd.Series(0.0, index=_dates(1))
    dummy_turnover = pd.Series([0.0])
    results = {"vol_erc": _result(dummy_returns, dummy_turnover, rc)}

    stability = risk_contribution_stability(results)

    assert list(stability.index) == ["vol_erc"]
    assert list(stability.columns) == ["A", "B"]
