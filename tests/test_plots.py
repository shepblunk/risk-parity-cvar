"""
Tests for riskparity.analysis.plots.

Figures are checked structurally (number of axes, number of lines/bars,
titles, whether a file actually gets written) -- not via pixel/image
comparison, which would be brittle and add no real coverage here.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend, must be set before pyplot is used anywhere

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from riskparity.analysis.plots import (
    plot_contribution_stability,
    plot_cumulative_returns,
    plot_risk_contributions,
)


def test_plot_risk_contributions_returns_figure_with_bars_and_reference_line():
    pct_contributions = {
        "vol_erc": pd.Series({"A": 0.34, "B": 0.33, "C": 0.33}),
        "cvar_parity": pd.Series({"A": 0.20, "B": 0.40, "C": 0.40}),
    }

    fig = plot_risk_contributions(pct_contributions)

    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert len(ax.patches) == 6  # 2 strategies * 3 assets
    assert len(ax.lines) == 1  # the 1/N reference line
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1 / 3, 1 / 3])


def test_plot_risk_contributions_saves_file(tmp_path):
    pct_contributions = {"strat": pd.Series({"A": 0.5, "B": 0.5})}
    save_path = tmp_path / "figures" / "rc.png"  # parent dir does not exist yet

    plot_risk_contributions(pct_contributions, save_path=str(save_path))

    assert save_path.exists()


def test_plot_cumulative_returns_one_line_per_strategy():
    dates = pd.bdate_range("2020-01-01", periods=10)
    returns = {
        "strat_a": pd.Series(0.001, index=dates),
        "strat_b": pd.Series(-0.0005, index=dates),
    }

    fig = plot_cumulative_returns(returns)

    ax = fig.axes[0]
    assert len(ax.lines) == 2
    wealth_a = ax.lines[0].get_ydata()
    np.testing.assert_allclose(wealth_a[-1], 1.001**10, rtol=1e-9)


def test_plot_cumulative_returns_saves_file(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=5)
    returns = {"strat": pd.Series(0.0, index=dates)}
    save_path = tmp_path / "cum_returns.png"

    plot_cumulative_returns(returns, save_path=str(save_path))

    assert save_path.exists()


def test_plot_contribution_stability_one_subplot_per_strategy():
    dates = pd.bdate_range("2020-01-01", periods=4)
    history = {
        "vol_erc": pd.DataFrame({"A": [0.5, 0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5, 0.5]}, index=dates),
        "cvar_parity": pd.DataFrame({"A": [0.3, 0.4, 0.35, 0.45], "B": [0.7, 0.6, 0.65, 0.55]}, index=dates),
    }

    fig = plot_contribution_stability(history)

    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert len(ax.lines) == 3  # 2 assets + the 1/N reference line
    assert fig.axes[0].get_title() == "vol_erc"
    assert fig.axes[1].get_title() == "cvar_parity"


def test_plot_contribution_stability_saves_file(tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=3)
    history = {"strat": pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]}, index=dates)}
    save_path = tmp_path / "stability.png"

    plot_contribution_stability(history, save_path=str(save_path))

    assert save_path.exists()
