"""
Plotting helpers -- kept separate from computation so figures can be
regenerated without re-running the backtest.

Suggested figures for results/figures/ (the set that tells the story
of the project in a README at a glance):
  1. Risk contribution bar chart (vol-ERC vs CVaR-parity), single date.
  2. Cumulative return, all strategies, full backtest.
  3. Rolling risk-contribution stability (per asset) through time --
     the "does it stay balanced" chart.
  4. Zoomed comparison on 1-2 stress windows from
     riskparity.backtest.stress_periods, showing where vol-ERC and
     CVaR-parity diverge most.

TODO: implement using matplotlib. Keep styling minimal and consistent
(shared color per strategy across all figures) -- consistency across
figures matters more than any single figure being fancy.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_risk_contributions(pct_contributions: dict[str, pd.Series], save_path: str | None = None):
    raise NotImplementedError


def plot_cumulative_returns(returns: dict[str, pd.Series], save_path: str | None = None):
    raise NotImplementedError


def plot_contribution_stability(history: dict[str, pd.DataFrame], save_path: str | None = None):
    raise NotImplementedError
