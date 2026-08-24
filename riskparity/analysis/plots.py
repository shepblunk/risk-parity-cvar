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

Implementation notes
---------------------
- Only three functions are implemented below, matching the three
  already declared in this module: `plot_risk_contributions` (figure
  1), `plot_cumulative_returns` (figure 2), `plot_contribution_stability`
  (figure 3). Figure 4 (the stress-window zoom) has no dedicated
  function: it is the same chart as figure 2, just called with each
  `pd.Series` in `returns` pre-sliced to a stress window's date range
  (e.g. `{name: r.loc[start:end] for name, r in returns.items()}`,
  using the dates in `riskparity.backtest.stress_periods.STRESS_PERIODS`)
  -- adding a second function that differs only in what slice of data it
  receives would duplicate `plot_cumulative_returns` for no reason.
- "Shared color per strategy across all figures" (the styling rule
  above) is achieved via matplotlib's default color cycle, which
  assigns colors to series in the order they are iterated -- NOT by
  hard-coding a strategy-name-to-color mapping (which would only work
  for a fixed, known set of strategy names, and this module has no
  reason to know what strategies a caller compares). This holds
  automatically as long as the caller passes strategies in the same
  key order to every plotting function it calls (a plain Python `dict`
  preserves insertion order), which is the natural way
  `scripts/run_pipeline.py` would build its `results` dict once and
  reuse it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finalize(fig: plt.Figure, save_path: str | None) -> plt.Figure:
    """Shared save/cleanup step for every plotting function below.

    Always returns `fig` (so a notebook cell ending in a call to one of
    these functions renders it inline), and additionally writes it to
    `save_path` if one is given. `plt.close(fig)` only removes `fig`
    from pyplot's internal figure registry (freeing memory, and
    avoiding the "too many open figures" warning when this module is
    called repeatedly in a pipeline) -- the returned `Figure` object
    itself stays fully valid to display or save.
    """
    fig.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_risk_contributions(pct_contributions: dict[str, pd.Series], save_path: str | None = None) -> plt.Figure:
    """Grouped bar chart of percentage risk contributions per asset, one
    group of bars per strategy, at a single date. A dashed line at
    1/N_assets marks the equal-contribution target (the default risk
    budget used throughout this project) for visual reference.
    """
    table = pd.DataFrame(pct_contributions)  # index = tickers, columns = strategies
    n_assets, n_strategies = table.shape
    x = np.arange(n_assets)
    width = 0.8 / n_strategies

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, strategy in enumerate(table.columns):
        offset = (i - (n_strategies - 1) / 2) * width
        ax.bar(x + offset, table[strategy], width, label=strategy)

    ax.axhline(1.0 / n_assets, color="black", linestyle="--", linewidth=1, label="equal contribution (1/N)")
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=45, ha="right")
    ax.set_ylabel("% risk contribution")
    ax.set_title("Risk contributions by strategy")
    ax.legend()

    return _finalize(fig, save_path)


def plot_cumulative_returns(returns: dict[str, pd.Series], save_path: str | None = None) -> plt.Figure:
    """Cumulative growth of $1, one line per strategy, over the full
    (or a caller-sliced, e.g. stress-window) date range of `returns`.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, r in returns.items():
        wealth = (1.0 + r).cumprod()
        ax.plot(wealth.index, wealth.to_numpy(), label=name)

    ax.set_ylabel("Cumulative growth of $1")
    ax.set_title("Cumulative returns by strategy")
    ax.legend()
    ax.grid(alpha=0.3)

    return _finalize(fig, save_path)


def plot_contribution_stability(history: dict[str, pd.DataFrame], save_path: str | None = None) -> plt.Figure:
    """One subplot per strategy (stacked vertically, sharing the x-axis),
    each showing every asset's %RC across all rebalance dates in that
    strategy's backtest -- the "does it stay balanced over time" chart.
    A dashed line at 1/N_assets marks the equal-contribution target.
    """
    n_strategies = len(history)
    fig, axes = plt.subplots(n_strategies, 1, figsize=(9, 3.5 * n_strategies), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for i, (ax, (name, rc_history)) in enumerate(zip(axes, history.items())):
        for ticker in rc_history.columns:
            ax.plot(rc_history.index, rc_history[ticker], label=ticker)
        ax.axhline(1.0 / rc_history.shape[1], color="black", linestyle="--", linewidth=1)
        ax.set_title(name)
        ax.set_ylabel("% risk contribution")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize="small")

    return _finalize(fig, save_path)
