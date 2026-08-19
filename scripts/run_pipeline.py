#!/usr/bin/env python
"""
End-to-end pipeline: download -> clean -> optimize (single-date) ->
backtest -> compare -> save figures/tables.

This is the single entry point a recruiter (or you, in six months)
should be able to run to reproduce every result in results/:

    python scripts/run_pipeline.py

TODO: wire this up once the modules below are implemented. Keep this
script thin -- it should just call into riskparity.*, not contain
logic of its own. If you find yourself writing real computation here,
it probably belongs in a module instead.
"""

from __future__ import annotations

from riskparity.config import UNIVERSE, RISK_PARAMS, OPT_PARAMS, BACKTEST_PARAMS
from riskparity.data.download import download_prices
from riskparity.data.clean import clean_prices, compute_log_returns

# from riskparity.risk.volatility import shrunk_covariance
# from riskparity.risk.contributions import volatility_risk_contributions, cvar_risk_contributions
# from riskparity.optimization.erc import solve_erc
# from riskparity.optimization.cvar_parity import solve_cvar_parity
# from riskparity.backtest.engine import run_backtest
# from riskparity.analysis.compare import performance_table, risk_contribution_stability
# from riskparity.analysis.plots import plot_risk_contributions, plot_cumulative_returns


def main() -> None:
    print("1/5 Downloading data...")
    raw_prices = download_prices(UNIVERSE)

    print("2/5 Cleaning data...")
    prices = clean_prices(raw_prices)
    returns = compute_log_returns(prices)

    print("3/5 Solving single-date allocations (vol-ERC vs CVaR-parity)...")
    # TODO

    print("4/5 Running rolling backtest for all strategies...")
    # TODO

    print("5/5 Building comparison tables and figures...")
    # TODO

    print("Done. See results/figures/ and results/tables/.")


if __name__ == "__main__":
    main()
