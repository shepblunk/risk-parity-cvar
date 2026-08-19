"""
riskparity
==========

A small research library comparing volatility-based Risk Parity / ERC
(Equal Risk Contribution) with tail-risk-based parity using Component CVaR,
on a multi-asset ETF universe.

Package layout
---------------
- riskparity.data           : download & clean price data
- riskparity.risk           : risk measurement (volatility, VaR/CVaR, Euler
                               risk-contribution decomposition)
- riskparity.optimization   : ERC (vol) and CVaR-parity solvers
- riskparity.backtest       : rolling backtest engine + known stress windows
- riskparity.analysis       : cross-strategy comparison and plotting
"""

__version__ = "0.1.0"
