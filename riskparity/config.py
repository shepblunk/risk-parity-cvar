"""
Central configuration for the project.

Keeping every tunable parameter here (rather than hard-coded inside
scripts) makes the pipeline reproducible and easy to defend in an
interview: "what happens if I change the lookback window / confidence
level?" becomes a one-line edit, not a code hunt.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Universe:
    """Multi-asset ETF universe used throughout the project.

    TODO: adjust tickers to what's actually available via yfinance and to
    the asset classes you want to represent (equities, sovereign bonds,
    credit, gold/commodities, real estate...). Keep enough breadth that
    risk parity is a meaningful exercise (>= 4-5 uncorrelated sleeves).
    """

    tickers: tuple = (
        "SPY",   # US equities
        "TLT",   # US long treasuries
        "IEF",   # US intermediate treasuries
        "GLD",   # gold
        "DBC",   # broad commodities
        "VNQ",   # US REITs
    )
    start_date: str = "2007-01-01"
    end_date: str | None = None  # None -> today


@dataclass(frozen=True)
class RiskParams:
    """Parameters shared by the risk-measurement modules."""

    # Volatility estimation
    vol_lookback_days: int = 252
    covariance_shrinkage: float = 0.10  # Ledoit-Wolf-style shrinkage intensity

    # VaR / CVaR estimation
    cvar_confidence: float = 0.975
    cvar_lookback_days: int = 500
    cvar_method: str = "historical"  # "historical" | "cornish_fisher"


@dataclass(frozen=True)
class OptimizationParams:
    """Parameters for the ERC / CVaR-parity solvers."""

    max_iter: int = 500
    tol: float = 1e-8
    weight_bounds: tuple = (0.0, 1.0)  # long-only by default
    # Step size for solve_cvar_parity's cyclical weight update (the vol-ERC
    # solver has no equivalent -- it is solved directly via cvxpy). Kept
    # here rather than hard-coded so it can be tuned without touching
    # optimization/cvar_parity.py.
    cvar_parity_learning_rate: float = 0.05


@dataclass(frozen=True)
class BacktestParams:
    rebalance_freq: str = "ME"  # pandas offset alias: "M", "Q", ...
    transaction_cost_bps: float = 5.0


UNIVERSE = Universe()
RISK_PARAMS = RiskParams()
OPT_PARAMS = OptimizationParams()
BACKTEST_PARAMS = BacktestParams()
