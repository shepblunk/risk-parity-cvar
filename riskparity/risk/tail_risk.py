"""
VaR and CVaR estimation for a portfolio's return distribution.

Definitions (confidence level alpha, e.g. 0.975)
-------------------------------------------------
VaR_alpha(R)  = -quantile of R at (1 - alpha)          [a loss, positive number]
CVaR_alpha(R) = -E[R | R <= -VaR_alpha(R)]              (expected shortfall)

TODO
----
- `historical_var` / `historical_cvar`: non-parametric, directly from the
  empirical distribution of portfolio returns over `cvar_lookback_days`.
  Simple, few assumptions, but noisy with limited history -- say so.
- `cornish_fisher_cvar`: parametric CVaR that adjusts the Gaussian
  quantile for sample skewness/kurtosis (Cornish-Fisher expansion).
  More stable than the historical estimator with less data, but can
  behave badly for extreme skew/kurtosis -- worth a sanity check plot
  comparing both methods on the same series.
- Both functions should accept a *vector of portfolio returns* (i.e.
  R_t = w' r_t already computed), not weights + covariance -- CVaR is
  estimated from the realized/simulated portfolio return series, not a
  closed form like volatility.
"""

from __future__ import annotations
from scipy.stats import skew,kurtosis,norm 
import numpy as np
import pandas as pd


def _validate_confidence(confidence: float) -> None:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")


def historical_var(portfolio_returns: pd.Series, confidence: float) -> float:
    _validate_confidence(confidence)
    portfolio_returns = np.asarray(portfolio_returns,dtype=float)
    percentile = (1-confidence)*100
    var_value = np.percentile(portfolio_returns,percentile)
    return -var_value


def historical_cvar(portfolio_returns: pd.Series, confidence: float) -> float:
    _validate_confidence(confidence)
    portfolio_returns = np.asarray(portfolio_returns,dtype=float)
    percentile=(1-confidence)*100
    var_value = np.percentile(portfolio_returns,percentile)
    tail_losses = portfolio_returns[portfolio_returns<=var_value]
    return -np.mean(tail_losses)

def cornish_fisher_cvar(portfolio_returns: pd.Series, confidence: float) -> float:
    _validate_confidence(confidence)
    portfolio_returns = np.asarray(portfolio_returns,dtype=float)
    mu = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns,ddof=1)
    S = skew(portfolio_returns)
    K = kurtosis(portfolio_returns,fisher=True)
    alpha = (1-confidence)
    z_alpha = norm.ppf(alpha)
    z_cf = (
        z_alpha
        + (z_alpha**2 - 1) * S / 6
        + (z_alpha**3 - 3 * z_alpha) * K / 24
        - (2 * z_alpha**3 - 5 * z_alpha) * (S**2) / 36
    )
    d_z = norm.pdf(z_cf)
    e_cf = (
        -d_z / alpha
        * (
            1
            + S / 6 * z_cf**3
            + K / 24 * (z_cf**4 - 2 * z_cf**2 - 1)
            + S**2 / 36 * (2 * z_cf**4 - 5 * z_cf**2 - 1)
        )
    )
    
    # CVaR modifiée
    m_cvar = -(mu + sigma * e_cf)
    return m_cvar