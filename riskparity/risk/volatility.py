"""
Covariance / volatility estimation.

Why shrinkage matters here specifically: with ~6 assets and a 252-day
window the sample covariance is reasonably well-conditioned, but
individual pairwise correlations are still estimated from a limited
sample and are noisier than the variances on the diagonal -- shrinking
them toward zero trades a small amount of bias for a reduction in
estimation-error variance. The effect is modest on this small,
well-conditioned universe; that is said explicitly here rather than
presenting shrinkage as a free lunch.

Design choice: `shrunk_covariance` uses a **fixed shrinkage intensity**
(`RiskParams.covariance_shrinkage`, a tunable config value) applied
manually, rather than `sklearn.covariance.LedoitWolf`'s automatically
estimated optimal intensity. The two are not composable -- `LedoitWolf`
picks its own intensity from the data, which would make
`RiskParams.covariance_shrinkage` dead configuration. Since the project's
whole point of centralizing parameters in `config.py` is to make
"what happens if I change the shrinkage intensity?" a one-line edit, the
fixed-intensity version is used, shrinking toward the diagonal target
(off-diagonal covariances shrunk to 0, i.e. correlations shrunk toward
0; variances on the diagonal are left unchanged since they are the
better-estimated component).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def sample_covariance(returns: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Rolling sample covariance matrix estimated on the last
    ``lookback_days`` observations of ``returns``. Returns an annualized
    covariance matrix (daily cov * 252).
    """
    window = returns.tail(lookback_days)
    if len(window) < 2:
        raise ValueError(
            f"sample_covariance: need at least 2 observations, got {len(window)}"
        )
    if len(window) < lookback_days:
        logger.warning(
            "sample_covariance: only %d/%d observations available for the "
            "requested lookback window",
            len(window), lookback_days,
        )

    daily_cov = window.cov()
    return daily_cov * TRADING_DAYS_PER_YEAR


def shrunk_covariance(returns: pd.DataFrame, lookback_days: int, shrinkage: float) -> pd.DataFrame:
    """Fixed-intensity shrinkage of the sample covariance toward its
    diagonal (variances kept as-is, covariances/correlations shrunk
    toward 0), annualized.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")

    sample_cov = sample_covariance(returns, lookback_days)
    target = pd.DataFrame(
        np.diag(np.diag(sample_cov.values)),
        index=sample_cov.index,
        columns=sample_cov.columns,
    )
    return (1.0 - shrinkage) * sample_cov + shrinkage * target


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """sigma_p = sqrt(w' Sigma w)."""
    weights = np.asarray(weights, dtype=float)
    cov = np.asarray(cov, dtype=float)
    variance = weights @ cov @ weights
    return float(np.sqrt(max(variance, 0.0)))
