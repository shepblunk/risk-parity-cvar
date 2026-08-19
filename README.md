# Risk Parity / ERC vs. Component CVaR Parity

A research project comparing **volatility-based risk parity (ERC)** to
**tail-risk-based parity using Component CVaR**, on a multi-asset ETF
universe (equities, sovereign bonds, gold, commodities, REITs).

> **Status: work in progress.** This README describes the intended
> methodology and final structure of the project. See
> [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) for the design rationale.

## Motivation

Volatility-based risk parity treats risk as symmetric and Gaussian: an
asset's contribution to portfolio risk depends only on its variance and
covariances. In practice, tail risk is not proportional to variance —
some assets have fatter left tails than their volatility alone
suggests (equities vs. long-duration bonds, for instance). This project
asks a concrete question: **does equalizing Component CVaR contributions
instead of volatility contributions produce a materially different,
more tail-robust portfolio — and if so, when does it matter most?**

## Methodology

1. **Risk decomposition (Euler's theorem).** For any risk measure that
   is positively homogeneous of degree 1 in the portfolio weights
   (true of both volatility and CVaR), the total risk decomposes
   exactly into a sum of per-asset contributions:
   `rho(w) = sum_i w_i * d(rho)/d(w_i)`. This identity is the backbone
   of the whole project and is unit-tested directly
   (`tests/test_contributions.py`).

2. **Volatility-ERC.** Closed-form marginal contributions from the
   covariance matrix; weights solved so all contributions are equal
   (Maillard, Roncalli & Teïletche, 2010).

3. **Component-CVaR parity.** Marginal CVaR contributions estimated
   from historical/simulated return scenarios in the tail beyond
   VaR_alpha (Boudt, Peterson & Carl, 2013); weights solved via a
   cyclical update analogous to the vol-ERC algorithm (no closed form
   exists here — see [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) for why
   this is harder and how convergence is validated).

4. **Backtest.** Rolling out-of-sample comparison of Equal-Weight,
   Min-Variance, Vol-ERC and CVaR-Parity, with particular attention to
   known stress windows (2008 GFC, Mar 2020, 2022 rate shock) and to
   how stable each portfolio's risk contributions stay *between*
   rebalances.

## Repository structure

```
riskparity/            # installable package -- all reusable logic
    data/               # download & clean price data
    risk/               # volatility, VaR/CVaR, Euler risk contributions
    optimization/       # vol-ERC and CVaR-parity solvers
    backtest/           # rolling backtest engine + stress windows
    analysis/           # cross-strategy comparison & plotting
    config.py           # universe & all tunable parameters, in one place

tests/                  # pytest — mathematical invariants, not just smoke tests
notebooks/              # 01 exploration (scratch) / 02 results (showcase)
data/                   # raw/processed price cache (gitignored, regenerable)
results/                # figures/ and tables/ produced by the pipeline
scripts/run_pipeline.py # single entry point: reproduces every result
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                        # verify the math (Euler identity, known cases)
python scripts/run_pipeline.py
```

Results (figures + tables) are written to `results/`. The full narrative
is in `notebooks/02_results_walkthrough.ipynb`.

## Key results

*(to fill in once the pipeline runs end to end — this is the section a
recruiter reads first, so it should lead with 2-3 concrete numbers/plots,
not just "see notebook")*

## References

- Maillard, S., Roncalli, T., & Teïletche, J. (2010). *The Properties of
  Equally Weighted Risk Contribution Portfolios.*
- Boudt, K., Peterson, B., & Carl, P. (2013). *Component VaR and CVaR.*
- Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of Conditional
  Value-at-Risk.*
- Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting.*

## License

MIT — see [LICENSE](LICENSE).
