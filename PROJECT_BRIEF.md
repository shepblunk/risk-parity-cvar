# Project brief — design notes

This file exists so the reasoning behind the project's scope and
structure is written down somewhere, not just in a chat history. Useful
both for you (picking the project back up after a break) and for a
recruiter/interviewer who wants to see how you think, not just the
final code.

## Target audience / goal

Portfolio piece for internship applications in **quant risk analysis at
an asset management firm**, specifically around **risk parity / ERC**
and **component risk decomposition**. The project should demonstrate:
mathematical rigor (not just calling a library), awareness of where a
standard technique (vol-based risk parity) has known limitations, and
software engineering habits (tests, reproducibility, clean structure)
that a risk desk would expect from production-adjacent code.

## Scope decision

Chose a **narrow, deep** scope over a broad one: a correctly implemented
and validated ERC + Component CVaR engine, rather than a wide but
shallow tour of many risk models. Rationale: a small number of
recruiters reviewing GitHub profiles will actually run the code or read
closely — depth and correctness on a focused topic reads better than
breadth. Time budget: 1-2 weekends for the core engine + validation +
a basic backtest.

## Why compare vol-ERC vs. CVaR-parity specifically

Vol-based risk parity is the industry-standard formulation but assumes
risk contribution scales with variance/covariance. It doesn't account
for skew or tail fatness differing across assets. Component CVaR
parity is the natural tail-aware alternative. Comparing the two head to
head — and being honest in the README about when they agree and when
they diverge — is a more interesting (and more defensible in an
interview) story than implementing either one in isolation.

## Why volatility-ERC is easy and CVaR-parity is hard

Volatility is a quadratic form in the weights, `sigma_p(w) = sqrt(w'
Sigma w)`, so its Euler risk contributions have a closed-form gradient
and the equal-risk-contribution problem has a well-known convex
reformulation (log-barrier trick, Maillard/Roncalli/Teïletche 2010) with
a unique solution.

CVaR is estimated from a scenario set (historical or simulated returns)
and has no equivalent closed form in `(w, Sigma)`. Its risk
contributions must be estimated from the tail scenarios directly, and
equalizing them is generally a non-convex problem. This project uses a
cyclical/iterative update (analogous in spirit to the vol-ERC algorithm)
and explicitly tracks and reports convergence, rather than assuming it.
This distinction — and being upfront about the non-convexity rather
than glossing over it — is one of the things meant to differentiate
this project from a superficial "risk parity in Python" tutorial clone.

## Structure decisions (and mistakes avoided)

An earlier draft of the structure used numbered folder prefixes
(`01_data/`, `02_risk/`, ...) mirroring the pipeline order. Dropped that
because Python module/package names can't start with a digit — it would
have forced ugly `sys.path`/`importlib` workarounds for any cross-module
import (e.g. `optimization/erc.py` needing something from `risk/
contributions.py`). Pipeline order is documented instead in this file
and in `scripts/run_pipeline.py`, which is the single script that runs
the stages in order — the *code* doesn't need to encode the order in
its folder names.

Kept a clean separation between:
- `riskparity/` — the installable, importable package (the actual
  logic, reusable and testable)
- `scripts/` — thin orchestration only
- `notebooks/` — exploration (`01_exploration.ipynb`, not load-bearing)
  vs. the showcase narrative (`02_results_walkthrough.ipynb`, the one
  link to send a recruiter)
- `tests/` — promoted to a top-level directory, not nested inside a
  numbered folder, because validating the Euler decomposition and the
  ERC/CVaR-parity optimum is one of the main things meant to
  differentiate this project

## Data

Free data via `yfinance` on liquid ETFs (SPY, TLT, IEF, GLD, DBC, VNQ by
default — see `riskparity/config.py`) rather than a paid vendor, so
anyone cloning the repo can reproduce every result with
`pip install -r requirements.txt && python scripts/run_pipeline.py`
and no data-access barrier.

## Open questions / things to revisit

- Exact confidence level for CVaR (currently `0.975` in
  `RiskParams.cvar_confidence`) — worth a short sensitivity check.
- Whether to also implement the Boudt/Peterson/Carl analytical Component
  CVaR formula as a cross-check against the scenario-based estimator
  (stretch goal, not required for the core deliverable).
- Whether unequal risk budgets are worth demonstrating (both solvers
  are structured to accept a `risk_budgets` argument, defaulting to
  1/N, so this is a small extension if there's time).
