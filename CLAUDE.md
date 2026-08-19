# CLAUDE.md

Context and working instructions for Claude Code (or any AI assistant)
picking up work in this repository. Read this before making changes —
it captures decisions already made and the reasoning behind them, so
they don't get silently re-litigated or reversed.

## What this project is

A portfolio piece built for internship applications in **quant risk
analysis at an asset management firm**, specifically targeting roles
touching **risk parity / ERC (Equal Risk Contribution)** and
**component risk decomposition**. The audience is a recruiter or
technical interviewer who may skim the README, may open the showcase
notebook, and may actually clone and run the code.

What it needs to demonstrate, in priority order:
1. **Mathematical rigor** — correct use of Euler's homogeneous-function
   decomposition, not just calling an optimization library and hoping
   the output looks right.
2. **Awareness of a real limitation in standard practice** — volatility-based
   risk parity assumes risk scales with variance/covariance; it doesn't
   account for differing tail behavior (skew, kurtosis) across assets.
   This project's central question: does equalizing **Component CVaR**
   contributions instead produce a materially different, more
   tail-robust portfolio, and when does the difference matter most?
3. **Software engineering habits a risk desk would expect**: tests that
   check mathematical invariants (not smoke tests), reproducibility
   (one command rebuilds every result), clean separation of concerns,
   CI.

Do not let scope creep dilute any of the three. A narrow, correct,
well-tested engine beats a broad, shallow tour of risk models — that
trade-off was made deliberately (see "Scope decision" below), not by
default.

## Scope & constraints

- Time budget for the core deliverable: **1-2 weekends**. Anything
  beyond the core ERC + Component CVaR engine, its validation, and a
  basic backtest is a stretch goal, not a requirement. Don't silently
  expand scope (e.g. adding new asset classes, new risk measures, a
  web dashboard) without flagging that it's beyond the agreed core.
- Data source: **free, reproducible data only** (`yfinance` on liquid
  ETFs). Anyone cloning the repo must be able to run
  `pip install -r requirements.txt && python scripts/run_pipeline.py`
  and get results, with no paid data-vendor dependency. If a paid
  source is ever added, it must be optional, with the free path as the
  default.
- Long-only, fully-invested portfolios (`sum(w) = 1`, `w >= 0`)
  throughout, unless a task explicitly asks to relax that.

## Methodology (do not change without updating this file + README)

**Euler risk decomposition.** For any risk measure `rho(w)` that is
positively homogeneous of degree 1 in the weights (true of both
portfolio volatility and portfolio CVaR), Euler's theorem gives an
exact decomposition:

```
rho(w) = sum_i w_i * d(rho)/d(w_i) = sum_i RC_i
```

`RC_i` is asset i's risk contribution; risk parity / ERC seeks weights
such that all `RC_i` are equal. This identity is the backbone of the
whole project and must remain unit-tested directly
(`tests/test_contributions.py`) — if a change breaks that invariant,
that's a correctness bug, not a tolerance to loosen.

**Volatility-ERC.** Closed form: `sigma_p(w) = sqrt(w' Sigma w)`,
`MRC_i = (Sigma w)_i / sigma_p(w)`. Convex reformulation via log-barrier
(Maillard, Roncalli & Teïletche, 2010) gives a unique solution — this
is the "easy" side of the project and should stay simple and fully
tested as the reference pattern for code style.

**Component-CVaR parity.** No closed form in `(w, Sigma)`. CVaR is
estimated from historical/simulated return scenarios; marginal CVaR
contribution for asset i is approximately the average return of asset i
across the scenarios in the tail beyond `VaR_alpha` (Boudt, Peterson &
Carl, 2013), sign-flipped to a loss. Equalizing these contributions is
generally **non-convex** — solved via a cyclical/iterative update
analogous to the vol-ERC algorithm. This is the harder, more
differentiating part of the project. Convergence is **not** assumed;
`solve_cvar_parity` must return its convergence history (dispersion of
`RC_i` per iteration) so it can be inspected and plotted, and any
non-convergence or instability found should be reported honestly in the
README, not hidden or handwaved.

Full rationale for these choices lives in `PROJECT_BRIEF.md` — treat
that file as the design log; update it (don't just overwrite) when a
significant methodological decision changes.

## Repository structure and the reasoning behind it

```
riskparity/            # installable package -- ALL reusable logic lives here
    data/               # download & clean price data (only place that talks to yfinance)
    risk/               # volatility, VaR/CVaR, Euler risk contributions
    optimization/       # vol-ERC and CVaR-parity solvers
    backtest/           # rolling backtest engine + known stress windows
    analysis/           # cross-strategy comparison & plotting
    config.py           # universe & every tunable parameter, in one place

tests/                  # pytest -- validate invariants, not just "does it run"
notebooks/              # 01_exploration.ipynb (scratch, not load-bearing)
                         # 02_results_walkthrough.ipynb (showcase narrative)
data/                   # raw/processed price cache -- gitignored, regenerable
results/                # figures/ and tables/ produced by the pipeline
docs/notes.tex          # living LaTeX notebook -- see "Project notebook" below
scripts/run_pipeline.py # single entry point, thin orchestration only
```

Important, already-made decision: **do not prefix package folders with
numbers** (`01_data/`, `02_risk/`, ...). An earlier draft did this and
it was rejected because Python module/package names cannot start with a
digit — it forces `sys.path`/`importlib` workarounds for any
cross-module import. Pipeline order is expressed in
`scripts/run_pipeline.py` and in this document, not in folder names.

Other conventions to preserve:
- `scripts/` stays thin — orchestration calls into `riskparity.*` only.
  If real computation starts accumulating in a script, move it into a
  module.
- `riskparity/config.py` is the only place hardcoded parameters
  (lookback windows, confidence levels, tickers, rebalance frequency)
  should live. Don't reintroduce magic numbers inside individual
  modules.
- `notebooks/01_exploration.ipynb` is scratch space; nothing there
  should be the sole source of a result used in the README. Anything
  that matters belongs in `riskparity/` and is called from the
  notebook, not duplicated in it.
- Data under `data/raw/` and `data/processed/` is gitignored — it's
  regenerated by `scripts/run_pipeline.py`, never committed.

## Project notebook (`docs/notes.tex`)

`docs/notes.tex` is a living LaTeX document (in English, for consistency
with the rest of the repo) that is the permanent record of the finance
concepts, mathematical derivations, and Python implementation notes for
this project — it is where the actual *content* lives, as opposed to
`PROJECT_BRIEF.md` (why the project is scoped/structured this way) and
this file (how to work in the repo). Compile with
`latexmk -pdf docs/notes.tex` (MiKTeX is installed); it must always
compile cleanly.

Rules for maintaining it:
- **Update it in the same change as the code it documents**, not as a
  separate pass. When a stub in `riskparity/` gets implemented, add its
  formula, reasoning, and a short annotated code excerpt to the matching
  subsection of `docs/notes.tex`'s "Implementation Log" section — do not
  leave a module documented only in code-comment docstrings.
- Figures are pulled from `results/figures/` via `\IfFileExists` guards,
  so the document compiles at every stage of the project, before and
  after the pipeline produces output — don't remove that guard pattern
  when adding a new figure.
- Derivations should be self-contained (a reader shouldn't need the cited
  paper to follow the math), matching the level of rigor already in the
  module docstrings.
- Only `docs/notes.tex` and the compiled `docs/notes.pdf` are committed;
  intermediate LaTeX build artifacts (`.aux`, `.log`, `.out`, `.toc`,
  `.fls`, `.fdb_latexmk`, `.synctex.gz`) are gitignored.
- **Every question the user asks in a session must be logged**, in the
  same change, as a new entry in `docs/notes.tex`'s "Session Q&A Log"
  section (`\label{sec:qa-log}`) — question (translated to English if
  asked in another language) plus the answer given, appended
  chronologically. This is append-only: don't remove or silently rewrite
  a past entry, even if a later answer supersedes it. Purely
  conversational chit-chat with no lasting informational content can be
  skipped, but default to logging when in doubt — the log exists so
  reasoning and context survive past the chat window they happened in.

## Testing philosophy

Tests exist primarily to validate **mathematical correctness**, in this
order of priority:
1. Euler's identity holds (`sum_i RC_i == rho(w)`) for arbitrary valid
   weights and covariance/return data — already implemented and passing
   for volatility in `tests/test_contributions.py`.
2. Known closed-form cases match (e.g. 2 uncorrelated assets → ERC
   weights proportional to `1/sigma_i`).
3. Edge cases behave sanely (concentrated portfolios, etc.).
4. Once solvers exist: contributions are actually equal at the reported
   optimum, within a **documented and justified** tolerance — CVaR's
   scenario-based estimator is noisier than the closed-form volatility
   case, so its tolerance should be looser, and that choice should be
   explained in a comment, not silently picked to make a test pass.

When implementing a new module, write its test(s) before or alongside
the implementation, following the pattern in
`tests/test_contributions.py::test_volatility_contributions_sum_to_total_risk`.
Do not mark a task as done if a test is skipped or failing — either fix
it or explicitly flag the gap.

## Current implementation status

Fully implemented and tested:
- `riskparity.risk.contributions.volatility_risk_contributions` — the
  reference pattern (docstring style, type hints, invariant-checking
  tests) for everything else in the codebase.

Everything else is a stub (`raise NotImplementedError`) with a detailed
docstring describing what it must do, the relevant formula/paper, and
known trade-offs to document rather than hide. See each module's
docstring before implementing it — the design intent is already spelled
out; don't redesign from scratch without a reason.

## Suggested build order

1. `riskparity/risk/volatility.py` — sample & shrunk covariance.
2. `riskparity/optimization/erc.py` — vol-ERC solver; validate against
   `tests/test_erc.py` TODOs (equal contributions at optimum, matches
   closed-form 2-asset case).
3. `riskparity/risk/tail_risk.py` — historical & Cornish-Fisher CVaR.
4. `riskparity/risk/contributions.py::cvar_risk_contributions` —
   scenario-based Component CVaR.
5. `riskparity/optimization/cvar_parity.py` — CVaR-parity solver +
   convergence tracking; validate against `tests/test_cvar_parity.py`
   TODOs, including the skewed-universe divergence test (the central
   empirical claim of the project).
6. `riskparity/data/download.py` + `clean.py` — real data plumbing (can
   be done in parallel with 1-5 using synthetic data for testing).
7. `riskparity/backtest/engine.py` — rolling backtest, no look-ahead.
8. `riskparity/analysis/compare.py` + `plots.py` — metrics and figures.
9. Wire it all together in `scripts/run_pipeline.py`.
10. Fill in the "Key results" section of `README.md` with real numbers
    and figures — this is what a recruiter reads first, don't leave it
    as a placeholder in the final version.

## References

- Maillard, S., Roncalli, T., & Teïletche, J. (2010). *The Properties of
  Equally Weighted Risk Contribution Portfolios.*
- Boudt, K., Peterson, B., & Carl, P. (2013). *Component VaR and CVaR.*
- Rockafellar, R. T., & Uryasev, S. (2000). *Optimization of Conditional
  Value-at-Risk.*
- Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting.*

## Notes for whoever (human or AI) picks this up next

- Don't silently loosen test tolerances or remove a `pytest.skip` by
  faking a pass — implement the underlying function.
- Don't reintroduce numbered folder prefixes.
- Don't add a paid data dependency as the default path.
- Don't implement a module without also updating `docs/notes.tex`'s
  Implementation Log for it — code and notebook stay in sync.
- When a design decision changes from what's written here or in
  `PROJECT_BRIEF.md`, update those files in the same change — they are
  meant to stay accurate, not archival.