# Rainfall-Runoff Paper

Research repository for a rainfall-runoff extension of the Shallow Water Moment (SWME)
framework: a 1D finite-volume solver for Shallow Water Moment Equations (SWME), their
hyperbolic-regularized variant (HSWME), and a rainfall/infiltration/exfiltration extension
(RechargeSWME), plus the symbolic derivations and post-processing that support the
accompanying thesis.

**This repository is under active restructuring.** See
[`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md) for the full design (arbitrary-N moment
support, well-balanced topography, wet-dry treatment, package layout) and a checklist of
what's done vs. still pending — read it before making structural changes, and keep it
updated as work lands.

## Repository layout

```text
src/moment_sw/     # the solver (SWME/HSWME/RechargeSWME); being renamed/split into
                   # src/swme/ + src/recharge/ per RESTRUCTURE_PLAN.md Step 1.5
processing/        # downstream post-processing scripts for solver CSV output
                   # (not part of the installable package)
matlab/            # legacy MATLAB reference implementation — scheduled for deletion,
                   # see RESTRUCTURE_PLAN.md decision #4
symbolic_math/     # sympy scripts for deriving the moment-basis projection tensors
                   # (A_ijk, B_ijk, C_ij, r_i, s_i, E_ij, F_ij); superseded by
                   # src/moment_sw/coefficients.py and scheduled for deletion
tests/             # pytest suite (currently: coefficients engine regression tests)
```

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
# Install/sync the environment (creates .venv/, resolves from pyproject.toml + uv.lock)
uv sync

# Run the test suite
uv run pytest -q

# Run a solver case from the config-driven entry point
# (the active config lives at src/moment_sw/config/config.txt)
uv run python -m moment_sw.main

# On a headless machine, avoid plt.show() blocking/erroring:
MPLBACKEND=Agg uv run python -m moment_sw.main
```

Solver output (CSV snapshots, history, hyperbolicity diagnostics) is written to
`Data-processing/Results/Recharge/` relative to wherever the command above is run from
(typically the repo root).

## Development

```bash
# Run a single test file / a keyword-matched subset
uv run pytest tests/test_coefficients.py -q
uv run pytest -k "closed_form" -q

# Add a new runtime dependency
uv add <package>

# Add a dev-only dependency
uv add --group dev <package>
```

Once [`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md) Step 9 lands, a documentation site will
be available via `uv run mkdocs serve` (local preview) / `uv run mkdocs build`.

## Status

See [`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md)'s "Execution checklist" section for the
authoritative, up-to-date list of completed vs. pending work. As of this writing: package
scaffolding and the arbitrary-N coefficient engine are done; the package rename
(`moment_sw` → `swme` + sibling `recharge`), YAML config migration, `matlab/` deletion,
hardcoded-model deletion, topography/well-balancing, wet-dry treatment, the CLI rewrite,
and the documentation site are still pending.
