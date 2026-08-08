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
src/swme/          # the core solver (SWME/HSWME/RechargeSWME transport, coefficients
                   # engine, mesh, simulation driver, numerical schemes)
src/recharge/      # rainfall/infiltration/exfiltration extension, a sibling package to
                   # src/swme/ (imports from it, e.g. `from swme.pde import SWME1D`)
processing/        # downstream post-processing scripts for solver CSV output
                   # (not part of the installable package)
symbolic_math/     # sympy scripts for deriving the moment-basis projection tensors
                   # (A_ijk, B_ijk, C_ij, r_i, s_i, E_ij, F_ij); superseded by
                   # src/swme/coefficients.py, scheduled for deletion (RESTRUCTURE_PLAN.md
                   # Step 4)
tests/             # pytest suite (currently: coefficients engine regression tests)
```

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
# Install/sync the environment (creates .venv/, resolves from pyproject.toml + uv.lock)
uv sync

# Run the test suite
uv run pytest -q

# Run the default case (src/swme/config/config.ini)
uv run moment-sw

# List the configs shipped in src/swme/config/
uv run moment-sw --list-configs

# Run a specific case by name (or by path), with a chosen output directory
uv run moment-sw --config thesis_5p3_pulse_N1 --output-dir results/5p3

# On a headless machine, avoid plt.show() blocking/erroring:
MPLBACKEND=Agg uv run moment-sw --config thesis_5p2_horton_at_rest
```

Solver output (CSV snapshots, history, hyperbolicity diagnostics) goes to
`--output-dir`, else the config's `postprocessing/recharge_output_dir`, else
`Data-processing/Results/Recharge/`. Filenames are prefixed with the config name, so
successive cases don't overwrite each other.

### Reproducing the thesis test cases

`src/swme/config/` ships a config per test case of the thesis's Chapter 5, transcribed
from its runtime-parameter tables — `thesis_5p1_mixing_aR{0,1,2}` (§5.1, rainfall-mixing
validation), `thesis_5p2_horton_at_rest` (§5.2), `thesis_5p3_pulse_N{0,1,2}` (§5.3),
`thesis_5p4_horton_N{0,1,2}` (§5.4, periodic), `thesis_5p5_horton_N{0,1,2}` (§5.5, open
boundary), and `thesis_5p6_source_{free,active}_N{1,2}` (§5.6 ablation). Run the whole
set with:

```bash
for c in $(uv run moment-sw --list-configs | grep '^thesis_'); do
    MPLBACKEND=Agg uv run moment-sw --config "$c" --output-dir results/"$c"
done
```

The two cases with closed-form solutions in the thesis are verified to reproduce it:
§5.1 matches eq. (5.5) for all three `alpha_R` branches, and §5.2 matches eqs. (5.9)–(5.10)
(transition time t\* = 510.8 vs. the thesis's ≈511; final depth 1.1694 vs. 1.1694).

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
be available via `uv sync --group docs` then `uv run mkdocs serve` (local preview) /
`uv run mkdocs build`.

## Status

See [`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md)'s "Execution checklist" section for the
authoritative, up-to-date list of completed vs. pending work. As of this writing: package
scaffolding, the arbitrary-N coefficient engine, and the `swme`/`recharge` sibling-package
rename (with `matlab/` deleted and `config/` cleaned up) are done. Still pending: deleting
the remaining hardcoded per-order models, topography/well-balancing, wet-dry treatment,
the YAML-driven CLI rewrite, and the documentation site.
