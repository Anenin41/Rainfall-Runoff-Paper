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
scripts/           # repo-level utility scripts, e.g. run_thesis_configs.sh
processing/        # downstream post-processing/plotting scripts that read solver CSV
                   # output from results/ and reproduce the thesis's Chapter 5 figures
                   # (not part of the installable package)
results/           # solver output (gitignored) - CSVs and figures, organized to match
                   # what processing/*.py expect; see "Reproducing the thesis test
                   # cases" below
tests/             # pytest suite (regression tests for the coefficients/pde/source-terms
                   # engine)
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
`Data-processing/Results/Recharge/`. Filenames follow the original naming
(`recharge_{swme,hswme}_N{order}_{infiltration}_*.csv`) with **no** run-identifying
prefix, since `processing/*.py`'s comparisons expect exactly that naming inside a
per-case subfolder — give each run its own `--output-dir` (as
`scripts/run_thesis_configs.sh` does below) rather than reusing one directory across
configs, or later runs will silently overwrite earlier ones.

### Reproducing the thesis test cases

`src/swme/config/` ships a config per test case of the thesis's Chapter 5, transcribed
from its runtime-parameter tables — `thesis_5p1_mixing_aR{0,1,2}` (§5.1, rainfall-mixing
validation), `thesis_5p2_horton_at_rest` (§5.2), `thesis_5p3_pulse_N{0,1,2}` (§5.3),
`thesis_5p4_horton[_aggressive]_N{0,1,2}` (§5.4, periodic, mild and aggressive pulses),
`thesis_5p5_horton_N{0,1,2}` (§5.5, open boundary), and
`thesis_5p6_source_{free,active}_N{1,2}` (§5.6 ablation) — 20 configs in total.

Run all of them, laid out under `results/` exactly as `processing/*.py` expects (see
below), with:

```bash
scripts/run_thesis_configs.sh          # sequential
scripts/run_thesis_configs.sh -j 4     # 4 in parallel
scripts/run_thesis_configs.sh 5p3      # only configs whose name contains "5p3"
```

The two cases with closed-form solutions in the thesis are verified to reproduce it:
§5.1 matches eq. (5.5) for all three `alpha_R` branches, and §5.2 matches eqs. (5.9)–(5.10)
(transition time t\* = 510.8 vs. the thesis's ≈511; final depth 1.1694 vs. 1.1694).

### Reproducing the thesis figures

`processing/` holds the plotting scripts originally used to generate the thesis's
Chapter 5 figures, updated to read from `results/` instead of their original
absolute paths. Each one corresponds to specific figures/sections (see each script's
module docstring for the exact figure list); run after `run_thesis_configs.sh`:

| Script | Thesis section | Reads |
|---|---|---|
| `ersoy_alpha_comparison.py` | §5.1 | `results/Ersoy/ErsoyData{0,1,2}/` |
| `plotter.py` | any single run (default: §5.2) | driven by `processing/config.ini` |
| `non_wrapping_pulse_model_comparison.py`, `plot_non_wrapping_zoom_profiles.py` | §5.3 | `results/Non_Wrapping_Pulse/` |
| `smooth_pulse_model_comparison_cases.py` | §5.4 | `results/Smooth_Pulse/` |
| `inflow_outflow_comparison.py` | §5.5 | `results/Smooth_Pulse_Inflow_Outflow/` |
| `dry_wet_ablation_comparison.py`, `zoomed_dry_wet_comparison.py` | §5.6 | `results/Dry_Wet_Test/` (`Dry`/`Wet` = source-free/source-active, a pre-existing naming choice unrelated to actual dry-cell numerics, which don't exist yet - see RESTRUCTURE_PLAN.md Step 6) |

```bash
cd processing
MPLBACKEND=Agg uv run python ersoy_alpha_comparison.py
MPLBACKEND=Agg uv run python plotter.py
MPLBACKEND=Agg uv run python non_wrapping_pulse_model_comparison.py
# ... etc; figures are written under results/<Section>/..._Figures/
```

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
authoritative, up-to-date list of completed vs. pending work. As of this writing:
scaffolding, the arbitrary-N coefficient engine, the `swme`/`recharge` sibling-package
rename, and deletion of the out-of-scope legacy models are all done, and the restructure
has been validated against the thesis's own results (§5.1/§5.2 reproduce the closed-form
solutions; §5.3-§5.6 reproducible via `scripts/run_thesis_configs.sh` +
`processing/*.py`). Still pending: bottom topography, well-balancing, wet-dry treatment,
the YAML-driven CLI rewrite, and the documentation site.
