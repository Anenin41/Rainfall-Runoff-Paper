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
                   # engine, mesh, bed topography, simulation driver, numerical schemes)
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
                   # engine, plus the topography/well-balancing suite)
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

### Bottom topography and well-balancing

A non-flat bed `Z(x)` is opt-in per config via a `[topography]` section. It is coupled into
the scheme by augmenting the path-conservative state to `W = (U, Z)` and widening the
system matrix, so the bed-slope non-conservative product goes through the same
Castro–Parés machinery already used for the moment transport — no separate hydrostatic
reconstruction step.

```ini
[pde_information]
initialCondition = lakeAtRest        # or perturbedLakeAtRest

[topography]
bed_profile = gaussian_bump          # flat | linear_slope | gaussian_bump |
                                     # parabolic_bump | sinusoidal | step | tanh_step
amplitude = 0.4                      # remaining keys are the profile's own parameters
center = 0.5
width = 0.1
reference_water_level = 2.0          # free surface H of a lake at rest: h(x) = H - Z(x)
```

```bash
uv run moment-sw --config topography_lake_at_rest      # the C-property benchmark
uv run moment-sw --config topography_perturbed_lake    # a small perturbation of it
```

The scheme is verified well balanced: a lake at rest over a Gaussian bump, a parabolic
bump, a linear slope, a sinusoid, or a *discontinuous* step stays at rest to machine
precision (`|h+Z-H| <= 2.2e-16` after ~1400 steps at N=2), for SWME, HSWME and
RechargeSWME alike — see `tests/test_topography.py`.

Two things worth knowing:

- **Use `pvm = Roe` (or `Osher`) with topography.** Well-balancing needs the scheme's
  numerical viscosity to annihilate the equilibrium jump, which holds exactly when its
  viscosity polynomial satisfies `P(0) = 0`. Roe and Osher qualify; `LF` and `PRICE` do
  not, and leave an `O(dx/dt · dh)` residual at rest. The solver emits a `RuntimeWarning`
  if you pair topography with one of them. All the thesis configs already use Roe.
- **A config without a `[topography]` section is completely unaffected**, and so is one
  with `bed_profile = flat`: an everywhere-zero bed leaves the solver on its original,
  non-augmented code path, verified bit-identical by test and by the reference run.

Dry cells are not supported yet, so the bed must stay below the water everywhere — a
reference level below the bed's high point fails loudly at setup rather than mid-run. See
`RESTRUCTURE_PLAN.md` Step 6.

### Hyperbolicity: SWME vs. HSWME

The plain SWME transport matrix is not globally hyperbolic. Its eigenvalues can leave the
real axis, and where they do the initial-value problem is ill posed — the solver will still
happily produce output. What is and isn't safe:

- **N = 0 and N = 1 are unconditionally hyperbolic.** The N=1 spectrum is exactly
  `{u_m, u_m ± sqrt(g·h + a₁²)}`, real for every state. At these orders HSWME *is* SWME.
- **N ≥ 2 can lose it.** The affected fraction of state space grows quickly with order
  (≈3 % at N=2, ≈21 % at N=4, ≈48 % at N=6 under uniform sampling).
- **HSWME (`pde_type = HSWME1D`, or `hyperbolic = True`) never loses it** at any order
  tested, while keeping the outer wave speeds `u_m ± sqrt(g·h + a₁²)`.

Hyperbolicity depends *only* on the scaled moments `aᵢ/sqrt(g·h)` — not on `u_m`, and not
on `h` and `aᵢ` separately. Note that at N=2 the unstable set is a narrow **wedge** in
slope, `|a₂/a₁| ∈ [1.14, 1.40]`, not a magnitude threshold: `a = (1.5, 1.8)` is unstable
while the larger `a = (2.0, 3.0)` is fine. "Keep the moments small" is the wrong criterion;
the *ratio* is what matters.

Every run now reports loss automatically — the schemes that eigendecompose the transport
matrix (Roe, Osher) track it for free and `run_simulation` emits a `RuntimeWarning` naming
the count and the worst `|Im(λ)|`. For a detailed per-cell log, set
`store_hyperbolicity = True` under `[postprocessing]`.

All 20 thesis Chapter 5 runs were audited state by state (14.4 M states): **zero
hyperbolicity loss**, with structural margin — their initial conditions set
`a₂ = -0.5·a₁`, a ray outside the unstable wedge at any magnitude.

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
rename, deletion of the out-of-scope legacy models, and well-balanced bottom topography
are all done, and the restructure has been validated against the thesis's own results
(§5.1/§5.2 reproduce the closed-form solutions; §5.3-§5.6 reproduced figure-by-figure via
`scripts/run_thesis_configs.sh` + `processing/*.py`). Still pending: wet-dry treatment,
the YAML-driven CLI rewrite, and the documentation site.
