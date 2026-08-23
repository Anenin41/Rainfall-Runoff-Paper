# Rainfall-Runoff Paper

Research repository for a rainfall-runoff extension of the Shallow Water Moment (SWME)
framework: a 1D finite-volume solver for the Shallow Water Moment Equations (SWME), their
hyperbolic-regularized variant (HSWME), and a rainfall/infiltration/exfiltration extension
(RechargeSWME), together with the configs, post-processing and run reports that reproduce
and document the accompanying thesis's results.

The restructure that produced this layout is **complete** — arbitrary-N moment support,
well-balanced topography, wet-dry treatment, the `swme`/`recharge` package split, the YAML
CLI, an in-package PDF report generator and a documentation site.
[`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md) remains the design record: why each decision
was taken, what was measured, and the defects found on the way. Read it before making
structural changes, and keep it updated as work lands.

New here? [`docs/`](docs/) is the place to start — or run the three demonstration cases in
[Showing that it works](#showing-that-it-works-in-three-runs) and read the reports.

## Repository layout

```text
src/swme/          # the core solver (SWME/HSWME transport, coefficients engine, mesh,
                   # bed topography, wet-dry treatment, simulation driver, numerical
                   # schemes, CLI)
src/swme/config/   # the shipped YAML cases: the thesis suite, the topography and
                   # wet-dry benchmarks, and the three smoke tests
src/swme/report/   # the multi-page PDF run report (`moment-sw-report`), reading CSVs
                   # rather than a live simulation
src/recharge/      # rainfall/infiltration/exfiltration extension, a sibling package to
                   # src/swme/ (imports from it, e.g. `from swme.pde import SWME1D`)
scripts/           # repo-level runners: run_thesis_configs.sh, run_smoke_tests.sh
processing/        # downstream post-processing/plotting scripts that read solver CSV
                   # output from results/ and reproduce the thesis's Chapter 5 figures
                   # (not part of the installable package)
docs/              # the documentation site's sources (mkdocs.yml at the repo root)
results/           # solver output (gitignored) - CSVs and figures, organized to match
                   # what processing/*.py expect; see "Reproducing the thesis test
                   # cases" below
tests/             # pytest suite (regression tests for the coefficients/pde/source-terms
                   # engine, plus the topography, hyperbolicity, wet-dry, CLI, report
                   # and diagnostics suites)
```

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
# Install/sync the environment (creates .venv/, resolves from pyproject.toml + uv.lock)
uv sync

# Run the test suite (748 tests, ~4 min)
uv run pytest -q

# Run the default case (src/swme/config/config.yaml)
uv run moment-sw

# List the configs shipped in src/swme/config/
uv run moment-sw --list-configs

# Run a specific case by name (or by path), with a chosen output directory
uv run moment-sw --config thesis_5p3_pulse_N1 --output-dir results/5p3

# Also write a multi-page PDF report next to the CSVs
uv run moment-sw --config smoke_test_1 --report

# Build that report later instead, from the CSVs of a finished run - or of a
# whole tree of them - without re-running anything
uv run moment-sw-report results/smoke_test_1
uv run moment-sw-report results/Dry_Wet_Test

# Show the interactive summary figure when the run finishes (off by default,
# because it blocks until the window is closed)
uv run moment-sw --config thesis_5p2_horton_at_rest --plot

# List the run directories under results/ with their sizes; deleting needs an
# explicit pattern and a confirmation
uv run purge
```

Three commands are installed: `moment-sw` (run a case), `moment-sw-report` (render a
report from finished output) and `purge` (list and clean up run directories).

Configs are YAML, with sections `pde`, `grid`, `numerics` (required) and
`topography`, `wet_dry`, `postprocessing` (optional). Unknown sections and unknown keys
are rejected rather than silently ignored, so a typo names itself instead of quietly
falling back to a default.

Solver output (CSV snapshots, history, hyperbolicity diagnostics) goes to
`--output-dir`, else the config's `postprocessing.output_dir`, else
`results/<config-name>/` — so two runs cannot silently overwrite each other. Filenames follow the original naming
(`recharge_{swme,hswme}_N{order}_{infiltration}_*.csv`) with **no** run-identifying
prefix, since `processing/*.py`'s comparisons expect exactly that naming inside a
per-case subfolder — give each run its own `--output-dir` (as
`scripts/run_thesis_configs.sh` does below) rather than reusing one directory across
configs, or later runs will silently overwrite earlier ones.

### Showing that it works, in three runs

`src/swme/config/smoke_test_{1,2,3}.yaml` are a demonstration suite rather than a test
fixture: between them they cover most of what the solver does, and each one's header
comment says what to look for and quotes the numbers the run should reproduce.

```bash
scripts/run_smoke_tests.sh --report     # all three in parallel, ~2 min, PDFs included
```

| Run | Case | What it covers |
|:--|:--|:--|
| 1 | Rainfall-runoff over a stepped hillslope | `RechargeSWME1D` with Horton infiltration and mixing friction, HSWME at N=3, non-flat bed via the augmented `W = (U, Z)` state, Roe, explicit friction |
| 2 | Dam break onto an exactly dry bed | wet-dry engine at a genuine vacuum front, stiff `ν/h²` friction integrated implicitly (`linear_source` + `ImplicitEuler`), Osher flux, N=2 |
| 3 | N=5 moment hierarchy under recharge | arbitrary-order moments, exfiltration (`I < 0`) with both mixing terms, periodic domain, and the SWME-vs-HSWME hyperbolicity contrast |

Between them the runs produce every page of the [run report](docs/reports.md), and each
carries a checkable claim: run 1's mean depth is flat to the last digit until the Horton
ponding time `t_p = ln((f0-fc)/(R-fc))/k = 0.0788` and rises after it; run 2 conserves
mass to 2.2e-16 across a vacuum front with zero negative heights; run 3's mean depth
rises at exactly the net source rate `R - I = 0.2`, and its per-cell spectrum is real in
every one of 44480 records under HSWME against 615 genuinely non-hyperbolic cells when
the same run is repeated with the plain SWME closure.

### Reproducing the thesis test cases

`src/swme/config/` ships a config per test case of the thesis's Chapter 5, transcribed
from its runtime-parameter tables — `thesis_5p1_mixing_aR{0,1,2}` (§5.1, rainfall-mixing
validation), `thesis_5p2_horton_at_rest` (§5.2), `thesis_5p3_pulse_N{0,1,2}` (§5.3),
`thesis_5p4_horton[_aggressive]_N{0,1,2}` (§5.4, periodic, mild and aggressive pulses),
`thesis_5p5_horton_N{0,1,2}` (§5.5, open boundary), and
`thesis_5p6_source_{free,active}_N{1,2}` (§5.6 ablation) — 20 configs in total. The other
seven of the 27 shipped cases are `config` (the default), the two `topography_*`
benchmarks, `wetdry_dam_break`, and the three `smoke_test_*` demonstrations above.

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
| `dry_wet_ablation_comparison.py`, `zoomed_dry_wet_comparison.py` | §5.6 | `results/Dry_Wet_Test/` (`Dry`/`Wet` = source-free/source-active, a pre-existing naming choice unrelated to the actual wet-dry treatment described below) |

```bash
cd processing
MPLBACKEND=Agg uv run python ersoy_alpha_comparison.py
MPLBACKEND=Agg uv run python plotter.py
MPLBACKEND=Agg uv run python non_wrapping_pulse_model_comparison.py
# ... etc; figures are written under results/<Section>/..._Figures/
```

### Bottom topography and well-balancing

A non-flat bed `Z(x)` is opt-in per config via a `topography:` section. It is coupled into
the scheme by augmenting the path-conservative state to `W = (U, Z)` and widening the
system matrix, so the bed-slope non-conservative product goes through the same
Castro–Parés machinery already used for the moment transport — no separate hydrostatic
reconstruction step.

```yaml
pde:
  initial_condition: lakeAtRest   # or perturbedLakeAtRest

topography:
  bed_profile: gaussian_bump      # flat | linear_slope | gaussian_bump |
                                  # parabolic_bump | sinusoidal | step | tanh_step
  amplitude: 0.4                  # remaining keys are the profile's own parameters
  center: 0.5
  width: 0.1
  reference_water_level: 2.0      # free surface H of a lake at rest: h(x) = H - Z(x)
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

- **Use `pvm: Roe` (or `Osher`) with topography.** Well-balancing needs the scheme's
  numerical viscosity to annihilate the equilibrium jump, which holds exactly when its
  viscosity polynomial satisfies `P(0) = 0`. Roe and Osher qualify; `LF` and `PRICE` do
  not, and leave an `O(dx/dt · dh)` residual at rest. The solver emits a `RuntimeWarning`
  if you pair topography with one of them. All the thesis configs already use Roe.
- **A config without a `topography:` section is completely unaffected**, and so is one
  with `bed_profile: flat`: an everywhere-zero bed leaves the solver on its original,
  non-augmented code path, verified bit-identical by test and by the reference run.

A bed poking above the reference water level fails loudly at setup rather than mid-run.
Dry cells themselves are supported — see "Wet and dry cells" below — but `lakeAtRest` is a
lake, and a bed above its surface means the initial condition is not the one you asked for.

### Wet and dry cells

Dry cells (`h = 0`) are supported. Before this the solver raised `RuntimeError` the instant
any height reached zero, so a dam break onto a dry bed could not be run at all:

```bash
uv run moment-sw --config wetdry_dam_break     # dam break onto an exactly dry bed
```

Three thresholds govern it, configurable per case and defaulting to values sized for the
thesis' `h ~ O(1)`:

```yaml
wet_dry:
  h_dry: 1.0e-4     # at/below: dry — no moments, velocity driven smoothly to zero
  h_wet: 1.0e-3     # at/above: ordinary wet flow, no regularization at all
  eps_div: 1.0e-14  # machine-precision division guard
```

Between them the moments ramp linearly to zero, so a vanishing film relaxes to plug flow
instead of carrying a vertical profile it cannot support. Mass is conserved exactly and the
timestep is capped so no cell can drain below zero.

Three things to know before running a drying case:

- **`h_dry` is a modelling choice, not a formality.** At a vacuum front the exact solution
  itself contains arbitrarily small depths, so a coarse `h_dry` truncates the leading edge
  and slows the front — and refining the mesh does *not* fix it. On the standard dry dam
  break at 800 cells, against an exact front speed of 2.0: `h_dry = 1e-4` gives 1.758,
  `1e-8` gives 1.934, `1e-12` gives 1.984. Pushing it down is not free either, since `h_dry`
  also floors the `ν/h²` friction term. Put it well below the smallest depth you need to
  resolve, then check `ν/h_dry²` is still sane.
- **With friction, use the implicit source path** (`linear_source: true` +
  `time_integrator: ImplicitEuler`). Navier-slip friction carries `ν/h²`, which is stiff
  near any drying front regardless of `h_dry`; explicit integration goes unstable, and the
  symptom is a negative height that looks like a positivity failure but isn't. The error
  message says so if you hit it.
- **A config that stays wet is completely unaffected** — verified bit-identical, both on the
  reference run and by byte-comparing regenerated thesis CSVs.
- **A vacuum front over a *sloping* bed is not supported.** Wet-dry and topography each
  work; together, at a genuine `h = 0` front, the run stops with a negative height. The
  bed-slope fluctuation is not positivity-preserving as `h → 0`, and it is the bed rather
  than the moment model at fault — the failure is identical at `N = 0`, i.e. for plain
  shallow water. Keep a vacuum front on a flat bed (as `wetdry_dam_break` and
  `smoke_test_2` do); see
  [What to watch out for](docs/limitations.md#a-vacuum-front-over-a-sloping-bed-is-not-supported).

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

Every run reports complex spectra automatically — the schemes that eigendecompose the
transport matrix (Roe, Osher) track it for free and `run_simulation` emits a
`RuntimeWarning` naming the count and the worst `|Im(λ)|`.

**Read that warning carefully.** It measures the scheme's *path-averaged interface* matrix,
not `A(U)` at any state. `A(U)` is nonlinear in `U`, so an average of hyperbolic matrices
need not be hyperbolic: a wet-dry front trips it even at N=0, i.e. for plain shallow water,
which is unconditionally hyperbolic. A nonzero count is therefore worth knowing but is not
by itself evidence that the model lost hyperbolicity. For that question, set
`store_hyperbolicity: true` under `postprocessing:`, which logs the spectrum of `A(U)`
cell by cell.

All 20 thesis Chapter 5 runs were audited state by state (14.4 M states): **zero
hyperbolicity loss**, with structural margin — their initial conditions set
`a₂ = -0.5·a₁`, a ray outside the unstable wedge at any magnitude.

## Development

The suite is 748 tests and takes about four minutes in full, so most of the time a file
or a keyword subset is the thing to run:

```bash
# Run a single test file / a keyword-matched subset
uv run pytest tests/test_coefficients.py -q
uv run pytest -k "closed_form" -q

# Add a new runtime dependency
uv add <package>

# Add a dev-only dependency
uv add --group dev <package>
```

## Documentation

A documentation site covers the model, the numerical method, every configuration key, the
report generator, and the measured limitations, plus an API reference generated from the
docstrings. Sources live in [`docs/`](docs/).

```bash
uv sync --group docs
uv run mkdocs serve    # local preview at http://127.0.0.1:8000
uv run mkdocs build    # render to site/ (gitignored)
```

`mkdocs build --strict` treats warnings as errors, including broken links and malformed
docstrings, so it is worth running before committing documentation changes.

## Status

Steps 0 through 9 of [`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md) are complete — see its
"Status" section for the authoritative list. In short: the solver runs entirely on the
arbitrary-N generic engine, the out-of-scope legacy models are gone, well-balanced
topography and wet-dry treatment are in, hyperbolicity is audited and reported, the CLI
takes validated YAML, runs can render their own PDF report, and the documentation site is
published from `docs/`.

The restructure is validated against the thesis's own results: §5.1 and §5.2 reproduce
their closed-form solutions, and §5.3–§5.6 were reproduced figure-by-figure via
`scripts/run_thesis_configs.sh` + `processing/*.py`.

What remains is not restructuring work but the open items in
[`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md) §7 and the measured constraints in
[What to watch out for](docs/limitations.md) — chief among them that the solver evaluates
the system matrix one cell at a time, which is the thing to fix before any large production
run.
