# Quick start

## Install

The project uses [uv](https://docs.astral.sh/uv/) for dependency management. One
command creates the virtual environment and installs everything from the lock file:

```bash
uv sync
```

To include the tools that build this documentation:

```bash
uv sync --group docs
```

Check the install by running the test suite:

```bash
uv run pytest -q
```

## See it working

Three shipped cases exist to be run and looked at. Between them they cover most of the
solver, and each one's header comment says what to look for and quotes the numbers the
run should reproduce.

```bash
scripts/run_smoke_tests.sh --report     # all three in parallel, ~2 min, PDFs included
```

| Case | What it demonstrates |
|:--|:--|
| `smoke_test_1` | Rainfall with Horton infiltration over a non-flat bed, HSWME at \(N = 3\). The ponded bed drains into the ground while the Horton capacity exceeds the rain, so mean depth falls until \(f_p = R\), then rises |
| `smoke_test_2` | A dam break onto exactly dry ground, with stiff friction integrated implicitly. Mass is conserved to \(2.2 \times 10^{-16}\) and no depth goes negative |
| `smoke_test_3` | Five moments under rainfall and exfiltration, and the SWME-versus-HSWME hyperbolicity contrast, which is one line of the config apart |

The three reports together contain every page the [report generator](reports.md) can
produce. If something in this package is broken, one of them will say so.

## Run a case

The solver is driven by YAML config files. A set of them ships inside the package.

```bash
# List the configs that come with the package
uv run experiment --list-configs

# Run one by name
uv run experiment --config wetdry_dam_break

# Run one by path, into a directory you choose
uv run experiment --config path/to/case.yaml --output-dir results/my_case
```

With no `--config`, the default case in `src/swme/config/config.yaml` runs.

Results go to `results/<config-name>/` unless you say otherwise. Each run gets its own
directory because every run writes the same filenames, so sharing a directory means the
second run silently overwrites the first.

### Useful flags

| Flag | Effect |
|:--|:--|
| `--output-dir DIR` | Write results to `DIR` |
| `--verbose` | Print progress every timestep. Off by default, because it is four lines per step |
| `--report` | Also write a multi-page PDF report to `<output-dir>/report.pdf` |
| `--plot` | Show an interactive summary window at the end. It blocks until you close it |
| `--list-configs` | Print the shipped config names and exit |

Both `--report` and `--plot` are off by default. A plain run does not import
matplotlib at all, which keeps scripted runs and batch sweeps fast.

## What a run writes

For a run with moment order \(N\), a filename stem is built from the model and order,
for example `swme_N2` or `recharge_hswme_N1_horton`.

| File | Contents |
|:--|:--|
| `<stem>_final.csv` | The final state, one row per cell: `x, h, u_m, a1, ..., aN` |
| `<stem>_summary_history.csv` | Per-step spatial statistics: means, minima, maxima |
| `<stem>_field_history.csv` | The full state at every stored step. This is the large one |
| `<stem>_run.json` | What the run did: model, scheme, thresholds, bed profile, timings |
| `<stem>_hyperbolicity_*.csv` | Diagnostics, only when `store_hyperbolicity` is on |
| `<stem>_mass_budget.csv` | Total mass against sources and boundary fluxes, only when `store_mass_budget` is on |

The history files are written only when `store_history` is on, which is the default.

`<stem>_run.json` matters more than its size suggests. The CSVs record numbers but not
the settings that produced them, so without it a result read months later cannot say
which flux scheme was used or what the bed looked like. See
[Run reports](reports.md#the-metadata-sidecar).

## Build a report

```bash
# From a finished run directory
uv run report results/wetdry_dam_break

# Every run under a directory tree
uv run report results/Dry_Wet_Test
```

This reads the CSVs and writes `report.pdf` next to them. Because it works from files
rather than from a live simulation, old results can be reported on without re-running
anything. See [Run reports](reports.md).

## Explore runs interactively

```bash
uv run visual results            # every run under results/, in the browser
```

A local web app over the same files: move through time, hover for values, pick where the
vertical profile is drawn, and put several runs on the same axes. See
[Interactive viewer](viewer.md).

## Manage output directories

Runs accumulate. To see what is on disk:

```bash
uv run purge
```

This lists each run directory with its size, file count, and modification time, newest
first. Deleting requires an explicit flag and a pattern, and asks for confirmation:

```bash
uv run purge --delete "old_test_*"
uv run purge --delete "old_test_*" --yes    # skip the prompt, for scripts
```

## Write your own config

Start from a shipped config and edit it. A minimal one looks like this:

```yaml
pde:
  type: SWME1D
  initial_condition: damBreak_noVelocity
  viscosity: 0.0
  slip_length: 1.0
  linear_source: false

grid:
  x1: -1.0
  x2: 1.0
  resolution_x: 200

numerics:
  order: 2
  t_end: 0.2
  method: classical
  fvm_type: PVM
  pvm: Roe
  time_integrator: ExplicitEuler
  boundary_condition: INFLOW_OUTFLOW
```

Unknown keys are rejected with an error naming the key, so a typo tells you about
itself instead of quietly falling back to a default. Every key is listed in
[Configuration](configuration.md).

!!! warning "One YAML trap"
    PyYAML follows YAML 1.1, where a float in exponent form needs a signed exponent.
    `1.0e-3` is a number, but `1.0e3` is parsed as a **string**. Write `1.0e+3`. The
    config loader catches the common cases and reports the rule, but it is easier to
    avoid.

## Reproduce the thesis

`src/swme/config/` ships one config per Chapter 5 test case, transcribed from the thesis's
parameter tables. Run them all, laid out under `results/` the way `processing/` expects:

```bash
scripts/run_thesis_configs.sh          # sequential
scripts/run_thesis_configs.sh -j 4     # 4 in parallel
scripts/run_thesis_configs.sh 5p3      # only configs whose name contains "5p3"
```

Then regenerate the figures from `processing/`. Each script's docstring lists the exact
figures it produces, and output goes under `results/<Section>/..._Figures/`:

```bash
cd processing
MPLBACKEND=Agg uv run python ersoy_alpha_comparison.py
```

| Section | Configs | Plotting scripts | Results directory |
|:--|:--|:--|:--|
| 5.1 Rainfall-mixing validation | `thesis_5p1_mixing_aR{0,1,2}` | `ersoy_alpha_comparison.py` | `Ersoy/ErsoyData{0,1,2}/` |
| 5.2 Horton at rest | `thesis_5p2_horton_at_rest` | `plotter.py` (driven by `config.ini`) | `5p2_Horton_At_Rest/` |
| 5.3 Non-wrapping pulse | `thesis_5p3_pulse_N{0,1,2}` | `non_wrapping_pulse_model_comparison.py`, `plot_non_wrapping_zoom_profiles.py` | `Non_Wrapping_Pulse/` |
| 5.4 Periodic pulses | `thesis_5p4_horton[_aggressive]_N{0,1,2}` | `smooth_pulse_model_comparison_cases.py` | `Smooth_Pulse/` |
| 5.5 Open boundary | `thesis_5p5_horton_N{0,1,2}` | `inflow_outflow_comparison.py` | `Smooth_Pulse_Inflow_Outflow/` |
| 5.6 Source ablation | `thesis_5p6_source_{free,active}_N{1,2}` | `dry_wet_ablation_comparison.py`, `zoomed_dry_wet_comparison.py` | `Dry_Wet_Test/` |

In section 5.6, `Dry`/`Wet` means source-free/source-active. The name predates, and is
unrelated to, the solver's wet-dry treatment.

Sections 5.1 and 5.2 match the thesis's closed-form solutions: eq. (5.5) for all three
`alpha_R` branches, and eqs. (5.9) and (5.10) with transition time \(t^* = 510.8\)
(thesis: about 511) and final depth 1.1694. Sections 5.3 to 5.6 were reproduced figure by
figure. The Horton thesis configs are pinned to the thesis's rainfall-limited closure
with `horton_limit_by_rainfall: true`; see [Configuration](configuration.md).

The other shipped cases are the default `config`, the `topography_lake_at_rest` and
`topography_perturbed_lake` benchmarks, `wetdry_dam_break`, the three smoke tests, and
`rain_fills_lake_overtops_dam`: rain filling a lake with dry banks until it spills over
a dam, with Horton infiltration. Its header comment quotes the lake level over time,
the overtopping time, the spilled volume and the closed mass budget.

## Develop

```bash
uv run pytest tests/test_coefficients.py -q     # one file
uv run pytest -k "closed_form" -q               # a keyword subset

uv add <package>                                # runtime dependency
uv add --group dev <package>                    # dev-only dependency

uv run docs serve                               # docs preview at http://127.0.0.1:8000
uv run docs build --strict                      # fails on broken links and bad docstrings
```

Run `uv run docs build --strict` before committing documentation changes.

The repository is laid out as follows:

```text
src/swme/          core solver: transport, coefficients engine, mesh, topography,
                   wet-dry, schemes, simulation driver, CLI
src/swme/config/   shipped YAML cases
src/swme/report/   PDF run report (uv run report), built from CSVs
src/swme/viewer/   browser viewer (uv run visual, Dash + Plotly), same loader as the report
src/recharge/      rainfall/infiltration/exfiltration extension, built on swme
scripts/           run_thesis_configs.sh, run_smoke_tests.sh
processing/        thesis Chapter 5 plotting scripts (not part of the package)
docs/              documentation site sources (mkdocs.yml at the repo root)
results/           solver output (gitignored)
tests/             pytest suite
```

`RESTRUCTURE_PLAN.md` at the repository root records why each design decision was taken,
what was measured, and the defects found on the way. Read it before changing the
structure, and keep it updated as work lands.
