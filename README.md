# Rainfall-Runoff Paper

A 1D finite-volume solver for the Shallow Water Moment Equations (SWME), their
hyperbolic-regularized variant (HSWME), and a rainfall/infiltration/exfiltration extension
(RechargeSWME), together with the configs, post-processing and reports that reproduce the
accompanying thesis's results.

- **New here?** Run the [three demonstration cases](#see-it-working) and read their
  reports, then browse the documentation in [`docs/`](docs/): the
  [quick start](docs/quickstart.md), the [model](docs/model.md), the
  [numerical method](docs/numerics.md), every [configuration key](docs/configuration.md),
  and [what to watch out for](docs/limitations.md).
- **Changing the structure?** Read [`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md) first.
  It records why each design decision was taken, what was measured, and the defects found
  on the way. Keep it updated as work lands.

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                   # create .venv/ from pyproject.toml + uv.lock
uv run pytest -q                          # 803 tests, ~4 min

uv run moment-sw --list-configs           # the cases shipped in src/swme/config/
uv run moment-sw --config smoke_test_1 --report
uv run moment-sw --config thesis_5p3_pulse_N1 --output-dir results/5p3
```

Four commands are installed:

| Command | What it does |
|:--|:--|
| `moment-sw` | Run a case. `--report` also writes a PDF; `--plot` shows a summary figure at the end (blocks until closed) |
| `moment-sw-report <dir>` | Build the PDF [report](docs/reports.md) from a finished run, or from every run under a tree, without re-running |
| `moment-sw-view [dir]` | Explore finished runs in the browser. See [below](#exploring-results-in-the-browser) |
| `purge` | List run directories under `results/` with their sizes. Deleting needs an explicit pattern and a confirmation |

**Configs** are YAML with required sections `pde`, `grid`, `numerics` and optional
`topography`, `wet_dry`, `postprocessing`. Unknown sections and keys are rejected, so a
typo fails loudly instead of falling back to a default. See
[Configuration](docs/configuration.md).

**Output** goes to `--output-dir`, else the config's `postprocessing.output_dir`, else
`results/<config-name>/`. Filenames carry no run identifier
(`recharge_{swme,hswme}_N{order}_{infiltration}_*.csv`, the naming `processing/*.py`
expects), so give every run its own directory or later runs overwrite earlier ones.

## See it working

`smoke_test_{1,2,3}` are demonstration cases that between them cover most of the solver
and produce every page of the report. Each config's header comment says what to look for
and the numbers the run should reproduce.

```bash
scripts/run_smoke_tests.sh --report     # all three in parallel, ~2 min, PDFs included
```

| Run | Case | Covers | Checkable result |
|:--|:--|:--|:--|
| 1 | Rainfall-runoff over a stepped hillslope | RechargeSWME, Horton infiltration, mixing friction, HSWME at N=3, non-flat bed, Roe | Mean depth is flat until the ponding time `t_p = 0.0788`, then rises |
| 2 | Dam break onto an exactly dry bed | Wet-dry front, stiff friction integrated implicitly, Osher, N=2 | Mass conserved to 2.2e-16, no negative depths |
| 3 | N=5 moment hierarchy under recharge | Arbitrary order, exfiltration, periodic domain, SWME vs. HSWME | Mean depth rises at exactly `R - I = 0.2`; HSWME spectrum real everywhere, plain SWME is not |

## Exploring results in the browser

`moment-sw-view` opens finished runs in a local web app. It shows the same data as the
PDF report, but you can move through time, hover for values, zoom in on a front, and
overlay runs. It only reads output on disk and never re-runs anything.

```bash
uv run moment-sw-view                        # every run under results/
uv run moment-sw-view results/Dry_Wet_Test   # one family of runs, or a single run directory
```

It prints `http://localhost:8050/` and opens it in your browser.

- **Pick a run** from the list on the left. Runs are grouped by directory, and the
  filter box searches by name.
- **Move through time** with the slider or ▶ to play. Every tab follows the same time.
- **Pick a position `x`** by clicking any map or typing a value. That cell is used for
  the vertical velocity profile, the eigenvalue spectrum and the point time series.
- **Browse the tabs:** fields, histories, space-time maps, vertical profiles,
  hyperbolicity, and wet-dry and topography when the run has them.
- **Compare runs** with **Compare** at the top: add up to six runs with **+**, or a
  whole group with **compare all**. They share axes, and colour follows the moment order.

**On a remote machine,** the viewer listens on that machine only (`127.0.0.1`).
VS Code Remote-SSH forwards port 8050 to your browser automatically. From a plain SSH
session, run `ssh -L 8050:localhost:8050 <host>` and open `http://localhost:8050/`.
Pass `--no-browser` there, and `--port` if 8050 is taken.

See [Interactive viewer](docs/viewer.md) for every tab and option.

## Reproducing the thesis

`src/swme/config/` ships one config per Chapter 5 test case, transcribed from the thesis's
parameter tables. Run them all, laid out under `results/` the way `processing/` expects:

```bash
scripts/run_thesis_configs.sh          # sequential
scripts/run_thesis_configs.sh -j 4     # 4 in parallel
scripts/run_thesis_configs.sh 5p3      # only configs whose name contains "5p3"
```

Then regenerate the figures from `processing/` (each script's docstring lists the exact
figures it produces; output goes under `results/<Section>/..._Figures/`):

```bash
cd processing
MPLBACKEND=Agg uv run python ersoy_alpha_comparison.py
```

| Section | Configs | Plotting scripts | Results directory |
|:--|:--|:--|:--|
| §5.1 Rainfall-mixing validation | `thesis_5p1_mixing_aR{0,1,2}` | `ersoy_alpha_comparison.py` | `Ersoy/ErsoyData{0,1,2}/` |
| §5.2 Horton at rest | `thesis_5p2_horton_at_rest` | `plotter.py` (driven by `config.ini`) | `5p2_Horton_At_Rest/` |
| §5.3 Non-wrapping pulse | `thesis_5p3_pulse_N{0,1,2}` | `non_wrapping_pulse_model_comparison.py`, `plot_non_wrapping_zoom_profiles.py` | `Non_Wrapping_Pulse/` |
| §5.4 Periodic pulses | `thesis_5p4_horton[_aggressive]_N{0,1,2}` | `smooth_pulse_model_comparison_cases.py` | `Smooth_Pulse/` |
| §5.5 Open boundary | `thesis_5p5_horton_N{0,1,2}` | `inflow_outflow_comparison.py` | `Smooth_Pulse_Inflow_Outflow/` |
| §5.6 Source ablation | `thesis_5p6_source_{free,active}_N{1,2}` | `dry_wet_ablation_comparison.py`, `zoomed_dry_wet_comparison.py` | `Dry_Wet_Test/` |

In §5.6, `Dry`/`Wet` means source-free/source-active. The name predates, and is
unrelated to, the solver's wet-dry treatment.

§5.1 and §5.2 match the thesis's closed-form solutions: eq. (5.5) for all three `alpha_R`
branches, and eqs. (5.9)–(5.10) with transition time t\* = 510.8 (thesis: ≈511) and final
depth 1.1694. §5.3–§5.6 were reproduced figure by figure.

The other shipped cases are the default `config`, the `topography_lake_at_rest` and
`topography_perturbed_lake` benchmarks, `wetdry_dam_break`, and the three smoke tests.

## Things to know before running

Each point is covered in full in [What to watch out for](docs/limitations.md).

- **Topography needs `pvm: Roe` or `Osher`.** Only these keep a lake at rest at rest
  (verified to machine precision); `LF` and `PRICE` leave a residual and trigger a
  warning. A config with no `topography:` section, or a flat bed, runs exactly as before.
- **`h_dry` is a modelling choice.** A coarse value slows a vacuum front, and refining the
  mesh does not fix it. Set it well below the smallest depth you care about, and check
  that `ν/h_dry²` stays reasonable.
- **With friction near drying fronts, integrate the source implicitly**
  (`linear_source: true`, `time_integrator: ImplicitEuler`). Explicit integration goes
  unstable and shows up as a negative depth.
- **A vacuum front over a sloping bed is not supported.** Wet-dry and topography each
  work alone; keep dry fronts on a flat bed.
- **SWME at N ≥ 2 can lose hyperbolicity; HSWME never does.** N = 0 and 1 are always
  safe. At N = 2 the unstable states lie in a narrow wedge of `|a₂/a₁|`, so the ratio of
  the moments matters, not their size. All 20 thesis runs were audited state by state
  and are clean.
- **The complex-spectra warning is not proof of hyperbolicity loss.** It measures the
  scheme's interface matrix, which a wet-dry front trips even at N = 0. Set
  `store_hyperbolicity: true` to check the model itself, cell by cell.
- **The solver is slow at scale.** It evaluates the system matrix one cell at a time;
  batching it is the fix to make before any large production run.

## Repository layout

```text
src/swme/          core solver: transport, coefficients engine, mesh, topography,
                   wet-dry, schemes, simulation driver, CLI
src/swme/config/   shipped YAML cases
src/swme/report/   PDF run report (moment-sw-report), built from CSVs
src/swme/viewer/   browser viewer (moment-sw-view, Dash + Plotly), same loader as the report
src/recharge/      rainfall/infiltration/exfiltration extension, built on swme
scripts/           run_thesis_configs.sh, run_smoke_tests.sh
processing/        thesis Chapter 5 plotting scripts (not part of the package)
docs/              documentation site sources (mkdocs.yml at the repo root)
results/           solver output (gitignored)
tests/             pytest suite
```

## Development

```bash
uv run pytest tests/test_coefficients.py -q     # one file
uv run pytest -k "closed_form" -q               # a keyword subset

uv add <package>                                # runtime dependency
uv add --group dev <package>                    # dev-only dependency

uv sync --group docs
uv run mkdocs serve                             # docs preview at http://127.0.0.1:8000
uv run mkdocs build --strict                    # fails on broken links and bad docstrings
```

Run `mkdocs build --strict` before committing documentation changes.

## Status

The restructure (steps 0–9 of [`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md)) is complete
and validated against the thesis's own results. What remains is in the plan's §7 and in
[What to watch out for](docs/limitations.md), chief among them the per-cell matrix
evaluation noted above.
