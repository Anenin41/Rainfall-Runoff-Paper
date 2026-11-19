# Rainfall-Runoff Paper

A 1D finite-volume solver for the Shallow Water Moment Equations (SWME), their
hyperbolic-regularised variant (HSWME), and a rainfall-runoff extension (RechargeSWME:
rainfall, infiltration, exfiltration and mixing friction), with bed topography and
wet-dry fronts. It ships the configs, post-processing and reports that reproduce the
accompanying thesis's results.

Shallow water moment models keep a cheap 1D grid but let the horizontal velocity vary
over the depth, through `N` extra unknowns per cell. `N = 0` is classical shallow water.

## Run it

Requires [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                                    # create .venv/ from pyproject.toml + uv.lock
uv run moment-sw --list-configs            # the cases shipped in src/swme/config/
uv run moment-sw --config smoke_test_1 --report
```

The last line runs one case, writes its CSVs to `results/smoke_test_1/` and builds a PDF
report next to them. To check the install, run the test suite:

```bash
uv run pytest -q                           # 1116 tests, ~9 min
```

## The four commands

| Command | What it does |
|:--|:--|
| `moment-sw` | Run a case from a YAML config. `--report` also writes a PDF; `--output-dir` chooses where results go |
| `moment-sw-report <dir>` | Build the PDF report from a finished run, or from every run under a tree, without re-running |
| `moment-sw-view [dir]` | Explore finished runs in the browser: move through time, zoom, overlay runs |
| `purge` | List run directories under `results/` with their sizes. Deleting needs an explicit pattern and a confirmation |

A config is a YAML file with sections `pde`, `grid`, `numerics` and, optionally,
`topography`, `wet_dry`, `postprocessing`. Unknown keys are rejected, so a typo fails
loudly. Every run writes the same filenames, so give each run its own directory (the
default is `results/<config-name>/`).

## Three cases to start with

```bash
scripts/run_smoke_tests.sh --report     # all three in parallel, ~2 min, PDFs included
```

| Case | What it shows | Result to check |
|:--|:--|:--|
| `smoke_test_1` | Rain and Horton infiltration over a stepped hillslope, HSWME at N = 3 | Mean depth falls while the Horton capacity exceeds the rain, bottoms out at `t_p = 0.0788`, then rises |
| `smoke_test_2` | Dam break onto an exactly dry bed, stiff friction integrated implicitly, N = 2 | Mass conserved to 2.2e-16, no negative depths |
| `smoke_test_3` | N = 5 under rainfall and exfiltration, SWME against HSWME | Mean depth rises at exactly `R - I = 0.2`; the HSWME spectrum is real everywhere, plain SWME's is not |

A longer case, `rain_fills_lake_overtops_dam` (about 3 minutes), puts the pieces together:
rain fills a lake with dry banks until it spills over a dam, with Horton infiltration.

Each config's header comment says what to look for and quotes the numbers the run
reproduces. `scripts/run_thesis_configs.sh` runs the twenty thesis cases; see
[Reproduce the thesis](docs/quickstart.md#reproduce-the-thesis).

## Before trusting a number

- Over topography use `pvm: Roe` or `Osher`. Only these keep a lake at rest at rest.
- Dry shores and dry fronts over a sloping bed run, with measured limits: a front
  climbing a dry slope is too slow.
- Plain SWME can lose hyperbolicity for N >= 2. HSWME does not.
- With friction near a drying front, integrate the source implicitly
  (`time_integrator: ImplicitEuler`).

Each of these, and the rest of the measured limits, is explained in
[What to watch out for](docs/limitations.md).

## Documentation

The documentation lives in [`docs/`](docs/) (`uv run --group docs mkdocs serve` to browse
it as a site):

| Page | Contents |
|:--|:--|
| [Quick start](docs/quickstart.md) | Installing, running, output files, reproducing the thesis, developing |
| [The model](docs/model.md) | The equations and where each term comes from |
| [Numerical method](docs/numerics.md) | The finite-volume scheme, topography, dry ground |
| [Configuration](docs/configuration.md) | Every config key |
| [Run reports](docs/reports.md) | The PDF report (`moment-sw-report`) |
| [Interactive viewer](docs/viewer.md) | The browser viewer (`moment-sw-view`) |
| [What to watch out for](docs/limitations.md) | Measured limits of the solver |

[`RESTRUCTURE_PLAN.md`](RESTRUCTURE_PLAN.md) records the design decisions, the
measurements behind them and the defects found on the way. Read it before changing the
structure.
