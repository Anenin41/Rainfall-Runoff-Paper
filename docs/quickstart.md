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
| `smoke_test_1` | Rainfall with Horton infiltration over a non-flat bed, HSWME at \(N = 3\). Mean depth is flat to the last digit until the ponding time, then rises |
| `smoke_test_2` | A dam break onto exactly dry ground, with stiff friction integrated implicitly. Mass is conserved to \(2.2 \times 10^{-16}\) and no depth goes negative |
| `smoke_test_3` | Five moments under rainfall and exfiltration, and the SWME-versus-HSWME hyperbolicity contrast, which is one line of the config apart |

The three reports together contain every page the [report generator](reports.md) can
produce. If something in this package is broken, one of them will say so.

## Run a case

The solver is driven by YAML config files. A set of them ships inside the package.

```bash
# List the configs that come with the package
uv run moment-sw --list-configs

# Run one by name
uv run moment-sw --config wetdry_dam_break

# Run one by path, into a directory you choose
uv run moment-sw --config path/to/case.yaml --output-dir results/my_case
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

The history files are written only when `store_history` is on, which is the default.

`<stem>_run.json` matters more than its size suggests. The CSVs record numbers but not
the settings that produced them, so without it a result read months later cannot say
which flux scheme was used or what the bed looked like. See
[Run reports](reports.md#the-metadata-sidecar).

## Build a report

```bash
# From a finished run directory
uv run moment-sw-report results/wetdry_dam_break

# Every run under a directory tree
uv run moment-sw-report results/Dry_Wet_Test
```

This reads the CSVs and writes `report.pdf` next to them. Because it works from files
rather than from a live simulation, old results can be reported on without re-running
anything. See [Run reports](reports.md).

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
