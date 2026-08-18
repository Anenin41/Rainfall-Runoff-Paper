# Run reports

The package can render a multi-page PDF describing a single run: the final state, time
histories, space-time maps, vertical velocity profiles, hyperbolicity diagnostics, and
where they apply, wet-dry behaviour and topography.

## Two ways to get one

```bash
# During the run
uv run moment-sw --config wetdry_dam_break --report

# Afterwards, from the files on disk
uv run moment-sw-report results/wetdry_dam_break
```

The second form matters more than it looks. It reads only the CSVs, so results
produced months ago can be reported on without re-running anything, which for a large
case is hours of compute saved.

`moment-sw-report` also accepts a directory tree and reports on every run beneath it:

```bash
uv run moment-sw-report results/Dry_Wet_Test
```

Options:

| Flag | Meaning |
|:--|:--|
| `-o, --output` | Write to a specific file instead of `report.pdf` in the run directory |
| `--prefix` | Choose a run when a directory holds more than one |
| `--max-snapshots` | How many stored steps to draw. Default 240 |
| `--chunk-rows` | Rows per read chunk. Bounds memory, does not change the result |
| `--all` | Treat the argument as a tree even if it looks like a single run |

Reporting is off by default during a run. Turning it on is a deliberate choice because
a plain solver run otherwise imports matplotlib and never uses it.

## The pages

| Page | Shown when |
|:--|:--|
| Run summary | Always. Settings, plus a list of which data sources were found |
| Final state | Always. Depth, velocity, and one panel per moment |
| Time histories | Always. A placeholder if history was not stored |
| Space-time evolution | Field history exists |
| Vertical velocity profiles | Always |
| Hyperbolicity | Always |
| Hyperbolicity in space and time | Per-cell spectra were recorded |
| Wet-dry behaviour | Some cell went below \(h_{\text{wet}}\) |
| Topography | The bed is not flat |

Page order is fixed. It does not depend on the data, so two reports of the same case
can be compared page by page.

## How the pages are laid out

Almost everything on a page has a length that depends on the run: the cover's warning
list, the prose on the hyperbolicity page, the captions under the diagnostics. The
layout is therefore measured rather than fixed, and three rules keep it from colliding
with itself:

- **Text is wrapped to the box it is drawn in.** `ReportStyle` knows the font metrics,
  so a block is wrapped to the column it will occupy and, if it still does not fit,
  shrunk until it does. Matplotlib's own `wrap=True` wraps to the *figure* width, not
  the artist's box, which is why it is not used anywhere here.
- **A caption takes its space out of its own panel.** The band a note occupies is
  subtracted from the bottom of the axes it belongs to, never borrowed from the gap
  above the next panel, so a long caption shortens its own plot instead of printing
  over the title below it.
- **A page is only as tall as it needs to be.** Rows whose height depends on the text
  are measured first; a page with one row of panels or a single placeholder uses part
  of the sheet rather than stretching two plots over A4.

Style is a value, not a global: every page function takes a `ReportStyle` and no
`rcParams` are touched, so rendering a report cannot change how anything else plots.

## Reading the hyperbolicity pages

This is the part most easily misread, so it is worth explaining what the numbers are.

There are **two different measurements**, kept on separate panels because they answer
different questions.

### Scheme-level counters

These come from the flux scheme and are always available. They eigendecompose the
**path-averaged interface matrix**, the quantity

\[
\sum_k w_k\, A\bigl(\psi(s_k)\bigr)
\]

which is an average of the system matrix taken over the path between two neighbouring
cells. It is not \(A(U)\) at any single state. Because \(A\) depends non-linearly on
\(U\), that average is not \(A\) of anything, and it can have complex eigenvalues even
when every matrix being averaged has real ones.

A concrete case: a dam break onto a dry bed trips these counters at 78 of 12462
interfaces at \(N = 0\). At \(N = 0\) the model is plain shallow water, which is
unconditionally hyperbolic and has no moments to destabilise. The count is identical at
\(N = 1\), \(N = 2\) and for HSWME, which is what gives it away as a property of the
averaging and the jump, not of the model.

**So a nonzero count here does not mean the model lost hyperbolicity.** The report says
so on the page.

The counters also mean different things depending on the scheme, which is why the page
names it:

- `Roe` records one path-averaged matrix per interface.
- `Osher` records five per interface, one per quadrature node, each scaled by its
  quadrature weight. Its counts and magnitudes are not comparable with Roe's.
- `LF` and `PRICE` never eigendecompose anything. Their counters are always zero, and
  that zero carries no information at all.

If the run has no metadata sidecar the scheme is unknown, and the report does not print
the counters, because they cannot be interpreted without it.

### Model-level spectrum

This is the measurement that actually says something about the model: the eigenvalues
of \(A(U)\) in each cell. It requires `store_hyperbolicity: true` in the config.

When it was not recorded, the report says so and tells you how to enable it. It never
shows a blank panel, because a reader cannot tell an empty figure from a clean result.

Cells that were never evaluated, because they were dry or because the eigensolve
failed, are counted and drawn **separately** from cells that genuinely lost
hyperbolicity. A dry cell has had its moments ramped away, so its spectrum says nothing
about the model, and counting it as a failure would blame the model for the wet-dry
treatment.

!!! note "Older runs carry a known error here"
    Runs produced before this diagnostic was fixed counted dry cells among the
    non-hyperbolic ones, and their recorded worst-cell entry could latch to a dry cell.
    When the per-cell file is available the report recomputes the correct numbers from
    it and ignores the recorded summary. When only the summary survives, the numbers
    are shown but labelled as uncorrectable.

## The metadata sidecar

Each run writes `<stem>_run.json` next to its CSVs. It records the model, moment order,
flux scheme, time integrator, wet-dry thresholds, bed profile and its parameters, the
domain, timings, and the end-of-run scheme counters.

It exists because the CSVs contain numbers but not the settings that produced them.
Without it a report rebuilt from disk cannot name the flux scheme, which as described
above makes the hyperbolicity counters uninterpretable, and it has no bed to draw on the
topography page.

The bed is stored as a profile name and its parameters rather than as a sampled array,
because evaluating the named profile at the cell centres in the CSV reproduces the
original sampling to round-off.

A run directory without a sidecar still produces a report. It loses the topography page
and the scheme counters, and the cover page lists what was missing.

## Working with large runs

Field history files get big. The largest in the reference results is 185 MB with 1.54
million rows.

The report never loads one whole. Time series come from the much smaller summary file,
and the field history is streamed in chunks, keeping an evenly spaced subset of stored
steps. Peak memory depends on the snapshot budget and the chunk size, not on the length
of the run. Loading that 185 MB file uses about 30 MB and takes around three seconds.

Snapshots are evenly spaced rather than randomly sampled, so the first and last steps
are always present and the space-time maps have uniform rows. Uneven rows would
misrepresent wave speeds to the eye.

## Using the report from Python

```python
from swme import report

run = report.from_directory("results/wetdry_dam_break")
result = report.build_report(run, "report.pdf")

print(result.page_keys)     # which pages were drawn
print(result.skipped_keys)  # which were not, and so what was missing
print(run.warnings)         # anything the loader could not find
```

`report.from_simulation(sim, final_state)` does the same for a run still in memory.

Individual pages can be pulled out without writing a PDF:

```python
from swme.report import assemble

for spec, figure in assemble.iter_figures(run):
    if spec.key == "velocity_profiles":
        figure.savefig("profiles.png", dpi=200)
```

Page functions take a run and a style and return a figure. They do not write files or
open windows, which is what allows them to be assembled into one document.
