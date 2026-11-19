# Interactive viewer

`uv run visual` opens every run under a results directory in a local web app. It
shows what the [run report](reports.md) shows, but you can move through time, hover
for values, zoom into a front, choose where the vertical profile is drawn, and put
several runs on the same axes.

The PDF and the viewer read the same files through the same loader, so they cannot
disagree about what a run contains. The PDF is the record to keep or send, and the
viewer is for exploring.

## Starting it

The viewer is installed with the package, so a plain `uv sync` is all it needs:

```bash
uv run visual results                    # every run under results/
uv run visual results/Dry_Wet_Test
uv run visual results/smoke_test_1
```

It prints a local address and opens it in your browser. The argument can be a whole
results tree, one family of runs, or a single run directory. Without one, `uv run visual`
prints its help; the alias `uv run moment-sw-view` opens `results/`. Nothing is re-run.
Like `uv run report`, it only reads finished output.

| Flag | Meaning |
|:--|:--|
| `--port` | Port to serve on. Default 8050 |
| `--host` | Interface to bind. Default `127.0.0.1`, meaning this machine only |
| `--max-snapshots` | Stored steps kept per run, for the time slider and the maps. Default 400 |
| `--chunk-rows` | Rows per read chunk when streaming field history. Bounds memory only |
| `--cache` | Runs kept loaded at once. Default 6 |
| `--no-browser` | Do not open a browser tab |
| `--debug` | Dash debug mode (error overlay, hot reload), and the server's log of every request. Without it the terminal shows only the start-up lines and errors |

By default the viewer is reachable only from your own machine. It serves whatever is
under the directory you give it, so pass `--host 0.0.0.0` only on a network you trust.

**On a remote machine,** the viewer listens on that machine only. VS Code Remote-SSH
forwards port 8050 to your browser automatically. From a plain SSH session, run
`ssh -L 8050:localhost:8050 <host>` and open `http://localhost:8050/`. Pass
`--no-browser` there, and `--port` if 8050 is taken.

## The layout

The **run list** on the left groups runs by the directory they sit in. Under
`results/` laid out by `scripts/run_thesis_configs.sh`, that is one group per thesis
section, such as `Non_Wrapping_Pulse` with its `N0`, `N1` and `N2` runs. Each entry
shows the moment order, the closure (SWME or HSWME), the flux scheme, and whether
per-cell spectra (λ) were recorded. The filter box matches names and config names,
and ↻ re-scans the directory for runs that finished after the viewer started.

Two controls along the top drive every view:

- **Time.** The slider covers the stored steps. ▶ plays through them, ◀ and ▶| step
  one stored step at a time, the dropdown beside them sets the speed, and **GIF**
  downloads the run as an animation (see [Playback](#playback)). Clicking a
  point on any space-time map moves the time there.
- **Position `x`.** This picks the cell used for the vertical profile, the eigenvalue
  spectrum and the point time series. Leave it blank for a sensible default, type a
  value, or click any map.

Every plot has the usual Plotly tools: drag to zoom, double-click to reset, hover for
values. The camera button saves an SVG. Zoom survives moving the time slider, so you
can zoom into a front and then play through it.

## Playback

Moving through time is meant to read as motion, so the viewer does four things:

- **Playing runs in the browser.** Pressing ▶ hands the clock to the page. It moves the
  slider and draws each stored step itself, from frames the server builds ahead of the
  cursor, about 0.4 s of work at a time. There is no server round trip per frame. Each
  frame is built by the same code, with the same settings, as the paused view at that
  time, so what plays is what you would see by stopping there. If the next frame has
  not arrived, the player waits for it rather than skipping it. The text panels (the
  wet-dry cell counts, the topography caption) and the other tabs catch up when
  playback stops, on the last step or when you pause.

- **The axes stay still.** Every view that changes with time fixes its axes, and its
  colour scales, to the range the data covers over the *whole* run. Only the curves
  move. A view that rescaled to each frame would make the axes jump while the data
  looked frozen.
- **Plots are updated, not redrawn.** A new time sends only the new trace data, the
  titles that quote the time, and the cursor line. The figure, its axes and your zoom
  stay as they are. The space-time maps and time histories are drawn once, and after
  that only their cursor moves. A spinner appears only if an update takes noticeably
  long, such as the first load of a large run, and the plot stays visible underneath.
- **Playback stops on the last stored step** rather than wrapping, so the final state
  stays on screen. Pressing ▶ there starts again from the beginning. Stepping pauses
  playback.

The speeds run from 0.25× to 2×, where 1× is 20 stored steps a second, the same in both
modes. The first play of a tab can only go as fast as the server builds frames.
Measured on the 200-cell lake run, that is about 18 steps a second on Fields, 11 on
Profiles (three figures) and 20 on Topography. A tab whose plots only move a cursor
(Histories, Space-time) keeps up at any speed. Changing the run, the tab, `x`, the
theme or a compare setting discards the frames buffered so far.

**GIF export.** The **GIF** button beside the speed downloads the selected run's stored
steps as an animated GIF that loops, playing at the chosen speed. It is drawn with
matplotlib in the style of the PDF report, only when you press the button, and
shows the panels of the report's final-state page at every stored step: `h`, `u_m`
and every moment, with the free surface `h + Z` over the bed first when the run has
topography. Every axis is fixed to the range over the whole run. Measured on the
200-cell lake run, 400 stored steps take about 11 s and give a 2.3 MB file. The button
is off in compare mode and for a run without field history.

A stored step is a snapshot of the run, not a solver step. How many there are depends
on the run's `history_stride` and on `--max-snapshots`.

## Single-run views

| Tab | What it shows |
|:--|:--|
| Overview | The settings from the run's sidecar, which data sources were found, the loader's notes, and the end of `run.log` |
| Fields | Depth, velocity and one panel per moment at the current time (or the final state), with the bed under `h` when there is one |
| Histories | Spatial mean and min–max envelope of every variable over time, plus the mean discharge |
| Space-time | Any field (including `h + Z` and `q = h·u_m`) as an `x`–`t` map |
| Profiles | `u(z)` at the chosen cell now and over time, `u(x, z)`, and its departure from plug flow |
| Hyperbolicity | The scheme counters and the model-level spectrum, the same text as the report, plus counts over time, a space-time map, and the eigenvalues of the chosen cell |
| Wet-dry | Depth against `h_dry` and `h_wet`, the dry/transition/wet map, and cell counts. Available when some cell went below `h_wet` |
| Topography | Free surface over the bed, the `h + Z − H` residual, and the bed. Available when the bed is not flat |

The same rules as the report apply. Missing data is shown as a placeholder that says
what is missing and how to record it, never as an empty plot. Dry cells are drawn
blank on the hyperbolicity maps and counted separately from cells that genuinely lost
hyperbolicity. When the per-cell spectra are present, the counts are recomputed from
them, so older runs with the dry-cell counting defect still show correct numbers. See
[Reading the hyperbolicity pages](reports.md#reading-the-hyperbolicity-pages).

## Comparing runs

Switch to **Compare** at the top and add runs with the **+** button next to each one,
or with **compare all** on a group. Up to six runs can be compared. One of them is the
reference for the difference plot.

| Plot | What it shows |
|:--|:--|
| Spatial means over time | One line per run, one panel per variable |
| Fields | Every run at the current time |
| Difference | One field of each run minus the reference, at the current time |
| Vertical profiles | `u(z)` and `u(z) − u_m` at the chosen `x` for every run. This shows what the moments add |
| Point series | One field at the chosen `x` over time, by default at the right boundary (an outlet hydrograph) |

Colour follows the moment order: N = 0 is always blue, N = 1 orange, N = 2 aqua, and
so on, in every view. Runs that share an order, such as source-free and source-active
at N = 1, keep that colour and differ in line style.

Comparisons quietly go wrong in three ways, and the viewer handles each explicitly:

- **Moments are matched by name and never padded.** An N = 1 run has no `α₂`, so it is
  absent from that panel rather than drawn as zero.
- **Each run is shown at its own nearest stored time.** When that differs from the
  slider's time, the legend says so. When two runs are further apart than their
  snapshot spacing, the difference plot says it is not simultaneous.
- **Runs on different grids are interpolated** onto the reference grid, with a note,
  because the difference then includes interpolation error.

The time slider in compare mode covers the interval that every selected run has data
for.

## Large runs

The viewer loads runs the same way the report does. Field history is streamed in
chunks and decimated to an evenly spaced set of stored steps (`--max-snapshots`), so
memory per run does not depend on the length of the run. Loaded runs are cached, and a
run whose files change on disk is reloaded on next use.

Per-cell hyperbolicity spectra are the largest files a run writes. They are read only
when the Hyperbolicity tab is opened, and fewer of them are kept in memory than runs.

Moving the time slider only re-sends what depends on time. The heavy maps are drawn
once per run and field, and afterwards only their cursor line moves.

## Using it from Python

The figures are plain functions of a loaded run and work in a notebook without the
app:

```python
from swme import report
from swme.viewer import figures, compare

run = report.from_directory("results/smoke_test_1")
figures.space_time_figure(run, "h").show()
figures.profile_lines_figure(run, t=0.4, x=0.3).show()

other = report.from_directory("results/smoke_test_3")
compare.profiles_figure([compare.Compared(run, "N3"), compare.Compared(other, "N5")],
                        x=0.5, t=0.2).show()
```
