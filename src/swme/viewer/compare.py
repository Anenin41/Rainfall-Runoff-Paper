"""Several runs on the same axes: overlays, differences, profiles.

This is what `processing/*.py` does by hand per thesis section (N = 0/1/2 of
one case, source-free against source-active), generalised to any runs the
viewer lists.

Three rules, each a way such comparisons quietly go wrong:

- **Moments are matched by name and never padded.** An N = 1 run has no
  `alpha_2`, so it is absent from that panel rather than drawn as zero, which
  would read as a measurement.
- **A run is shown at its own nearest stored time**, and the legend names that
  time whenever it is not the requested one, so two curves are never presented
  as simultaneous when they are not.
- **Different grids are interpolated onto the reference grid**, and the figure
  carries a note saying so - a difference of two runs on different meshes
  includes interpolation error, and the reader should know it is there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..report.data import RunData
from .figures import (PANEL_HEIGHT, Z_POINTS, _finish, _position, _vertical_cursor,
                      cell_history, field_values, padded_range, sample_indices,
                      fmt, moment_label, placeholder, snapshot)
from .theme import LIGHT, Theme, run_styles

MAX_RUNS = 6
DIFFERENCE_SAMPLES = 40


@dataclass(frozen=True)
class Compared:
    """A run in a comparison: its data, its legend label and its moment order."""

    run: RunData
    label: str

    @property
    def order(self) -> int:
        return self.run.order


def _styles(runs, theme):
    return run_styles([item.order for item in runs], theme)


def common_time_range(runs) -> tuple[float, float] | None:
    """The interval every run has stored snapshots for, if they overlap."""
    ranges = [(float(item.run.snapshots.times[0]), float(item.run.snapshots.times[-1]))
              for item in runs if item.run.snapshots is not None]
    if not ranges:
        return None
    start, end = max(r[0] for r in ranges), min(r[1] for r in ranges)
    return (start, end) if start <= end else None


def _series(runs) -> list[tuple[str, str]]:
    top = max(item.order for item in runs)
    return ([("h", "water depth h"), ("u_m", "mean velocity u<sub>m</sub>")]
            + [(f"a{i}", moment_label(i)) for i in range(1, top + 1)])


def _column(run: RunData, column: str) -> int | None:
    names = ("x", "h", "u_m", *run.moment_columns)
    return names.index(column) if column in names else None


def _time_label(label: str, actual, t, final: bool) -> str:
    if final:
        return f"{label} (final, t = {fmt(actual, '.4g')})"
    if t is not None and actual is not None and not math.isclose(
            actual, t, rel_tol=1e-6, abs_tol=1e-12):
        return f"{label} (t = {actual:.4g})"
    return label


def _spacing(run: RunData) -> float:
    """Median gap between stored snapshots: the time resolution of `run`."""
    times = run.snapshots.times if run.snapshots is not None else []
    return float(np.median(np.diff(times))) if len(times) > 1 else 0.0


def _grid(count: int, titles):
    rows = max(1, math.ceil(count / 2))
    cols = 1 if count == 1 else 2
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(titles),
                        shared_xaxes="all", vertical_spacing=min(0.12, 0.30 / rows),
                        horizontal_spacing=0.09)
    return fig, rows, cols


# ---------------------------------------------------------------------------
# time histories
# ---------------------------------------------------------------------------

def histories_figure(runs, t: float | None = None, theme: Theme = LIGHT) -> go.Figure:
    """Spatial means over time, one line per run, one panel per variable."""
    usable = [item for item in runs if item.run.summary_history is not None]
    if not usable:
        return placeholder("No time histories",
                           "None of the selected runs stored a summary history.", theme)
    series = _series(usable)
    fig, rows, cols = _grid(len(series), [label for _, label in series])
    for item, style in zip(usable, _styles(usable, theme)):
        summary = item.run.summary_history
        for position, (column, _) in enumerate(series):
            name = f"mean_{column}"
            if name not in summary:
                continue
            row, col = _position(position, cols)
            fig.add_trace(go.Scatter(
                x=summary["time"], y=summary[name], mode="lines", name=item.label,
                legendgroup=item.label, showlegend=position == 0,
                line=dict(color=style.color, dash=style.dash, width=2),
                hovertemplate=f"{item.label}: %{{y:.6g}}<extra></extra>"), row=row, col=col)
    fig.update_xaxes(title_text="t", row=rows)
    fig.update_layout(shapes=histories_cursor(runs, t, theme))
    return _finish(fig, theme, height=PANEL_HEIGHT * rows + 100,
                   title="Spatial means over time", legend=True)


def histories_cursor(runs, t: float | None, theme: Theme = LIGHT) -> list[dict]:
    usable = [item for item in runs if item.run.summary_history is not None]
    if t is None or not usable:
        return []
    return _vertical_cursor(range(1, len(_series(usable)) + 1), t, theme)


# ---------------------------------------------------------------------------
# fields at a time
# ---------------------------------------------------------------------------

def fields_figure(runs, t: float | None = None, theme: Theme = LIGHT,
                  *, frame: bool = False) -> go.Figure:
    """h, u_m and every moment at `t`, each run at its nearest stored state.

    `frame=True` skips the run-wide ranges and the styling, for a frame patch.
    """
    series = _series(runs)
    fig, rows, cols = _grid(len(series), [label for _, label in series])
    for item, style in zip(runs, _styles(runs, theme)):
        values, actual, final = snapshot(item.run, t)
        label = _time_label(item.label, actual, t, final)
        for position, (column, _) in enumerate(series):
            index = _column(item.run, column)
            if index is None:
                continue
            row, col = _position(position, cols)
            fig.add_trace(go.Scatter(
                x=values[:, 0], y=values[:, index], mode="lines", name=label,
                legendgroup=item.label, showlegend=position == 0,
                line=dict(color=style.color, dash=style.dash, width=2),
                hovertemplate=f"{item.label}: %{{y:.6g}}<extra></extra>"), row=row, col=col)
    # One fixed range per panel, over every run and every stored time.
    for position, (column, _) in enumerate(series if not frame else []):
        values = [field_values(item.run, column) for item in runs
                  if _column(item.run, column) is not None]
        row, col = _position(position, cols)
        fig.update_yaxes(range=padded_range(np.concatenate(values)), row=row, col=col)
    fig.update_xaxes(title_text="x", row=rows)
    return _finish(fig, theme, frame=frame, height=PANEL_HEIGHT * rows + 100,
                   title=f"Fields at t = {fmt(t, '.4g')}", legend=True)


# ---------------------------------------------------------------------------
# difference against a reference
# ---------------------------------------------------------------------------

def _on_grid(x_target, x_source, values):
    """`values` sampled on `x_target`, and whether that needed interpolation."""
    if len(x_source) == len(x_target) and np.allclose(x_source, x_target,
                                                      rtol=0, atol=1e-12):
        return values, False
    return np.interp(x_target, x_source, values), True


def _difference(base: Compared, item: Compared, field: str, t):
    """`item - base` for `field` on the base grid at `t`.

    Returns `(x, difference, base_time, item_time, item_values, interpolated)`.
    """
    base_values, base_time, _ = snapshot(base.run, t)
    values, actual, _ = snapshot(item.run, t)
    x = base_values[:, 0]
    sampled, interpolated = _on_grid(x, values[:, 0], values[:, _column(item.run, field)])
    return (x, sampled - base_values[:, _column(base.run, field)], base_time, actual,
            values, interpolated)


def _difference_range(base: Compared, others, field: str) -> list[float] | None:
    """Symmetric fixed range for the difference plot, over the reference's times."""
    times = [None] if base.run.snapshots is None else [
        float(base.run.snapshots.times[i])
        for i in sample_indices(len(base.run.snapshots.times), DIFFERENCE_SAMPLES)]
    peak = max((float(np.nanmax(np.abs(_difference(base, item, field, moment)[1])))
                for item in others for moment in times), default=0.0)
    limit = 1.05 * peak if peak > 0 else 1e-15
    return [-limit, limit]


def difference_figure(runs, reference: int = 0, field: str = "h",
                      t: float | None = None, theme: Theme = LIGHT, *, frame: bool = False):
    """`run - reference` for one field at `t`. Returns `(figure, notes)`."""
    notes: list[str] = []
    if len(runs) < 2:
        return placeholder("Pick at least two runs",
                           "A difference needs a reference and one other run.",
                           theme), notes
    reference = min(max(int(reference), 0), len(runs) - 1)
    base = runs[reference]
    base_index = _column(base.run, field)
    if base_index is None:
        return placeholder(f"The reference has no {field}",
                           f"{base.label} is N = {base.order}; pick a reference that "
                           "carries this moment.", theme), notes
    _, base_time, base_final = snapshot(base.run, t)
    fig = go.Figure()
    styles = _styles(runs, theme)
    others = []
    for position, (item, style) in enumerate(zip(runs, styles)):
        if position == reference:
            continue
        if _column(item.run, field) is None:
            notes.append(f"{item.label} has no {field} (N = {item.order}); left out.")
            continue
        others.append(item)
        x, difference, _, actual, values, interpolated = _difference(base, item, field, t)
        if interpolated:
            notes.append(
                f"{item.label} ({len(values)} cells) was linearly interpolated onto "
                f"the reference grid ({len(x)} cells); the difference includes "
                "interpolation error.")
            if values[0, 0] > x[0] or values[-1, 0] < x[-1]:
                notes.append(f"{item.label} does not span the reference domain; "
                             "values outside it are held constant.")
        if (actual is not None and base_time is not None
                and abs(actual - base_time) > 0.5 * max(_spacing(item.run),
                                                        _spacing(base.run))):
            notes.append(f"{item.label} is at t = {actual:.4g}, the reference at "
                         f"t = {base_time:.4g}: the difference is not simultaneous.")
        fig.add_trace(go.Scatter(
            x=x, y=difference, mode="lines",
            name=f"{item.label} − {base.label}",
            line=dict(color=style.color, dash=style.dash, width=2),
            hovertemplate=f"{item.label}: %{{y:.4e}}<extra></extra>"))
    if not fig.data:
        return placeholder("Nothing to difference",
                           "No other selected run carries this field.", theme), notes
    fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[0, 0], mode="lines", showlegend=False,
                             line=dict(color=theme.axis, width=1), hoverinfo="skip"))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text=f"Δ {field}",
                     range=_difference_range(base, others, field) if not frame else None)
    when = "final state" if base_final else f"t = {fmt(base_time, '.4g')}"
    _finish(fig, theme, frame=frame, height=420, legend=True,
            title=f"{field} relative to {base.label}, {when}")
    return fig, notes


# ---------------------------------------------------------------------------
# vertical profiles
# ---------------------------------------------------------------------------

def profiles_figure(runs, x: float | None = None, t: float | None = None,
                    theme: Theme = LIGHT, *, frame: bool = False) -> go.Figure:
    """u(z), and its departure from plug flow, at one position for every run."""
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=["velocity profile u(z)",
                                        "departure from plug flow u(z) − u<sub>m</sub>"])
    velocity_extent, deviation_extent = [], []
    for item, style in zip(runs, _styles(runs, theme)):
        values, actual, final = snapshot(item.run, t)
        target = float(np.mean(values[:, 0])) if x is None else float(x)
        cell = int(np.argmin(np.abs(values[:, 0] - target)))
        profile = item.run.velocity_profile(values[cell:cell + 1], Z_POINTS)[0]
        if not frame:
            history = cell_history(item.run, cell)
            over_time = item.run.velocity_profile(history, Z_POINTS)
            velocity_extent.append(over_time.ravel())
            deviation_extent.append((over_time - history[:, 2][:, None]).ravel())
        label = _time_label(item.label, actual, t, final)
        if not math.isclose(values[cell, 0], target, abs_tol=1e-9):
            label += f" @ x = {values[cell, 0]:.4g}"
        line = dict(color=style.color, dash=style.dash, width=2)
        fig.add_trace(go.Scatter(x=profile, y=Z_POINTS, mode="lines", name=label,
                                 legendgroup=item.label, line=line,
                                 hovertemplate=f"{item.label}: u = %{{x:.6g}}<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=profile - values[cell, 2], y=Z_POINTS, mode="lines",
                                 name=label, legendgroup=item.label, showlegend=False,
                                 line=line,
                                 hovertemplate=f"{item.label}: u − u_m = %{{x:.4e}}<extra></extra>"),
                      row=1, col=2)
    fig.update_xaxes(title_text="u", row=1, col=1,
                     range=padded_range(np.concatenate(velocity_extent)) if not frame else None)
    fig.update_xaxes(title_text="u − u_m", row=1, col=2,
                     range=padded_range(np.concatenate(deviation_extent)) if not frame else None)
    fig.update_yaxes(title_text="z  (0 = bed, 1 = surface)", range=[0, 1], row=1, col=1)
    where = "domain centre" if x is None else f"x = {float(x):.4g}"
    _finish(fig, theme, frame=frame, height=440, legend=True,
            title=f"Vertical profiles at {where}, t = {fmt(t, '.4g')}")
    fig.update_layout(hovermode="closest")
    return fig


# ---------------------------------------------------------------------------
# a point over time
# ---------------------------------------------------------------------------

def point_series_figure(runs, x: float | None = None, field: str = "h",
                        t: float | None = None, theme: Theme = LIGHT, *, frame: bool = False):
    """One field at one position over time (an outlet hydrograph, say).

    Sampled from the stored snapshots, so its time resolution is the snapshot
    spacing, not the solver's step. Returns `(figure, notes)`.
    """
    notes: list[str] = []
    fig = go.Figure()
    for item, style in zip(runs, _styles(runs, theme)):
        snapshots = item.run.snapshots
        if snapshots is None:
            notes.append(f"{item.label} stored no field history; left out.")
            continue
        index = _column(item.run, field) if field != "q" else None
        if field != "q" and index is None:
            notes.append(f"{item.label} has no {field} (N = {item.order}); left out.")
            continue
        target = float(snapshots.x[-1]) if x is None else float(x)
        cell = int(np.argmin(np.abs(snapshots.x - target)))
        if field == "q":
            series = snapshots.values[:, cell, 1] * snapshots.values[:, cell, 2]
        else:
            series = snapshots.values[:, cell, index]
        spacing = _spacing(item.run)
        notes.append(f"{item.label}: {len(snapshots.times)} snapshots of "
                     f"{snapshots.total_available} stored steps, every ~{spacing:.3g} in t, "
                     f"at x = {snapshots.x[cell]:.4g}.")
        fig.add_trace(go.Scatter(x=snapshots.times, y=series, mode="lines",
                                 name=item.label,
                                 line=dict(color=style.color, dash=style.dash, width=2),
                                 hovertemplate=f"{item.label}: %{{y:.6g}}<extra></extra>"))
    if not fig.data:
        return placeholder("No field history",
                           "Point time series need postprocessing.store_history: true.",
                           theme), notes
    fig.update_xaxes(title_text="t")
    fig.update_yaxes(title_text=field if field != "q" else "q = h·u_m")
    where = "the right boundary" if x is None else f"x = {float(x):.4g}"
    fig.update_layout(shapes=_vertical_cursor([1], t, theme) if t is not None else [])
    _finish(fig, theme, frame=frame, height=400, legend=True, title=f"{field} at {where} over time")
    return fig, notes
