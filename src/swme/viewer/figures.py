"""Single-run figures: `RunData -> go.Figure`, one function per view.

The views mirror the PDF report's pages (`swme.report.pages`,
`swme.report.hyperbolicity`) and keep their rules - a panel per moment with no
upper bound on N, `h + Z` rather than bare `h` over a bed, dry cells drawn
apart from non-hyperbolic ones, and an explicit placeholder rather than an empty
axis whenever data is missing. What changes is that every view takes a time
`t` (and where it matters a position `x`) instead of being fixed at `t_end`.

Figures whose payload is heavy and whose only time dependence is a cursor line
(the space-time maps, the time histories) come in pairs: `<view>_figure` builds
the whole thing once, and `<view>_cursor` returns just the cursor shapes, which
the app sends as a partial update while the slider moves. For that to work a
paired figure's `layout.shapes` must hold the cursor and nothing else -
reference lines in those figures are drawn as traces.

Builders do not know about Dash: they take data and a `Theme` and return a
figure, so each can be tested and used from a notebook on its own.
"""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..report.data import RunData, parse_eigenvalue_column
from .theme import LIGHT, Theme

PANEL_HEIGHT = 250
Z_POINTS = np.linspace(0.0, 1.0, 101)
MAX_MAP_ROWS = 400
PROFILE_CURVES = 6
PROFILE_MAP_SAMPLES = 24
MAGNITUDE_FLOOR = 1e-16


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def fmt(value, spec: str = ".6g") -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(float(value), spec) if np.isfinite(value) else "n/a"
    return str(value)


def moment_label(index: int) -> str:
    return f"moment α<sub>{index}</sub>"


def field_columns(run: RunData) -> list[tuple[str, str]]:
    """(column, label) for h, u_m and every moment, in layout order."""
    series = [("h", "water depth h"), ("u_m", "mean velocity u<sub>m</sub>")]
    series += [(name, moment_label(index))
               for index, name in enumerate(run.moment_columns, start=1)]
    return series


def _column_index(run: RunData, column: str) -> int:
    return ("x", "h", "u_m", *run.moment_columns).index(column)


def snapshot(run: RunData, t: float | None):
    """`(values, time, is_final)` for the stored state nearest `t`.

    `values` is `[x, h, u_m, a_1..a_N]` per cell. With no `t`, or no field
    history, it is the final state.
    """
    if t is None or run.snapshots is None:
        end = run.meta.t_end
        if end is None and run.snapshots is not None:
            end = float(run.snapshots.times[-1])
        return run.final_array(), end, True
    index = run.snapshots.nearest_index(t)
    return run.snapshots.values[index], float(run.snapshots.times[index]), False


def padded_range(values, pad: float = 0.05) -> list[float] | None:
    """`[low, high]` of the finite `values`, widened by `pad` of the span.

    An axis that follows the data at one time rescales every frame, and the
    eye then reads the moving axis instead of the moving curve. Every view
    that plays through time therefore fixes its axes to the range the data
    takes over the *whole* run, computed here once per full build. A field
    that never changes still gets a non-degenerate range.
    """
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    margin = pad * span if span > 0 else (0.1 * abs(high) or 1.0)
    return [low - margin, high + margin]


def field_values(run: RunData, column: str) -> np.ndarray:
    """Every value `column` takes in the stored snapshots and the final state."""
    final = run.final[column].to_numpy(dtype=np.float64)
    if run.snapshots is None:
        return final
    return np.concatenate([run.snapshots.field(column).ravel(), final])


def field_range(run: RunData, column: str) -> list[float] | None:
    """The fixed axis range for `column` over the whole run."""
    values = field_values(run, column)
    if column == "h" and run.has_topography:
        values = np.concatenate([values, run.bed_elevation])
    return padded_range(values)


def sample_indices(count: int, budget: int) -> np.ndarray:
    """At most `budget` evenly spaced indices into `count` items, ends included."""
    if count <= budget:
        return np.arange(count)
    return np.unique(np.linspace(0, count - 1, budget).round().astype(int))


def time_range(run: RunData) -> tuple[float, float] | None:
    if run.snapshots is None:
        return None
    return float(run.snapshots.times[0]), float(run.snapshots.times[-1])


def axis_ref(index: int) -> tuple[str, str]:
    """('x2', 'y2')-style axis names of the `index`-th subplot (1-based)."""
    suffix = "" if index == 1 else str(index)
    return f"x{suffix}", f"y{suffix}"


def _finish(fig: go.Figure, theme: Theme, *, height: int, title: str | None = None,
            legend: bool = False, frame: bool = False) -> go.Figure:
    """Apply the theme, size, title and legend.

    With `frame=True` only what a frame patch carries is set - the title and
    the subplot-title annotations. Applying the template and layout is most
    of the cost of building a figure, and a patch never sends any of it.
    """
    if frame:
        fig.layout.title.text = title
        fig.update_annotations(font=dict(color=theme.ink_secondary, size=12))
        return fig
    fig.update_layout(
        template=theme.template,
        height=height,
        margin=dict(l=64, r=24, t=72 if title else 48, b=48),
        title=dict(text=title) if title else None,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right",
                    x=1.0) if legend else None,
        hovermode="x unified",
    )
    # Subplot titles are annotations; give them secondary ink, left-aligned.
    fig.update_annotations(font=dict(color=theme.ink_secondary, size=12))
    return fig


def placeholder(title: str, message: str, theme: Theme = LIGHT,
                height: int = 220) -> go.Figure:
    """An explicit "not available" panel. Never an empty axis."""
    fig = go.Figure()
    fig.add_annotation(text=f"<b>{title}</b><br><br>{_wrap_html(message)}",
                       x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, align="center",
                       font=dict(color=theme.ink_secondary, size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=theme.template, height=height,
                      margin=dict(l=24, r=24, t=24, b=24))
    fig.layout.meta = {"placeholder": title}
    return fig


def is_placeholder(fig: go.Figure) -> bool:
    meta = fig.layout.meta
    return isinstance(meta, dict) and "placeholder" in meta


def _subplot_placeholder(fig: go.Figure, index: int, title: str, message: str,
                         theme: Theme) -> None:
    xref, yref = axis_ref(index)
    fig.add_annotation(text=f"<b>{title}</b><br>{_wrap_html(message, 48)}",
                       x=0.5, y=0.5, xref=f"{xref} domain", yref=f"{yref} domain",
                       showarrow=False, font=dict(color=theme.ink_secondary, size=12))
    fig.update_xaxes(visible=False, selector=dict(anchor=yref))
    fig.update_yaxes(visible=False, selector=dict(anchor=xref))


def _wrap_html(text: str, width: int = 72) -> str:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return "<br>".join(lines)


def _vertical_cursor(indices, t: float, theme: Theme) -> list[dict]:
    shapes = []
    for index in indices:
        xref, yref = axis_ref(index)
        shapes.append(dict(type="line", xref=xref, yref=f"{yref} domain",
                           x0=t, x1=t, y0=0, y1=1,
                           line=dict(color=theme.ink_secondary, width=1)))
    return shapes


def _horizontal_cursor(indices, t: float, theme: Theme) -> list[dict]:
    shapes = []
    for index in indices:
        xref, yref = axis_ref(index)
        shapes.append(dict(type="line", xref=f"{xref} domain", yref=yref,
                           x0=0, x1=1, y0=t, y1=t,
                           line=dict(color=theme.ink, width=1.5)))
    return shapes


def decimate_rows(times, values, max_rows: int = MAX_MAP_ROWS):
    """Evenly spaced rows of a space-time array, first and last kept."""
    if len(times) <= max_rows:
        return times, values
    picks = np.unique(np.linspace(0, len(times) - 1, max_rows).round().astype(int))
    return times[picks], values[picks]


def _fixed_scale(values, theme: Theme) -> dict:
    """`_signed_scale` with explicit limits, for a map whose data changes per frame."""
    scale = _signed_scale(values, theme)
    if "zmin" not in scale:
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size:
            scale.update(zmin=float(finite.min()), zmax=float(finite.max()))
    return scale


def _signed_scale(values, theme: Theme) -> dict:
    """Diverging around zero if the field changes sign, sequential if not."""
    finite = values[np.isfinite(values)]
    if finite.size and finite.min() < 0.0 < finite.max():
        limit = float(np.max(np.abs(finite)))
        return dict(colorscale=theme.diverging_scale, zmin=-limit, zmax=limit, zmid=0.0)
    return dict(colorscale=theme.sequential_scale)


def _grid(count: int, titles, *, shared: bool = True):
    rows = max(1, math.ceil(count / 2))
    cols = 1 if count == 1 else 2
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(titles),
                        shared_xaxes="all" if shared else False,
                        vertical_spacing=min(0.12, 0.30 / rows),
                        horizontal_spacing=0.09)
    return fig, rows, cols


def _position(index: int, cols: int) -> tuple[int, int]:
    return index // cols + 1, index % cols + 1


# ---------------------------------------------------------------------------
# fields at a time
# ---------------------------------------------------------------------------

def fields_figure(run: RunData, t: float | None = None, theme: Theme = LIGHT,
                  *, frame: bool = False) -> go.Figure:
    """h, u_m and every moment at the stored state nearest `t`.

    `frame=True` builds only what a frame patch carries (trace data, titles,
    shapes) and skips the run-wide axis ranges and the layout styling, which
    a patch never touches. Every builder that playback patches takes it.
    """
    values, actual, final = snapshot(run, t)
    series = field_columns(run)
    fig, rows, cols = _grid(len(series), [label for _, label in series])
    x = values[:, 0]
    color = theme.series(0)
    for position, (column, label) in enumerate(series):
        row, col = _position(position, cols)
        fig.add_trace(go.Scatter(
            x=x, y=values[:, _column_index(run, column)], mode="lines",
            name=column if column != "h" else "depth h",
            line=dict(color=color, width=2),
            showlegend=column == "h" and run.has_topography,
            hovertemplate=f"{column} = %{{y:.6g}}<extra></extra>"), row=row, col=col)
        if column == "h" and run.has_topography:
            fig.add_trace(go.Scatter(
                x=x, y=run.bed_elevation, mode="lines", name="bed Z",
                line=dict(color=theme.muted, width=1.5),
                hovertemplate="Z = %{y:.6g}<extra></extra>"), row=row, col=col)
        if not frame:
            fig.update_yaxes(range=field_range(run, column), row=row, col=col)
    fig.update_xaxes(title_text="x", row=rows)
    when = "final state" if final else "nearest stored step"
    return _finish(fig, theme, frame=frame, height=PANEL_HEIGHT * rows + 80,
                   title=f"t = {fmt(actual)} <span style='font-size:12px'>({when})</span>",
                   legend=run.has_topography)


# ---------------------------------------------------------------------------
# time histories
# ---------------------------------------------------------------------------

def _history_series(run: RunData) -> list[tuple[str, str]]:
    series = [("h", "water depth h"), ("u_m", "mean velocity u<sub>m</sub>")]
    if run.snapshots is not None:
        series.append(("q", "discharge q = h·u<sub>m</sub> (mean)"))
    series += [(name, moment_label(index))
               for index, name in enumerate(run.moment_columns, start=1)]
    return series


def discharge_history(run: RunData) -> tuple[np.ndarray, np.ndarray]:
    """Spatial mean of `h * u_m` per stored snapshot."""
    snapshots = run.snapshots
    return snapshots.times, np.mean(snapshots.field("h") * snapshots.field("u_m"), axis=1)


def histories_figure(run: RunData, t: float | None = None,
                     theme: Theme = LIGHT) -> go.Figure:
    summary = run.summary_history
    if summary is None:
        return placeholder(
            "Time histories not stored",
            "Set postprocessing.store_history: true (and history_stride) in the "
            "config and re-run.", theme)
    series = _history_series(run)
    fig, rows, cols = _grid(len(series), [label for _, label in series])
    time = summary["time"].to_numpy()
    color = theme.series(0)
    legend_shown = False
    for position, (column, _) in enumerate(series):
        row, col = _position(position, cols)
        if column == "q":
            times, values = discharge_history(run)
            fig.add_trace(go.Scatter(x=times, y=values, mode="lines", name="mean",
                                     line=dict(color=color, width=2), showlegend=False,
                                     hovertemplate="q = %{y:.6g}<extra></extra>"),
                          row=row, col=col)
            continue
        low, high = f"min_{column}", f"max_{column}"
        if {low, high} <= set(summary.columns):
            fig.add_trace(go.Scatter(x=time, y=summary[high], mode="lines",
                                     line=dict(width=0), showlegend=False,
                                     legendgroup="band", hoverinfo="skip"),
                          row=row, col=col)
            fig.add_trace(go.Scatter(x=time, y=summary[low], mode="lines",
                                     line=dict(width=0), fill="tonexty",
                                     fillcolor=theme.band, name="min–max",
                                     legendgroup="band", showlegend=not legend_shown,
                                     hoverinfo="skip"),
                          row=row, col=col)
        fig.add_trace(go.Scatter(x=time, y=summary[f"mean_{column}"], mode="lines",
                                 name="spatial mean", legendgroup="mean",
                                 line=dict(color=color, width=2),
                                 showlegend=not legend_shown and {low, high} <= set(summary.columns),
                                 hovertemplate=f"mean {column} = %{{y:.6g}}<extra></extra>"),
                      row=row, col=col)
        if {low, high} <= set(summary.columns):
            legend_shown = True
    fig.update_xaxes(title_text="t", row=rows)
    fig.update_layout(shapes=histories_cursor(run, t, theme))
    return _finish(fig, theme, height=PANEL_HEIGHT * rows + 80,
                   title=f"Spatial statistics over {len(summary)} recorded steps",
                   legend=True)


def histories_cursor(run: RunData, t: float | None, theme: Theme = LIGHT) -> list[dict]:
    if t is None or run.summary_history is None:
        return []
    return _vertical_cursor(range(1, len(_history_series(run)) + 1), t, theme)


# ---------------------------------------------------------------------------
# space-time
# ---------------------------------------------------------------------------

def space_time_fields(run: RunData) -> list[tuple[str, str]]:
    fields = field_columns(run)
    if run.has_topography:
        fields.insert(1, ("surface", "free surface h + Z"))
    fields.insert(2 if run.has_topography else 1, ("q", "discharge q = h·u<sub>m</sub>"))
    return fields


def space_time_values(run: RunData, field: str) -> np.ndarray:
    snapshots = run.snapshots
    if field == "surface":
        return snapshots.field("h") + run.bed_elevation[None, :]
    if field == "q":
        return snapshots.field("h") * snapshots.field("u_m")
    return snapshots.field(field)


def space_time_figure(run: RunData, field: str = "h", t: float | None = None,
                      theme: Theme = LIGHT) -> go.Figure:
    if run.snapshots is None:
        return placeholder("No field history",
                           "Space-time maps need postprocessing.store_history: true.",
                           theme)
    labels = dict(space_time_fields(run))
    if field not in labels:
        field = "h"
    snapshots = run.snapshots
    values = space_time_values(run, field)
    plain = labels[field].split("<")[0].strip()
    fig = go.Figure(go.Heatmap(
        x=snapshots.x, y=snapshots.times, z=values.astype(np.float32),
        colorbar=dict(title=dict(text=plain, side="right"), outlinewidth=0,
                      tickfont=dict(color=theme.muted)),
        hovertemplate="x = %{x:.4g}<br>t = %{y:.4g}<br>value = %{z:.6g}<extra></extra>",
        **_signed_scale(values, theme)))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="t")
    fig.update_layout(shapes=space_time_cursor(run, t, theme))
    _finish(fig, theme, height=520,
            title=f"{labels[field]} — {len(snapshots.steps)} of "
                  f"{snapshots.total_available} stored steps")
    fig.update_layout(hovermode="closest")
    return fig


def space_time_cursor(run: RunData, t: float | None, theme: Theme = LIGHT) -> list[dict]:
    if t is None or run.snapshots is None:
        return []
    return _horizontal_cursor([1], t, theme)


# ---------------------------------------------------------------------------
# vertical velocity profiles
# ---------------------------------------------------------------------------

def profile_cell(run: RunData, x: float | None) -> int:
    """Cell nearest `x`; by default the middle of the wet part of the domain.

    The default is taken from the *final* state, not the displayed one, so the
    chosen cell stays put while playing through time - a profile whose cell
    jumped between frames would show the jump, not the flow.
    """
    values = run.final_array()
    if x is not None:
        return int(np.argmin(np.abs(values[:, 0] - float(x))))
    wet = np.flatnonzero(values[:, 1] > run.effective_thresholds.h_wet)
    pool = wet if wet.size else np.arange(len(values))
    return int(pool[pool.size // 2])


def profile_lines_figure(run: RunData, t: float | None = None, x: float | None = None,
                         theme: Theme = LIGHT, *, frame: bool = False) -> go.Figure:
    """u(z) at one cell: at time `t`, and at that cell over time."""
    values, actual, final = snapshot(run, t)
    cell = profile_cell(run, x)
    x_cell = float(values[cell, 0])
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=[f"u(z) at x = {x_cell:.4g}, t = {fmt(actual, '.4g')}",
                                        f"u(z) at x = {x_cell:.4g} over time"])
    profile = run.velocity_profile(values[cell:cell + 1], Z_POINTS)[0]
    mean = float(values[cell, 2])
    fig.add_trace(go.Scatter(x=profile, y=Z_POINTS, mode="lines", name="u(z)",
                             line=dict(color=theme.series(run.order), width=2.5),
                             hovertemplate="u = %{x:.6g}<br>z = %{y:.3f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[mean, mean], y=[0, 1], mode="lines",
                             name="plug flow u<sub>m</sub>",
                             line=dict(color=theme.muted, width=1.5, dash="dot"),
                             hoverinfo="skip"), row=1, col=1)

    if run.snapshots is None:
        _subplot_placeholder(fig, 2, "No field history",
                             "Profiles over time need postprocessing.store_history: true.",
                             theme)
    else:
        snapshots = run.snapshots
        picks = np.unique(np.linspace(0, len(snapshots.times) - 1,
                                      min(PROFILE_CURVES, len(snapshots.times)))
                          .round().astype(int))
        ramp = _ordinal_ramp(theme, len(picks))
        for shade, index in zip(ramp, picks):
            curve = run.velocity_profile(snapshots.values[index][cell:cell + 1], Z_POINTS)[0]
            fig.add_trace(go.Scatter(
                x=curve, y=Z_POINTS, mode="lines",
                name=f"t = {snapshots.times[index]:.4g}", legendgroup="times",
                line=dict(color=shade, width=1.5),
                hovertemplate=f"t = {snapshots.times[index]:.4g}<br>u = %{{x:.6g}}<extra></extra>"),
                row=1, col=2)
        # Always present (not only when t is set), so the trace list does not
        # depend on t and a frame can be sent as a patch of the data alone.
        fig.add_trace(go.Scatter(
            x=profile, y=Z_POINTS, mode="lines", name=f"current t = {fmt(actual, '.4g')}",
            line=dict(color=theme.series(run.order), width=2.5),
            hovertemplate="u = %{x:.6g}<extra></extra>"), row=1, col=2)
    fig.update_xaxes(title_text="u(z)", range=profile_range(run, cell) if not frame else None)
    fig.update_yaxes(title_text="z  (0 = bed, 1 = surface)", row=1, col=1, range=[0, 1])
    _finish(fig, theme, frame=frame, height=420, legend=True)
    fig.update_layout(hovermode="closest",
                      legend=dict(orientation="v", yanchor="top", y=1.0,
                                  xanchor="left", x=1.02))
    return fig


def cell_history(run: RunData, cell: int) -> np.ndarray:
    """`[x, h, u_m, a_1..a_N]` of one cell at every stored step and at the end."""
    final = run.final_array()[cell:cell + 1]
    if run.snapshots is None:
        return final
    return np.vstack([run.snapshots.values[:, cell, :], final])


def profile_range(run: RunData, cell: int, *, deviation: bool = False) -> list[float] | None:
    """Fixed `u` axis for one cell's profile: every value it takes over the run.

    One call to the profile routine: the cell's history is passed as if each
    stored step were a cell, which is exactly the row layout it expects.
    """
    history = cell_history(run, cell)
    profiles = run.velocity_profile(history, Z_POINTS)
    if deviation:
        profiles = profiles - history[:, 2][:, None]
    return padded_range(profiles)


def _ordinal_ramp(theme: Theme, count: int) -> list[str]:
    """`count` shades of the sequential hue, skipping the steps that vanish
    into the surface, early times lighter (light theme) or darker (dark)."""
    usable = theme.sequential[2:] if theme.name == "light" else theme.sequential[1:-1]
    if count <= 1:
        return [usable[-1]]
    picks = np.linspace(0, len(usable) - 1, count).round().astype(int)
    return [usable[i] for i in picks]


def _profile_map_values(run: RunData, values, kind: str) -> np.ndarray:
    profiles = run.velocity_profile(values, Z_POINTS)
    return profiles - values[:, 2][:, None] if kind == "deviation" else profiles


def profile_map_figure(run: RunData, t: float | None = None, x: float | None = None,
                       kind: str = "velocity", theme: Theme = LIGHT, *,
                       frame: bool = False) -> go.Figure:
    """u(x, z), or its departure from plug flow, at time `t`."""
    values, actual, _ = snapshot(run, t)
    if kind == "deviation" and run.order == 0:
        return placeholder(
            "No vertical structure at N = 0",
            "The N = 0 model carries no moments, so u(z) is the constant u_m and "
            "the departure from plug flow is identically zero.", theme, height=380)
    data = _profile_map_values(run, values, kind)
    if kind == "deviation":
        title, bar = "departure from plug flow u − u<sub>m</sub>", "u − u_m"
    else:
        title, bar = "velocity u(x, z)", "u"
    # The colour scale spans a sample of the whole run, not this frame, so a
    # colour means the same velocity at every time.
    states = [run.final_array()] if not frame else [values]
    if not frame and run.snapshots is not None:
        states += [run.snapshots.values[i] for i in
                   sample_indices(len(run.snapshots.times), PROFILE_MAP_SAMPLES)]
    sample = np.concatenate([_profile_map_values(run, state, kind).ravel()
                             for state in states])
    fig = go.Figure(go.Heatmap(
        x=values[:, 0], y=Z_POINTS, z=data.T.astype(np.float32),
        colorbar=dict(title=dict(text=bar, side="right"), outlinewidth=0,
                      tickfont=dict(color=theme.muted)),
        hovertemplate="x = %{x:.4g}<br>z = %{y:.3f}<br>value = %{z:.6g}<extra></extra>",
        **_fixed_scale(sample, theme)))
    cell = profile_cell(run, x)
    fig.update_layout(shapes=[dict(type="line", xref="x", yref="y domain",
                                   x0=values[cell, 0], x1=values[cell, 0], y0=0, y1=1,
                                   line=dict(color=theme.ink, width=1.5))])
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="z")
    _finish(fig, theme, frame=frame, height=380, title=f"{title} at t = {fmt(actual, '.4g')}")
    fig.update_layout(hovermode="closest")
    return fig


# ---------------------------------------------------------------------------
# hyperbolicity
# ---------------------------------------------------------------------------

def hyperbolicity_series_figure(derived, tolerance: float, t: float | None = None,
                                theme: Theme = LIGHT) -> go.Figure:
    """Per-step counts and the largest |Im lambda|, from the derived summary."""
    if derived is None or derived.empty:
        return placeholder(
            "Model-level hyperbolicity not captured",
            "Set postprocessing.store_hyperbolicity: true (and hyperbolicity_stride) "
            "in the config and re-run to capture the cell-by-cell spectrum of A(U).",
            theme)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14,
                        subplot_titles=["cells per recorded step",
                                        "largest |Im λ| over evaluated cells"])
    time = derived["time"].to_numpy()
    fig.add_trace(go.Scatter(x=time, y=derived["num_nonhyperbolic_cells"], mode="lines",
                             name="non-hyperbolic", line=dict(color=theme.critical, width=2),
                             hovertemplate="non-hyperbolic: %{y}<extra></extra>"),
                  row=1, col=1)
    if "num_not_evaluated_cells" in derived:
        fig.add_trace(go.Scatter(x=time, y=derived["num_not_evaluated_cells"], mode="lines",
                                 name="not evaluated (dry or failed)",
                                 line=dict(color=theme.muted, width=2),
                                 hovertemplate="not evaluated: %{y}<extra></extra>"),
                      row=1, col=1)
    magnitude = derived["max_abs_imag_eig"].to_numpy(dtype=np.float64)
    finite = np.isfinite(magnitude)
    peak = float(magnitude[finite].max()) if finite.any() else 0.0
    shown = np.where(magnitude > 0.0, magnitude, np.nan) if peak > 0 else magnitude
    fig.add_trace(go.Scatter(x=time, y=shown, mode="lines", name="max |Im λ|",
                             line=dict(color=theme.series(0), width=2),
                             hovertemplate="max |Im λ| = %{y:.3e}<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=[time[0], time[-1]], y=[tolerance, tolerance],
                             mode="lines", name="tolerance",
                             line=dict(color=theme.ink_secondary, width=1, dash="dash"),
                             hoverinfo="skip"), row=2, col=1)
    if peak > 0:
        fig.update_yaxes(type="log", row=2, col=1)
    else:
        # A clean run is identically zero: give it a range in units of the
        # tolerance so the curve sits on the axis, visibly under the line.
        fig.update_yaxes(range=[-0.2 * tolerance, 4.0 * tolerance], row=2, col=1)
    fig.update_xaxes(title_text="t", row=2, col=1)
    fig.update_layout(shapes=hyperbolicity_series_cursor(derived, t, theme))
    return _finish(fig, theme, height=520, legend=True)


def hyperbolicity_series_cursor(derived, t, theme: Theme = LIGHT) -> list[dict]:
    if t is None or derived is None or derived.empty:
        return []
    return _vertical_cursor([1, 2], t, theme)


def _cell_grid(cells, column: str, values=None):
    """(x, t, array) of one per-cell column, NaN where never evaluated."""
    frame = cells if values is None else cells.assign(_value=values)
    table = frame.pivot_table(index="time", columns="x",
                              values=column if values is None else "_value",
                              aggfunc="max", dropna=False)
    return table.columns.to_numpy(), table.index.to_numpy(), table.to_numpy()


def hyperbolicity_map_figure(cells, tolerance: float, kind: str = "magnitude",
                             t: float | None = None, theme: Theme = LIGHT) -> go.Figure:
    """|Im lambda| per cell over time, or the hyperbolic/lost classification.

    Blank cells were never evaluated (dry, or a failed eigensolve); they are
    not zeros and not failures.
    """
    if cells is None or len(cells) == 0:
        return placeholder(
            "No per-cell spectra",
            "Set postprocessing.store_hyperbolicity: true in the config and re-run.",
            theme)
    magnitude = cells["max_abs_imag_eig"].to_numpy(dtype=np.float64)
    finite = np.isfinite(magnitude)
    peak = float(magnitude[finite].max()) if finite.any() else 0.0

    if kind == "classification":
        state = np.where(~finite, np.nan, np.where(magnitude > tolerance, 0.0, 1.0))
        x, times, grid = _cell_grid(cells, None, state)
        times, grid = decimate_rows(times, grid)
        trace = go.Heatmap(
            x=x, y=times, z=grid.astype(np.float32), zmin=0, zmax=1,
            colorscale=[[0, theme.critical], [0.5, theme.critical],
                        [0.5, theme.good], [1, theme.good]],
            colorbar=dict(tickvals=[0.25, 0.75], ticktext=["lost", "hyperbolic"],
                          outlinewidth=0, tickfont=dict(color=theme.ink_secondary)),
            hovertemplate="x = %{x:.4g}<br>t = %{y:.4g}<br>"
                          "1 = hyperbolic, 0 = lost: %{z}<extra></extra>")
        title = "cell classification (blank = not evaluated)"
    else:
        x, times, grid = _cell_grid(cells, "max_abs_imag_eig")
        times, grid = decimate_rows(times, grid)
        if peak > 0:
            logs = np.log10(np.maximum(grid, MAGNITUDE_FLOOR))
            low = math.floor(np.nanmin(logs)) if np.isfinite(logs).any() else -16
            high = math.ceil(np.nanmax(logs)) if np.isfinite(logs).any() else 0
            ticks = list(range(low, high + 1, max(1, (high - low) // 6 or 1)))
            colorbar = dict(tickvals=ticks, ticktext=[f"1e{k}" for k in ticks],
                            title=dict(text="|Im λ|", side="right"), outlinewidth=0,
                            tickfont=dict(color=theme.muted))
            z, extra = logs, dict(zmin=low, zmax=high)
            hover = "log10 |Im λ| = %{z:.3f}"
        else:
            colorbar = dict(title=dict(text="|Im λ|", side="right"), outlinewidth=0,
                            tickfont=dict(color=theme.muted))
            z, extra = grid, dict(zmin=0.0, zmax=max(peak, tolerance))
            hover = "|Im λ| = %{z:.3e}"
        trace = go.Heatmap(x=x, y=times, z=z.astype(np.float32),
                           colorscale=theme.sequential_alt_scale, colorbar=colorbar,
                           hovertemplate=f"x = %{{x:.4g}}<br>t = %{{y:.4g}}<br>{hover}"
                                         "<extra></extra>", **extra)
        title = "magnitude of the imaginary part (blank = not evaluated)"
        if peak == 0:
            title += " — every evaluated cell is exactly zero"
    fig = go.Figure(trace)
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="t")
    fig.update_layout(shapes=_horizontal_cursor([1], t, theme) if t is not None else [])
    _finish(fig, theme, height=420, title=title)
    fig.update_layout(hovermode="closest")
    return fig


def hyperbolicity_map_cursor(t, theme: Theme = LIGHT) -> list[dict]:
    return [] if t is None else _horizontal_cursor([1], t, theme)


def spectrum_at(cells, t: float | None = None, x: float | None = None):
    """The recorded spectrum of one cell: `(row, eigenvalues)` or `None`.

    The step is the recorded one nearest `t`; the cell is the one nearest `x`,
    or by default the step's worst evaluated cell.
    """
    if cells is None or len(cells) == 0 or "eigvals_real" not in cells:
        return None
    times = np.unique(cells["time"].to_numpy())
    moment = times[-1] if t is None else times[np.argmin(np.abs(times - float(t)))]
    rows = cells[cells["time"] == moment]
    if x is not None:
        row = rows.iloc[int(np.argmin(np.abs(rows["x"].to_numpy() - float(x))))]
    else:
        magnitude = rows["max_abs_imag_eig"].to_numpy(dtype=np.float64)
        if not np.isfinite(magnitude).any():
            row = rows.iloc[len(rows) // 2]
        else:
            row = rows.iloc[int(np.nanargmax(np.where(np.isfinite(magnitude),
                                                      magnitude, -np.inf)))]
    real = parse_eigenvalue_column([row["eigvals_real"]])[0]
    imag = parse_eigenvalue_column([row["eigvals_imag"]])[0]
    if real.size == 0:
        return row, None
    if imag.size != real.size:
        imag = np.zeros_like(real)
    return row, real + 1j * imag


def spectrum_figure(cells, tolerance: float, t: float | None = None,
                    x: float | None = None, theme: Theme = LIGHT) -> go.Figure:
    found = spectrum_at(cells, t, x)
    if found is None:
        return placeholder("No per-cell spectra",
                           "The eigenvalues of A(U) need store_hyperbolicity: true.",
                           theme, height=380)
    row, eigenvalues = found
    where = f"x = {row['x']:.4g}, t = {row['time']:.4g}"
    if eigenvalues is None:
        return placeholder("Cell not evaluated",
                           f"The cell at {where} was dry or its eigensolve failed, so "
                           "no spectrum was recorded. Its moments were ramped away, so "
                           "it says nothing about the model.", theme, height=380)
    lost = np.abs(eigenvalues.imag) > tolerance
    fig = go.Figure()
    for mask, name, color in [(~lost, "real (hyperbolic)", theme.series(0)),
                              (lost, "complex (lost)", theme.critical)]:
        if mask.any():
            fig.add_trace(go.Scatter(
                x=eigenvalues.real[mask], y=eigenvalues.imag[mask], mode="markers",
                name=name, marker=dict(size=10, color=color,
                                       line=dict(width=2, color=theme.surface)),
                hovertemplate="Re λ = %{x:.6g}<br>Im λ = %{y:.3e}<extra></extra>"))
    span = float(np.max(np.abs(eigenvalues.imag))) if eigenvalues.size else 0.0
    if span <= tolerance:
        fig.update_yaxes(range=[-1, 1])
    fig.update_xaxes(title_text="Re λ")
    fig.update_yaxes(title_text="Im λ", zeroline=True, zerolinecolor=theme.axis)
    _finish(fig, theme, height=380, title=f"eigenvalues of A(U) at {where}", legend=True)
    fig.update_layout(hovermode="closest")
    return fig


# ---------------------------------------------------------------------------
# wet-dry
# ---------------------------------------------------------------------------

def wet_dry_depth_figure(run: RunData, t: float | None = None,
                         theme: Theme = LIGHT, *, frame: bool = False) -> go.Figure:
    values, actual, _ = snapshot(run, t)
    thresholds = run.effective_thresholds
    x, h = values[:, 0], values[:, 1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=np.maximum(h, 1e-300), mode="lines", name="depth h",
                             line=dict(color=theme.series(0), width=2),
                             hovertemplate="h = %{y:.4g}<extra></extra>"))
    for value, name, color, dash in [(thresholds.h_wet, "h_wet", theme.ink_secondary, "dash"),
                                     (thresholds.h_dry, "h_dry", theme.critical, "dot")]:
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[value, value], mode="lines",
                                 name=f"{name} = {value:.3g}",
                                 line=dict(color=color, width=1.2, dash=dash),
                                 hoverinfo="skip"))
    # Dry cells hold exactly zero; the view stops three decades under h_dry
    # rather than running 300 decades down to the smallest double.
    peak = float(np.nanmax(field_values(run, "h") if not frame else h))
    top = max(1.6 * peak, 10.0 * thresholds.h_wet)
    fig.update_yaxes(type="log", range=[math.log10(thresholds.h_dry * 1e-3),
                                        math.log10(top)], title_text="depth (log)")
    fig.update_xaxes(title_text="x")
    return _finish(fig, theme, frame=frame, height=380, legend=True,
                   title=f"depth against the thresholds at t = {fmt(actual, '.4g')}")


def wet_dry_state(run: RunData, depths: np.ndarray) -> np.ndarray:
    """0 = dry (h <= h_dry), 1 = transition, 2 = wet (h >= h_wet)."""
    thresholds = run.effective_thresholds
    state = np.full(depths.shape, 2.0)
    state[depths < thresholds.h_wet] = 1.0
    state[depths <= thresholds.h_dry] = 0.0
    return state


def wet_dry_map_figure(run: RunData, t: float | None = None,
                       theme: Theme = LIGHT) -> go.Figure:
    if run.snapshots is None:
        return placeholder("No field history",
                           "The wet-dry map over time needs store_history: true.", theme)
    state = wet_dry_state(run, run.snapshots.field("h"))
    low, mid, high = theme.ordinal3
    fig = go.Figure(go.Heatmap(
        x=run.snapshots.x, y=run.snapshots.times, z=state.astype(np.float32),
        zmin=0, zmax=2,
        colorscale=[[0, low], [1 / 3, low], [1 / 3, mid], [2 / 3, mid],
                    [2 / 3, high], [1, high]],
        colorbar=dict(tickvals=[1 / 3, 1, 5 / 3], ticktext=["dry", "transition", "wet"],
                      outlinewidth=0, tickfont=dict(color=theme.ink_secondary)),
        hovertemplate="x = %{x:.4g}<br>t = %{y:.4g}<br>0 dry · 1 transition · 2 wet: "
                      "%{z}<extra></extra>"))
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="t")
    fig.update_layout(shapes=space_time_cursor(run, t, theme))
    _finish(fig, theme, height=420, title="wet-dry state over time")
    fig.update_layout(hovermode="closest")
    return fig


def wet_dry_counts(run: RunData, t: float | None = None) -> list[tuple[str, int, float]]:
    values, _, _ = snapshot(run, t)
    state = wet_dry_state(run, values[:, 1])
    total = max(len(state), 1)
    return [(label, int(np.sum(state == code)), 100.0 * np.sum(state == code) / total)
            for code, label in [(0, "dry (h ≤ h_dry)"), (1, "transition"),
                                (2, "wet (h ≥ h_wet)")]]


# ---------------------------------------------------------------------------
# topography
# ---------------------------------------------------------------------------

def topography_residual(run: RunData, t: float | None = None) -> float | None:
    """max |h + Z - H| at the stored state nearest `t`, if H was recorded."""
    reference = run.meta.reference_water_level
    if reference is None or not run.has_topography:
        return None
    values, _, _ = snapshot(run, t)
    return float(np.max(np.abs(values[:, 1] + run.bed_elevation - reference)))


def topography_figure(run: RunData, t: float | None = None,
                      theme: Theme = LIGHT, *, frame: bool = False) -> go.Figure:
    """Free surface over the bed, its departure from H, and the bed itself.

    `h + Z` rather than bare `h`: over a bump a lake at rest looks like an
    inverted bump in `h`, which reads as a failure when it is the right answer.
    """
    if not run.has_topography:
        return placeholder("Flat bed", "This run has no topography.", theme)
    values, actual, _ = snapshot(run, t)
    x, h, bed = values[:, 0], values[:, 1], run.bed_elevation
    surface = h + bed
    reference = run.meta.reference_water_level
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=[f"free surface over the bed at t = {fmt(actual, '.4g')}",
                                        "departure from the still water level h + Z − H",
                                        "bed Z, re-derived from the recorded profile"])
    fig.add_trace(go.Scatter(x=x, y=bed, mode="lines", name="bed Z", fill="tozeroy",
                             line=dict(color=theme.muted, width=1),
                             fillcolor=theme.grid,
                             hovertemplate="Z = %{y:.6g}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=surface, mode="lines", name="free surface h + Z",
                             fill="tonexty", fillcolor=theme.band,
                             line=dict(color=theme.series(0), width=2),
                             hovertemplate="h + Z = %{y:.6g}<extra></extra>"), row=1, col=1)
    if reference is None:
        _subplot_placeholder(fig, 2, "No reference water level recorded",
                             "The h + Z − H residual needs topography.reference_water_level "
                             "from the run's sidecar.", theme)
    else:
        fig.add_trace(go.Scatter(x=x, y=surface - reference, mode="lines",
                                 name=f"h + Z − H (H = {reference:g})",
                                 line=dict(color=theme.series(0), width=2),
                                 showlegend=False,
                                 hovertemplate="h + Z − H = %{y:.3e}<extra></extra>"),
                      row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=bed, mode="lines", name="bed", showlegend=False,
                             line=dict(color=theme.muted, width=2),
                             hovertemplate="Z = %{y:.6g}<extra></extra>"), row=3, col=1)
    # Fixed over the run: the bed is filled from zero, so zero is in range.
    surfaces = field_values(run, "h").reshape(-1, len(x)) + bed[None, :] if not frame else None
    if not frame:
        fig.update_yaxes(range=padded_range(np.concatenate([surfaces.ravel(), bed, [0.0]])),
                         row=1, col=1)
    if not frame and reference is not None:
        peak = float(np.nanmax(np.abs(surfaces - reference)))
        limit = 1.1 * peak if peak > 0 else 1e-15
        fig.update_yaxes(range=[-limit, limit], row=2, col=1)
    fig.update_xaxes(title_text="x", row=3, col=1)
    return _finish(fig, theme, frame=frame, height=720, legend=True)
