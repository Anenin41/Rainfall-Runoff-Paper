"""The Dash app: layout, and callbacks that wire the controls to the figures.

Two shared cursors drive every view:

- **time** - one slider over a time grid (the selected run's stored steps, or
  in compare mode every run's steps inside their common range). It is moved by
  dragging, by the play button, and by clicking any space-time map.
- **position** - an `x` that picks the cell for the vertical profile, the
  spectrum and the point time series. Blank means "a sensible default"; it is
  set by typing or by clicking a map.

The view functions (`view_*`) are plain functions of a `RunStore` and the
control values, so tests call them without a browser. The Dash callbacks are
thin wrappers that only add the partial-update logic: when nothing but the time
changed, a heavy figure is sent a new cursor (`Patch`) instead of rebuilt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from _plotly_utils.utils import to_typed_array_spec
from dash import ALL, Dash, Input, Output, Patch, State, ctx, dcc, html, no_update

from ..report.hyperbolicity import STALE_SUMMARY_NOTE, model_level_lines, scheme_counter_lines
from ..report.style import Prose
from . import compare as cmp
from . import figures as fig
from . import theme as themes
from .catalog import RunEntry, families
from .store import RunStore

ASSETS = Path(__file__).parent / "assets"
GRAPH_CONFIG = {"displaylogo": False,
                "toImageButtonOptions": {"format": "svg", "scale": 1}}
MAX_TIME_GRID = 400
PLAY_INTERVAL_MS = 300
SPEEDS = {"0.25×": 1200, "0.5×": 600, "1×": PLAY_INTERVAL_MS, "2×": 150}
COMPARE_MIN_INTERVAL_MS = 300
MAP_GRAPHS = ("space-time-graph", "wet-dry-map-graph", "hyperbolicity-map-graph")

TABS = [("overview", "Overview"), ("fields", "Fields"), ("histories", "Histories"),
        ("space-time", "Space-time"), ("profiles", "Profiles"),
        ("hyperbolicity", "Hyperbolicity"), ("wet-dry", "Wet-dry"),
        ("topography", "Topography")]


# ---------------------------------------------------------------------------
# small pieces
# ---------------------------------------------------------------------------

def _loading(children) -> dcc.Loading:
    """A spinner that never hides what is already on screen.

    `dcc.Loading` hides its children while any of them is updating (its
    default overlay is `visibility: hidden`), which blanked every plot on
    every time step. Here the plot stays visible, dimmed only if an update is
    slow enough to notice - a first load, not a frame.
    """
    return dcc.Loading(children, type="dot", delay_show=400,
                       overlay_style={"visibility": "visible", "opacity": 0.55})


def _graph(id_: str, **kwargs) -> dcc.Graph:
    return dcc.Graph(id=id_, config=GRAPH_CONFIG, className="graph", **kwargs)


def _size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


def _note(text: str, kind: str = "note"):
    return html.Div(text, className=f"callout callout-{kind}")


def _error_panel(message: str):
    return html.Div([html.Strong("This run could not be loaded."), html.P(message)],
                    className="callout callout-error")


def _labels(entries: list[RunEntry]) -> list[str]:
    names = [entry.label for entry in entries]
    return [f"{entry.family}/{entry.label}" if names.count(entry.label) > 1 and entry.family
            else entry.label for entry in entries]


def lines_to_html(lines) -> list:
    """The report's text blocks as HTML.

    The report writes UPPERCASE headings, `  key   value` rows whose column
    alignment carries meaning, and `Prose` paragraphs it may re-wrap. Each
    becomes the HTML element that does the same job.
    """
    out, rows = [], []

    def flush():
        if rows:
            out.append(html.Table(html.Tbody([html.Tr([html.Th(k), html.Td(v)])
                                              for k, v in rows]), className="kv"))
            rows.clear()

    for line in lines:
        if isinstance(line, Prose):
            flush()
            out.append(html.P(str(line).lstrip("- ").strip()))
        elif not line.strip():
            continue
        elif line.isupper() or (line == line.upper() and not line.startswith(" ")):
            flush()
            out.append(html.H4(line.title()))
        elif line.startswith("  "):
            parts = line.strip().split("  ", 1)
            rows.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))
        else:
            flush()
            out.append(html.P(line))
    flush()
    return out


# ---------------------------------------------------------------------------
# view functions (plain, testable)
# ---------------------------------------------------------------------------

def load(store: RunStore, key: str | None):
    """`(run, error)`: never raises, so a broken run cannot break the page."""
    if not key:
        return None, "No run selected."
    try:
        return store.run(key), None
    except (FileNotFoundError, ValueError, KeyError) as error:
        return None, str(error)


def view_run_list(store: RunStore, search: str | None, selected: str | None,
                  compare_set) -> list:
    compare_set = set(compare_set or [])
    needle = (search or "").strip().lower()
    groups = []
    for family, entries in families(store.entries()).items():
        shown = [entry for entry in entries
                 if not needle or needle in f"{family} {entry.name} {entry.config_name}".lower()]
        if not shown:
            continue
        rows = []
        for entry in shown:
            badges = [html.Span(f"N={entry.order}", className="badge"),
                      html.Span(entry.closure, className="badge"),
                      html.Span(entry.scheme or "?", className="badge badge-muted")]
            if not entry.has_history:
                badges.append(html.Span("no history", className="badge badge-muted"))
            if entry.has_hyperbolicity:
                badges.append(html.Span("λ", className="badge badge-muted",
                                        title="per-cell spectra recorded"))
            in_compare = entry.key in compare_set
            rows.append(html.Div([
                html.Button([html.Div(entry.name, className="run-name"),
                             html.Div(badges, className="run-badges"),
                             html.Div(f"{entry.config_name or 'no sidecar'} · "
                                      f"{_size(entry.size_bytes)}", className="run-sub")],
                            id={"type": "run-select", "key": entry.key},
                            className="run-button" + (" selected" if entry.key == selected else ""),
                            n_clicks=0),
                html.Button("✓" if in_compare else "+",
                            id={"type": "run-compare", "key": entry.key},
                            className="compare-toggle" + (" on" if in_compare else ""),
                            title="remove from comparison" if in_compare else "add to comparison",
                            n_clicks=0),
            ], className="run-row"))
        header = [html.Span(family or "runs", className="family-name")]
        if len(shown) > 1:
            header.append(html.Button("compare all", className="link-button",
                                      id={"type": "family-compare", "family": family},
                                      n_clicks=0))
        groups.append(html.Div([html.Div(header, className="family-header"), *rows],
                               className="family"))
    if not groups:
        return [html.P("No runs match." if needle else
                       f"No run output under {store.root}.", className="muted")]
    return groups


def time_grid(store: RunStore, mode: str, selected: str | None, compare_set) -> list[float]:
    """The times the slider can take, ascending, at most `MAX_TIME_GRID`."""

    if mode == "compare":
        items = compared(store, compare_set)
        span = cmp.common_time_range(items)
        times = np.unique(np.concatenate(
            [item.run.snapshots.times for item in items if item.run.snapshots is not None]
            or [np.zeros(0)]))
        if span is not None:
            times = times[(times >= span[0]) & (times <= span[1])]
    else:
        run, _ = load(store, selected)
        if run is None:
            return [0.0]
        if run.snapshots is None:
            return [float(run.meta.t_end or 0.0)]
        times = run.snapshots.times
    if len(times) == 0:
        return [0.0]
    if len(times) > MAX_TIME_GRID:
        picks = np.unique(np.linspace(0, len(times) - 1, MAX_TIME_GRID).round().astype(int))
        times = times[picks]
    return [float(value) for value in times]


def compared(store: RunStore, compare_set) -> list[cmp.Compared]:
    entries = []
    for key in list(compare_set or [])[:cmp.MAX_RUNS]:
        try:
            entries.append(store.entry(key))
        except KeyError:
            continue
    items = []
    for entry, label in zip(entries, _labels(entries)):
        run, _ = load(store, entry.key)
        if run is not None:
            items.append(cmp.Compared(run, label))
    return items


def view_overview(store: RunStore, key: str | None):
    run, error = load(store, key)
    if run is None:
        return _error_panel(error)
    entry = store.entry(key)
    meta = run.meta
    model = [("type", meta.model), ("closure", "HSWME (hyperbolic)" if meta.hyperbolic
                                    else "SWME" if meta.hyperbolic is not None else None),
             ("moment order N", run.order), ("initial condition", meta.initial_condition),
             ("viscosity", meta.viscosity), ("slip length", meta.slip_length)]
    if meta.model == "RechargeSWME1D":
        model += [("infiltration", meta.infiltration_type),
                  ("rainfall rate", meta.rainfall_rate)]
    discretisation = [("flux scheme", meta.scheme),
                      ("well balanced", meta.scheme_well_balanced),
                      ("time integrator", meta.time_integrator), ("cells", run.n_cells),
                      ("domain", meta.domain), ("boundary", meta.boundary_condition),
                      ("t_end", meta.t_end), ("wall time (s)", meta.elapsed_seconds)]
    thresholds = run.effective_thresholds
    wet_dry = [("h_dry", thresholds.h_dry), ("h_wet", thresholds.h_wet),
               ("minimum depth", run.min_depth), ("went dry", run.went_dry)]
    if meta.bed_profile:
        wet_dry += [("bed profile", meta.bed_profile)]

    def table(title, rows):
        return html.Div([html.H4(title), html.Table(html.Tbody(
            [html.Tr([html.Th(name), html.Td(fig.fmt(value))]) for name, value in rows]),
            className="kv")], className="card")

    inventory = html.Div([html.H4("Data available"), html.Table(html.Tbody([
        html.Tr([html.Td("✓" if present else "–",
                         className="ok" if present else "missing"),
                 html.Th(label), html.Td(note)])
        for label, present, note in run.inventory()]), className="kv")], className="card")

    notes = [html.Li(text) for text in run.warnings] or [
        html.Li("All expected outputs were present.")]
    log_path = entry.directory / "run.log"
    log = []
    if log_path.exists():
        text = log_path.read_text(errors="replace")[-4000:]
        log = [html.Div([html.H4("run.log"), html.Pre(text)], className="card wide")]

    return html.Div([
        html.H2(run.title, className="run-title"),
        html.P(f"{entry.directory}  ·  {meta.generated_at or ''}", className="muted"),
        html.Div([table("Model", model), table("Discretisation", discretisation),
                  table("Wet-dry and bed", wet_dry), inventory], className="cards"),
        html.Div([html.H4("Notes"), html.Ul(notes)], className="card wide"),
        *log,
    ])


def view_hyperbolicity_text(store: RunStore, key: str):
    view = store.hyperbolicity(key)
    blocks = [html.Div(lines_to_html(scheme_counter_lines(view.run)), className="card"),
              html.Div(lines_to_html(model_level_lines(view.run)), className="card")]
    extra = []
    if view.derived is not None and not view.corrected:
        extra.append(_note(STALE_SUMMARY_NOTE, "warning"))
    return [html.Div(blocks, className="cards cards-2"), *extra]


def view_topography_caption(store: RunStore, key: str, t):
    run, _ = load(store, key)
    if run is None or not run.has_topography:
        return ""
    peak = fig.topography_residual(run, t)
    scheme = run.meta.scheme or "an unrecorded scheme"
    balanced = fig.fmt(run.meta.scheme_well_balanced)
    lead = (f"max |h + Z − H| = {peak:.3e} at this time. " if peak is not None
            else "No reference water level was recorded. ")
    return _note(lead + "For a lake-at-rest case this is the C-property residual and "
                 "should sit at round-off. It can only do so if the scheme's viscosity "
                 "polynomial satisfies P(0) = 0, which holds for Roe and Osher but not for "
                 f"LF or PRICE. This run used {scheme} (well balanced: {balanced}). The CSVs "
                 "carry no bed column: the bed is the sidecar's profile evaluated at the "
                 "cell centres.")


def view_wet_dry_counts(store: RunStore, key: str, t):
    run, _ = load(store, key)
    if run is None:
        return ""
    rows = [html.Tr([html.Th(label), html.Td(f"{count}"), html.Td(f"{share:.1f} %")])
            for label, count, share in fig.wet_dry_counts(run, t)]
    rows.append(html.Tr([html.Th("minimum depth over the run"),
                         html.Td(fig.fmt(run.min_depth)), html.Td("")]))
    if run.meta.mass_created_by_clamping:
        rows.append(html.Tr([html.Th("mass added by clamping"),
                             html.Td(f"{run.meta.mass_created_by_clamping:.3e}"),
                             html.Td("")]))
    return html.Div([html.H4("Cell counts at this time"),
                     html.Table(html.Tbody(rows), className="kv")], className="card")


def play_interval(speed: str | None, mode: str | None) -> int:
    """Milliseconds between frames.

    A compare frame redraws four multi-run figures, ~200 ms of server work,
    so compare mode never ticks faster than `COMPARE_MIN_INTERVAL_MS`: a tick
    that arrives before the last frame is drawn only queues up behind it,
    and playback would then lag the slider instead of keeping pace.
    """
    interval = SPEEDS.get(speed or "", PLAY_INTERVAL_MS)
    return max(interval, COMPARE_MIN_INTERVAL_MS) if mode == "compare" else interval


def transport(trigger, index, grid, current, playing: bool, clicked_t=None):
    """The time controls as a pure function: `(position, t, playing)`.

    `trigger` is the id of whatever moved the time. Playback stops on the last
    stored step rather than wrapping, so the end state stays on screen; play
    pressed there starts again from the first. Stepping pauses playback, so a
    step is never immediately overtaken by the next tick.
    """
    grid = grid or [0.0]
    last = len(grid) - 1
    index = int(min(max(index or 0, 0), last))

    def at(position, still_playing=playing):
        position = int(min(max(position, 0), last))
        return position, float(grid[position]), still_playing

    if trigger == "time-index":
        return at(index)
    if trigger == "play":
        if playing:
            return at(index, False)
        return at(0 if index >= last else index, last > 0)
    if trigger == "player":
        if not playing:
            return at(index, False)
        return at(index + 1, index + 1 < last)
    if trigger == "step-back":
        return at(index - 1, False)
    if trigger == "step-forward":
        return at(index + 1, False)
    if trigger in MAP_GRAPHS:
        if clicked_t is None:
            return at(index)
        return at(int(np.argmin(np.abs(np.asarray(grid) - float(clicked_t)))))
    # The grid changed (new run, new comparison): stay at the same time if
    # there is one, otherwise start at the end - where a report would.
    if current is None:
        return at(last)
    return at(int(np.argmin(np.abs(np.asarray(grid) - float(current)))))


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------

def _sidebar():
    return html.Aside([
        html.Div([
            dcc.Input(id="run-search", type="search", placeholder="Filter runs…",
                      debounce=True, className="search"),
            html.Button("↻", id="reload", title="re-scan the results directory",
                        className="icon-button", n_clicks=0),
        ], className="sidebar-tools"),
        html.Div(id="run-list", className="run-list"),
        html.Div([
            html.Span(id="compare-count", className="muted"),
            html.Button("clear", id="compare-clear", className="link-button", n_clicks=0),
        ], className="compare-footer"),
    ], className="sidebar")


def _timebar():
    return html.Div([
        html.Div([
            html.Button("◀", id="step-back", className="icon-button", n_clicks=0,
                        title="one stored step back"),
            html.Button("▶", id="play", className="icon-button", n_clicks=0,
                        title="play through the stored steps"),
            html.Button("▶|", id="step-forward", className="icon-button", n_clicks=0,
                        title="one stored step forward"),
            dcc.Dropdown(id="speed", value="1×", clearable=False, searchable=False,
                         options=list(SPEEDS), className="speed"),
        ], className="transport"),
        html.Div(dcc.Slider(id="time-index", min=0, max=0, step=1, value=0,
                            marks=None, updatemode="mouseup"), className="slider"),
        html.Span(id="time-label", className="time-label"),
        html.Label(["x ", dcc.Input(id="x-input", type="number", debounce=True,
                                    placeholder="auto", className="x-input")],
                   className="x-label",
                   title="position for profiles, spectra and point series; click a map to set it"),
        dcc.Interval(id="player", interval=PLAY_INTERVAL_MS, disabled=True),
    ], className="timebar")


def _single_view():
    def tab(value, label, children):
        return dcc.Tab(label=label, value=value, id=f"tab-{value}", children=children,
                       className="tab", selected_className="tab-selected")

    return html.Div(dcc.Tabs(id="tabs", value="overview", className="tabs", children=[
        tab("overview", "Overview", _loading(html.Div(id="overview"))),
        tab("fields", "Fields", [
            dcc.RadioItems(id="fields-mode", value="t", inline=True, className="radios",
                           options=[{"label": "at the time cursor", "value": "t"},
                                    {"label": "final state", "value": "final"}]),
            _loading(_graph("fields-graph"))]),
        tab("histories", "Histories", _loading(_graph("histories-graph"))),
        tab("space-time", "Space-time", [
            dcc.Dropdown(id="space-time-field", value="h", clearable=False,
                         className="dropdown"),
            html.P("Click the map to move the time cursor and the profile position there.",
                   className="hint"),
            _loading(_graph("space-time-graph"))]),
        tab("profiles", "Profiles", [
            html.P("u(z) = u_m + Σ αᵢ φᵢ(z), φᵢ(z) = Pᵢ(1 − 2z). Click either map to "
                   "choose the position.", className="hint"),
            _loading(_graph("profile-lines-graph")),
            html.Div([_graph("profile-map-graph"), _graph("profile-deviation-graph")],
                     className="pair")]),
        tab("hyperbolicity", "Hyperbolicity", _loading([
            html.Div(id="hyperbolicity-text"),
            _graph("hyperbolicity-series-graph"),
            dcc.RadioItems(id="hyperbolicity-map-kind", value="magnitude", inline=True,
                           className="radios",
                           options=[{"label": "|Im λ|", "value": "magnitude"},
                                    {"label": "classification", "value": "classification"}]),
            html.Div([_graph("hyperbolicity-map-graph"), _graph("spectrum-graph")],
                     className="pair")])),
        tab("wet-dry", "Wet-dry", _loading([
            html.Div([_graph("wet-dry-depth-graph"), html.Div(id="wet-dry-counts")],
                     className="pair"),
            _graph("wet-dry-map-graph")])),
        tab("topography", "Topography", _loading([
            html.Div(id="topography-caption"), _graph("topography-graph")])),
    ]), id="single-view")


def _compare_view():
    return html.Div([
        html.Div(id="compare-empty"),
        html.Div([
            html.Label(["reference ", dcc.Dropdown(id="compare-reference", clearable=False,
                                                   className="dropdown")]),
            html.Label(["difference of ", dcc.Dropdown(id="compare-field", value="h",
                                                       clearable=False,
                                                       className="dropdown")]),
            html.Label(["point series of ", dcc.Dropdown(id="compare-point-field", value="h",
                                                         clearable=False,
                                                         className="dropdown")]),
        ], className="compare-controls"),
        _loading([
            _graph("compare-histories-graph"),
            _graph("compare-fields-graph"),
            html.Div(id="compare-difference-notes"),
            _graph("compare-difference-graph"),
            _graph("compare-profiles-graph"),
            html.Div(id="compare-point-notes"),
            _graph("compare-point-graph"),
        ]),
    ], id="compare-view", style={"display": "none"})


def build_layout(store: RunStore):
    first = store.entries()[0].key if store.entries() else None
    return html.Div([
        dcc.Store(id="theme"),
        dcc.Store(id="theme-choice", storage_type="local"),
        dcc.Store(id="selected", data=first),
        dcc.Store(id="compare-set", data=[]),
        dcc.Store(id="time-grid", data=[0.0]),
        dcc.Store(id="current-t"),
        dcc.Store(id="cursor-x"),
        html.Header([
            html.Div([html.Span("moment-sw-view", className="brand"),
                      html.Span(str(store.root), className="muted root")]),
            dcc.RadioItems(id="mode", value="single", inline=True, className="mode",
                           options=[{"label": "Single run", "value": "single"},
                                    {"label": "Compare", "value": "compare"}]),
            html.Button("◐", id="theme-toggle", className="icon-button", n_clicks=0,
                        title="toggle light / dark"),
        ], className="header"),
        html.Div([
            _sidebar(),
            html.Main([_timebar(), _single_view(), _compare_view()], className="main"),
        ], className="body"),
    ], className="app")


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

def _only(*ids) -> bool:
    """True when every input that triggered this call is one of `ids`."""
    triggered = ctx.triggered_prop_ids
    return bool(triggered) and all(prop.split(".")[0] in ids for prop in triggered)


def _with_revision(figure, revision: str):
    figure.layout.uirevision = revision
    return figure


FRAME_PROPS = ("x", "y", "z", "name")


def _frame_patch(figure):
    """Send a new frame as data, not as a new figure.

    For a fixed run and fixed controls a builder's trace list does not depend
    on `t` (tests hold every patched builder to that), so moving in time only
    changes trace data, the titles that quote `t`, and the cursor shapes.
    Axes are never touched: the fixed run-wide ranges and any zoom stay put.
    A placeholder is a different figure altogether and is always sent whole.
    """
    if fig.is_placeholder(figure):
        return figure
    patch = Patch()
    for index, trace in enumerate(figure.data):
        for prop in FRAME_PROPS:
            if prop in trace and trace[prop] is not None:
                value = trace[prop]
                # Arrays travel as plotly's base64 typed-array spec, the same
                # encoding a full figure uses - a heatmap frame as decimal
                # text was ~20x larger.
                if isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
                    value = to_typed_array_spec(value)
                elif isinstance(value, tuple):
                    value = list(value)
                patch["data"][index][prop] = value
    patch["layout"]["title"]["text"] = figure.layout.title.text
    patch["layout"]["annotations"] = [note.to_plotly_json()
                                      for note in figure.layout.annotations]
    patch["layout"]["shapes"] = [shape.to_plotly_json() for shape in figure.layout.shapes]
    return patch


def _cursor_patch(shapes):
    patch = Patch()
    patch["layout"]["shapes"] = shapes
    return patch


def create_app(root=None, *, store: RunStore | None = None, **store_options) -> Dash:
    """The viewer for every run under `root` (or an existing `store`)."""
    store = store or RunStore(root or "results", **store_options)
    app = Dash(__name__, title="moment-sw-view", assets_folder=str(ASSETS),
               suppress_callback_exceptions=True, update_title=None)
    app.layout = lambda: build_layout(store)
    app.store = store

    # -- theme: OS preference on first load, remembered once toggled ----
    app.clientside_callback(
        """
        function(clicks, choice, current) {
            const system = window.matchMedia &&
                window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            let theme = choice || system;
            if (clicks) {
                theme = (current || theme) === 'dark' ? 'light' : 'dark';
                choice = theme;
            }
            document.documentElement.dataset.theme = theme;
            return [theme, choice || null];
        }
        """,
        Output("theme", "data"), Output("theme-choice", "data"),
        Input("theme-toggle", "n_clicks"), State("theme-choice", "data"),
        State("theme", "data"))

    # -- run browser ---------------------------------------------------
    @app.callback(Output("run-list", "children"), Output("compare-count", "children"),
                  Input("run-search", "value"), Input("reload", "n_clicks"),
                  Input("selected", "data"), Input("compare-set", "data"))
    def _run_list(search, _reload, selected, compare_set):
        if ctx.triggered_id == "reload":
            store.refresh()
        count = len(compare_set or [])
        return (view_run_list(store, search, selected, compare_set),
                f"{count} run{'s' if count != 1 else ''} in comparison")

    @app.callback(Output("selected", "data"),
                  Input({"type": "run-select", "key": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def _select(clicks):
        if not ctx.triggered_id or not any(clicks):
            return no_update
        value = ctx.triggered[0]["value"] if ctx.triggered else None
        return ctx.triggered_id["key"] if value else no_update

    @app.callback(Output("compare-set", "data"),
                  Input({"type": "run-compare", "key": ALL}, "n_clicks"),
                  Input({"type": "family-compare", "family": ALL}, "n_clicks"),
                  Input("compare-clear", "n_clicks"),
                  State("compare-set", "data"), prevent_initial_call=True)
    def _compare_set(_runs, _families, _clear, current):
        current = list(current or [])
        trigger = ctx.triggered_id
        if trigger is None or not (ctx.triggered and ctx.triggered[0]["value"]):
            return no_update
        if trigger == "compare-clear":
            return []
        if trigger["type"] == "run-compare":
            key = trigger["key"]
            if key in current:
                current.remove(key)
            elif len(current) < cmp.MAX_RUNS:
                current.append(key)
            return current
        members = [entry.key for entry in store.entries()
                   if entry.family == trigger["family"]]
        return members[:cmp.MAX_RUNS]

    @app.callback(Output("single-view", "style"), Output("compare-view", "style"),
                  Input("mode", "value"))
    def _mode(mode):
        hidden, shown = {"display": "none"}, {}
        return (hidden, shown) if mode == "compare" else (shown, hidden)

    # -- tabs that only apply to some runs -----------------------------
    @app.callback(Output("tab-wet-dry", "disabled"), Output("tab-topography", "disabled"),
                  Output("tabs", "value"), Output("space-time-field", "options"),
                  Input("selected", "data"), State("tabs", "value"))
    def _tabs(key, current):
        run, _ = load(store, key)
        if run is None:
            return True, True, "overview", []
        wet_dry, topography = not run.went_dry, not run.has_topography
        if (current == "wet-dry" and wet_dry) or (current == "topography" and topography):
            current = "overview"
        options = [{"label": label.replace("<sub>", "_").replace("</sub>", ""), "value": key}
                   for key, label in fig.space_time_fields(run)]
        return wet_dry, topography, current, options

    # -- time ----------------------------------------------------------
    @app.callback(Output("time-grid", "data"), Output("time-index", "max"),
                  Output("time-index", "marks"),
                  Input("mode", "value"), Input("selected", "data"),
                  Input("compare-set", "data"))
    def _grid(mode, key, compare_set):
        grid = time_grid(store, mode, key, compare_set)
        last = len(grid) - 1
        marks = {0: fig.fmt(grid[0], ".3g")}
        if last:
            marks[last] = fig.fmt(grid[-1], ".3g")
        return grid, last, marks

    @app.callback(Output("time-index", "value"), Output("current-t", "data"),
                  Output("player", "disabled"), Output("play", "children"),
                  Input("time-index", "value"), Input("time-grid", "data"),
                  Input("player", "n_intervals"), Input("play", "n_clicks"),
                  Input("step-back", "n_clicks"), Input("step-forward", "n_clicks"),
                  *(Input(graph, "clickData") for graph in MAP_GRAPHS),
                  State("current-t", "data"), State("player", "disabled"))
    def _time(index, grid, *rest):
        current, paused = rest[-2], rest[-1]
        trigger = ctx.triggered_id
        point = None
        if trigger in MAP_GRAPHS:
            point = ((ctx.triggered[0]["value"] or {}).get("points") or [{}])[0].get("y")
        position, t, playing = transport(trigger, index, grid, current,
                                         paused is False, point)
        if trigger == "time-index":
            position = no_update                  # the slider already shows it
        changed = playing != (paused is False)
        return (position, t, (not playing) if changed else no_update,
                ("❚❚" if playing else "▶") if changed else no_update)

    @app.callback(Output("player", "interval"), Input("speed", "value"),
                  Input("mode", "value"))
    def _speed(speed, mode):
        return play_interval(speed, mode)

    @app.callback(Output("time-label", "children"), Input("current-t", "data"),
                  Input("time-index", "value"), Input("time-grid", "data"))
    def _time_label(t, index, grid):
        return f"t = {fig.fmt(t, '.5g')}   ({(index or 0) + 1} / {len(grid or [0])})"

    @app.callback(Output("cursor-x", "data"), Output("x-input", "value"),
                  Input("x-input", "value"),
                  Input("space-time-graph", "clickData"),
                  Input("profile-map-graph", "clickData"),
                  Input("profile-deviation-graph", "clickData"),
                  Input("wet-dry-map-graph", "clickData"),
                  Input("hyperbolicity-map-graph", "clickData"),
                  prevent_initial_call=True)
    def _x(typed, *_clicks):
        if ctx.triggered_id == "x-input":
            return typed, no_update
        point = (ctx.triggered[0]["value"] or {}).get("points", [{}])[0]
        if "x" not in point:
            return no_update, no_update
        value = round(float(point["x"]), 6)
        return value, value

    # -- single-run tabs -----------------------------------------------
    @app.callback(Output("overview", "children"),
                  Input("selected", "data"), Input("tabs", "value"),
                  Input("reload", "n_clicks"))
    def _overview(key, tab, _reload):
        return view_overview(store, key) if tab == "overview" else no_update

    @app.callback(Output("fields-graph", "figure"),
                  Input("selected", "data"), Input("current-t", "data"),
                  Input("fields-mode", "value"), Input("tabs", "value"),
                  Input("theme", "data"))
    def _fields(key, t, mode, tab, theme):
        if tab != "fields":
            return no_update
        if mode == "final" and _only("current-t"):
            return no_update
        run, error = load(store, key)
        if run is None:
            return fig.placeholder("Run not loaded", error, themes.get(theme))
        frame = _only("current-t")
        figure = fig.fields_figure(run, None if mode == "final" else t, themes.get(theme),
                                   frame=frame)
        if frame:
            return _frame_patch(figure)
        return _with_revision(figure, f"{key}:fields")

    @app.callback(Output("histories-graph", "figure"),
                  Input("selected", "data"), Input("current-t", "data"),
                  Input("tabs", "value"), Input("theme", "data"))
    def _histories(key, t, tab, theme):
        if tab != "histories":
            return no_update
        run, error = load(store, key)
        if run is None:
            return fig.placeholder("Run not loaded", error, themes.get(theme))
        if _only("current-t"):
            return _cursor_patch(fig.histories_cursor(run, t, themes.get(theme)))
        return _with_revision(fig.histories_figure(run, t, themes.get(theme)),
                              f"{key}:histories")

    @app.callback(Output("space-time-graph", "figure"),
                  Input("selected", "data"), Input("space-time-field", "value"),
                  Input("current-t", "data"), Input("tabs", "value"),
                  Input("theme", "data"))
    def _space_time(key, field, t, tab, theme):
        if tab != "space-time":
            return no_update
        run, error = load(store, key)
        if run is None:
            return fig.placeholder("Run not loaded", error, themes.get(theme))
        if _only("current-t"):
            return _cursor_patch(fig.space_time_cursor(run, t, themes.get(theme)))
        return _with_revision(fig.space_time_figure(run, field or "h", t, themes.get(theme)),
                              f"{key}:space-time:{field}")

    @app.callback(Output("profile-lines-graph", "figure"),
                  Output("profile-map-graph", "figure"),
                  Output("profile-deviation-graph", "figure"),
                  Input("selected", "data"), Input("current-t", "data"),
                  Input("cursor-x", "data"), Input("tabs", "value"),
                  Input("theme", "data"))
    def _profiles(key, t, x, tab, theme):
        if tab != "profiles":
            return no_update, no_update, no_update
        palette = themes.get(theme)
        run, error = load(store, key)
        if run is None:
            empty = fig.placeholder("Run not loaded", error, palette)
            return empty, empty, empty
        # A new x picks a new cell and with it new axis ranges: rebuild.
        frame = _only("current-t")
        figures = (fig.profile_lines_figure(run, t, x, palette, frame=frame),
                   fig.profile_map_figure(run, t, x, "velocity", palette, frame=frame),
                   fig.profile_map_figure(run, t, x, "deviation", palette, frame=frame))
        if frame:
            return tuple(_frame_patch(figure) for figure in figures)
        return tuple(_with_revision(figure, f"{key}:{name}")
                     for figure, name in zip(figures, ("pl", "pm", "pd")))

    @app.callback(Output("hyperbolicity-text", "children"),
                  Output("hyperbolicity-series-graph", "figure"),
                  Output("hyperbolicity-map-graph", "figure"),
                  Output("spectrum-graph", "figure"),
                  Input("selected", "data"), Input("current-t", "data"),
                  Input("cursor-x", "data"), Input("hyperbolicity-map-kind", "value"),
                  Input("tabs", "value"), Input("theme", "data"))
    def _hyperbolicity(key, t, x, kind, tab, theme):
        if tab != "hyperbolicity":
            return no_update, no_update, no_update, no_update
        palette = themes.get(theme)
        run, error = load(store, key)
        if run is None:
            empty = fig.placeholder("Run not loaded", error, palette)
            return _error_panel(error), empty, empty, empty
        view = store.hyperbolicity(key)
        spectra = view.run.hyperbolicity
        cells = spectra.cells if spectra is not None else None
        tolerance = spectra.tolerance if spectra is not None else 1e-10
        spectrum = _with_revision(fig.spectrum_figure(cells, tolerance, t, x, palette),
                                  f"{key}:spectrum")
        if _only("current-t", "cursor-x"):
            return (no_update,
                    _cursor_patch(fig.hyperbolicity_series_cursor(view.derived, t, palette)),
                    _cursor_patch(fig.hyperbolicity_map_cursor(t, palette)
                                  if cells is not None and len(cells) else []),
                    spectrum)
        return (view_hyperbolicity_text(store, key),
                _with_revision(fig.hyperbolicity_series_figure(view.derived, tolerance, t,
                                                               palette), f"{key}:hs"),
                _with_revision(fig.hyperbolicity_map_figure(cells, tolerance, kind, t,
                                                            palette), f"{key}:hm:{kind}"),
                spectrum)

    @app.callback(Output("wet-dry-depth-graph", "figure"),
                  Output("wet-dry-map-graph", "figure"),
                  Output("wet-dry-counts", "children"),
                  Input("selected", "data"), Input("current-t", "data"),
                  Input("tabs", "value"), Input("theme", "data"))
    def _wet_dry(key, t, tab, theme):
        if tab != "wet-dry":
            return no_update, no_update, no_update
        palette = themes.get(theme)
        run, error = load(store, key)
        if run is None:
            empty = fig.placeholder("Run not loaded", error, palette)
            return empty, empty, ""
        frame = _only("current-t")
        depth = fig.wet_dry_depth_figure(run, t, palette, frame=frame)
        counts = view_wet_dry_counts(store, key, t)
        if frame:
            return (_frame_patch(depth),
                    _cursor_patch(fig.space_time_cursor(run, t, palette)), counts)
        depth = _with_revision(depth, f"{key}:wd")
        return (depth, _with_revision(fig.wet_dry_map_figure(run, t, palette), f"{key}:wm"),
                counts)

    @app.callback(Output("topography-graph", "figure"),
                  Output("topography-caption", "children"),
                  Input("selected", "data"), Input("current-t", "data"),
                  Input("tabs", "value"), Input("theme", "data"))
    def _topography(key, t, tab, theme):
        if tab != "topography":
            return no_update, no_update
        palette = themes.get(theme)
        run, error = load(store, key)
        if run is None:
            return fig.placeholder("Run not loaded", error, palette), ""
        frame = _only("current-t")
        figure = fig.topography_figure(run, t, palette, frame=frame)
        if frame:
            return _frame_patch(figure), view_topography_caption(store, key, t)
        return (_with_revision(figure, f"{key}:topo"),
                view_topography_caption(store, key, t))

    # -- compare -------------------------------------------------------
    @app.callback(Output("compare-reference", "options"),
                  Output("compare-reference", "value"),
                  Output("compare-field", "options"),
                  Output("compare-point-field", "options"),
                  Output("compare-empty", "children"),
                  Input("compare-set", "data"), State("compare-reference", "value"))
    def _compare_controls(compare_set, reference):
        items = compared(store, compare_set)
        options = [{"label": item.label, "value": index} for index, item in enumerate(items)]
        if reference is None or reference >= len(items):
            reference = 0
        top = max((item.order for item in items), default=0)
        fields = [{"label": "h", "value": "h"}, {"label": "u_m", "value": "u_m"}] + [
            {"label": f"α{i}", "value": f"a{i}"} for i in range(1, top + 1)]
        point = fields[:2] + [{"label": "q = h·u_m", "value": "q"}] + fields[2:]
        empty = ([] if len(items) >= 2 else _note(
            "Add two or more runs with the + buttons in the run list (or “compare all” on a "
            f"family). Up to {cmp.MAX_RUNS} runs can be compared at once.", "info"))
        return options, reference, fields, point, empty

    @app.callback(Output("compare-histories-graph", "figure"),
                  Output("compare-fields-graph", "figure"),
                  Output("compare-difference-graph", "figure"),
                  Output("compare-difference-notes", "children"),
                  Output("compare-profiles-graph", "figure"),
                  Output("compare-point-graph", "figure"),
                  Output("compare-point-notes", "children"),
                  Input("mode", "value"), Input("compare-set", "data"),
                  Input("current-t", "data"), Input("cursor-x", "data"),
                  Input("compare-reference", "value"), Input("compare-field", "value"),
                  Input("compare-point-field", "value"), Input("theme", "data"))
    def _compare(mode, compare_set, t, x, reference, field, point_field, theme):
        if mode != "compare":
            return (no_update,) * 7
        palette = themes.get(theme)
        items = compared(store, compare_set)
        if not items:
            empty = fig.placeholder("No runs selected",
                                    "Add runs with the + buttons in the run list.", palette)
            return empty, empty, empty, "", empty, empty, ""
        revision = "cmp:" + ",".join(compare_set or [])
        histories = (_cursor_patch(cmp.histories_cursor(items, t, palette))
                     if _only("current-t", "cursor-x")
                     else _with_revision(cmp.histories_figure(items, t, palette),
                                         revision + ":h"))
        frame = _only("current-t")
        difference, difference_notes = cmp.difference_figure(
            items, reference or 0, field or "h", t, palette, frame=frame)
        point, point_notes = cmp.point_series_figure(items, x, point_field or "h", t, palette,
                                                     frame=frame)
        fields = cmp.fields_figure(items, t, palette, frame=frame)
        profiles = cmp.profiles_figure(items, x, t, palette, frame=frame)
        notes = ([_note(text) for text in difference_notes],
                 [_note(text) for text in point_notes])
        if frame:
            return (histories, _frame_patch(fields), _frame_patch(difference), notes[0],
                    _frame_patch(profiles), _frame_patch(point), notes[1])
        return (histories,
                _with_revision(fields, revision + ":f"),
                _with_revision(difference, revision + f":d:{field}"),
                notes[0],
                _with_revision(profiles, revision + ":p"),
                _with_revision(point, revision + f":x:{point_field}"),
                notes[1])

    return app


__all__ = ["create_app", "build_layout", "time_grid", "compared", "load",
           "transport", "play_interval",
           "view_run_list", "view_overview", "view_hyperbolicity_text",
           "view_topography_caption", "view_wet_dry_counts", "lines_to_html"]
