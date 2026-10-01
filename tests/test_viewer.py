"""The interactive viewer (`swme.viewer`, `moment-sw-view`).

The tests that matter most pin the viewer to the same rules as the PDF report,
because a view that disagrees with the report about what a run contains is
worse than no view:

* **Missing data is a placeholder, never an empty axis** - N = 0, no field
  history, no spectra, flat bed.
* **Comparisons never invent values** - a moment a run does not have is left
  out rather than drawn as zero, and different grids are interpolated with a
  note rather than silently.
* **A time-only update touches only the cursor.** The heavy figures are sent
  once and then patched; that only works if a paired figure's shapes are
  exactly its cursor, which is asserted here for every pair.
"""

from __future__ import annotations

import json
import os
import shutil
import textwrap

import numpy as np
import pytest

import plotly.graph_objects as go
from dash import dcc

from swme import cli
from swme.report import data as report_data
from swme.report.hyperbolicity import recompute_summary
from swme.report.style import Prose
from swme.viewer import app as viewer_app
from swme.viewer import catalog, compare, figures, store as viewer_store, theme
from swme.viewer import cli as viewer_cli

DRY = textwrap.dedent("""
    pde:
      type: SWME1D
      initial_condition: damBreak_dryBed
      viscosity: 0.0
      slip_length: 1.0
      linear_source: false
      hyperbolic: false
    grid:
      x1: -1.0
      x2: 1.0
      resolution_x: 40
    numerics:
      order: {order}
      t_end: 0.3
      method: classical
      fvm_type: PVM
      pvm: Roe
      time_integrator: ExplicitEuler
      boundary_condition: INFLOW_OUTFLOW
    wet_dry:
      h_dry: 1.0e-4
      h_wet: 1.0e-3
      eps_div: 1.0e-14
    postprocessing:
      store_history: {history}
      store_hyperbolicity: {spectra}
      history_stride: 1
      hyperbolicity_stride: 1
""")

LAKE = textwrap.dedent("""
    pde:
      type: SWME1D
      initial_condition: lakeAtRest
      viscosity: 0.0
      slip_length: 1.0
      linear_source: false
      hyperbolic: false
    grid:
      x1: 0.0
      x2: 1.0
      resolution_x: 40
    numerics:
      order: 1
      t_end: 0.1
      method: classical
      fvm_type: PVM
      pvm: Roe
      time_integrator: ExplicitEuler
      boundary_condition: INFLOW_OUTFLOW
    topography:
      bed_profile: gaussian_bump
      amplitude: 0.4
      center: 0.5
      width: 0.1
      reference_water_level: 2.0
    postprocessing:
      store_history: true
      history_stride: 1
""")


def _run(directory, text):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "case.yaml").write_text(text)
    cli.run(directory / "case.yaml", output_dir=str(directory))
    return directory


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """A results tree laid out like `run_thesis_configs.sh` writes one.

        tree/Family/Dry_N2     dam break onto a dry bed, N = 2, spectra
        tree/Family/Lake_N1    lake at rest over a bump, N = 1
        tree/Family/Figures    no run output at all
        tree/Plain_N0          N = 0, no field history, no spectra
    """
    root = tmp_path_factory.mktemp("results")
    _run(root / "Family" / "Dry_N2", DRY.format(order=2, history="true", spectra="true"))
    _run(root / "Family" / "Lake_N1", LAKE)
    (root / "Family" / "Figures").mkdir()
    (root / "Family" / "Figures" / "plot.svg").write_text("<svg/>")
    _run(root / "Plain_N0", DRY.format(order=0, history="false", spectra="false"))
    return root


@pytest.fixture(scope="module")
def store(tree):
    return viewer_store.RunStore(tree)


def _key(store, name):
    return next(entry.key for entry in store.entries() if entry.name == name)


@pytest.fixture(scope="module")
def dry(store):
    return store.run(_key(store, "Dry_N2"))


@pytest.fixture(scope="module")
def lake(store):
    return store.run(_key(store, "Lake_N1"))


@pytest.fixture(scope="module")
def plain(store):
    return store.run(_key(store, "Plain_N0"))


# ---------------------------------------------------------------------------
# the data-layer switch the viewer relies on
# ---------------------------------------------------------------------------

def test_from_directory_can_defer_hyperbolicity(tree):
    directory = tree / "Family" / "Dry_N2"
    assert report_data.from_directory(directory).hyperbolicity.has_cells
    assert report_data.from_directory(directory, hyperbolicity=False).hyperbolicity is None


# ---------------------------------------------------------------------------
# catalog
# ---------------------------------------------------------------------------

def test_discover_walks_families_and_skips_figure_folders(tree):
    entries = {entry.name: entry for entry in catalog.discover(tree)}
    assert set(entries) == {"Dry_N2", "Lake_N1", "Plain_N0"}
    assert entries["Dry_N2"].family == "Family"
    assert entries["Plain_N0"].family == ""
    assert (entries["Dry_N2"].order, entries["Lake_N1"].order, entries["Plain_N0"].order) == (2, 1, 0)
    assert entries["Dry_N2"].has_history and entries["Dry_N2"].has_hyperbolicity
    assert not entries["Plain_N0"].has_history
    assert entries["Dry_N2"].scheme == "Roe" and entries["Dry_N2"].closure == "SWME"


def test_a_run_directory_is_its_own_root(tree):
    entries = catalog.discover(tree / "Family" / "Lake_N1")
    assert [entry.name for entry in entries] == ["Lake_N1"]


def test_a_directory_with_several_runs_lists_each(tmp_path, tree):
    source = tree / "Family" / "Dry_N2"
    for name in ("first", "second"):
        for path in source.glob("swme_N2_*"):
            shutil.copy(path, tmp_path / path.name.replace("swme_N2", name))
    entries = catalog.discover(tmp_path)
    assert sorted(entry.prefix for entry in entries) == ["first", "second"]
    assert len({entry.key for entry in entries}) == 2


def test_missing_or_broken_sidecar_is_tolerated(tmp_path, tree):
    shutil.copy(tree / "Family" / "Dry_N2" / "swme_N2_final.csv", tmp_path)
    (entry,) = catalog.discover(tmp_path)
    assert entry.order == 2 and not entry.has_sidecar and entry.scheme is None

    (tmp_path / "swme_N2_run.json").write_text("{not json")
    (entry,) = catalog.discover(tmp_path)
    assert entry.order == 2 and not entry.has_sidecar


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

def test_store_caches_and_defers_spectra(store, dry):
    key = _key(store, "Dry_N2")
    assert store.run(key) is dry
    assert dry.hyperbolicity is None, "spectra must not load with the run"
    view = store.hyperbolicity(key)
    assert view.corrected and view.run.hyperbolicity.has_cells
    expected = recompute_summary(view.run.hyperbolicity.cells, view.run.hyperbolicity.tolerance)
    assert view.derived.equals(expected)
    assert store.hyperbolicity(key) is view


def test_store_reloads_a_run_whose_files_changed(tree):
    local = viewer_store.RunStore(tree)
    key = _key(local, "Lake_N1")
    first = local.run(key)
    final = tree / "Family" / "Lake_N1" / "swme_N1_final.csv"
    stamp = final.stat().st_mtime_ns + 5_000_000_000
    os.utime(final, ns=(stamp, stamp))
    assert local.run(key) is not first


def test_store_evicts_beyond_capacity(tree):
    local = viewer_store.RunStore(tree, capacity=1)
    for entry in local.entries():
        local.run(entry.key)
    assert len(local._runs) == 1


def test_unknown_key_is_a_clear_error(store):
    with pytest.raises(KeyError, match="reload"):
        store.run("nope::nope")


# ---------------------------------------------------------------------------
# single-run figures
# ---------------------------------------------------------------------------

def _builders(run, t):
    return [
        figures.fields_figure(run, t), figures.fields_figure(run),
        figures.histories_figure(run, t),
        *(figures.space_time_figure(run, field, t)
          for field, _ in figures.space_time_fields(run)),
        figures.profile_lines_figure(run, t, None),
        figures.profile_map_figure(run, t, None, "velocity"),
        figures.profile_map_figure(run, t, None, "deviation"),
        figures.wet_dry_depth_figure(run, t), figures.wet_dry_map_figure(run, t),
        figures.topography_figure(run, t),
    ]


@pytest.mark.parametrize("name", ["Dry_N2", "Lake_N1", "Plain_N0"])
@pytest.mark.parametrize("palette", [theme.LIGHT, theme.DARK])
def test_every_builder_returns_a_figure(store, name, palette):
    run = store.run(_key(store, name))
    t = None if run.snapshots is None else float(run.snapshots.times[1])
    for figure in _builders(run, t):
        assert isinstance(figure, go.Figure)
        assert figures.is_placeholder(figure) or len(figure.data) > 0


@pytest.mark.parametrize("name", ["Dry_N2", "Lake_N1", "Plain_N0"])
def test_one_panel_per_moment(store, name):
    run = store.run(_key(store, name))
    figure = figures.fields_figure(run)
    fields = [trace for trace in figure.data if trace.name != "bed Z"]
    assert len(fields) == 2 + run.order
    assert len(figure.layout.annotations) == 2 + run.order


def test_missing_data_draws_placeholders_not_empty_axes(plain):
    assert figures.is_placeholder(figures.histories_figure(plain))
    assert figures.is_placeholder(figures.space_time_figure(plain, "h"))
    assert figures.is_placeholder(figures.wet_dry_map_figure(plain))
    assert figures.is_placeholder(figures.topography_figure(plain))
    assert figures.is_placeholder(figures.profile_map_figure(plain, kind="deviation"))
    lines = figures.profile_lines_figure(plain)
    assert any("No field history" in note.text for note in lines.layout.annotations)
    assert figures.hyperbolicity_series_cursor(None, 0.1) == []


def test_snapshot_is_the_nearest_stored_state(dry):
    times = dry.snapshots.times
    target = 0.5 * (times[1] + times[2]) + 1e-9
    values, actual, final = figures.snapshot(dry, target)
    assert not final and actual == times[2]
    np.testing.assert_array_equal(values, dry.snapshots.values[2])
    _, actual, final = figures.snapshot(dry, None)
    assert final


def test_lake_at_rest_stays_at_rest_through_the_viewer(lake):
    """The C-property, as the topography view reports it: round-off at every time."""
    for t in lake.snapshots.times:
        assert figures.topography_residual(lake, t) < 1e-12
    surface = figures.space_time_values(lake, "surface")
    np.testing.assert_allclose(surface, 2.0, atol=1e-12)


@pytest.mark.parametrize("pair", ["histories", "space_time", "wet_dry_map"])
def test_paired_figures_hold_only_their_cursor(dry, pair):
    t = float(dry.snapshots.times[3])
    build = {"histories": lambda: figures.histories_figure(dry, t),
             "space_time": lambda: figures.space_time_figure(dry, "h", t),
             "wet_dry_map": lambda: figures.wet_dry_map_figure(dry, t)}[pair]
    cursor = {"histories": figures.histories_cursor,
              "space_time": figures.space_time_cursor,
              "wet_dry_map": figures.space_time_cursor}[pair]
    shapes = [shape.to_plotly_json() for shape in build().layout.shapes]
    assert shapes == cursor(dry, t)
    assert shapes and all(t in (shape["x0"], shape["y0"]) for shape in shapes)


def test_history_cursor_spans_every_panel(dry):
    panels = 2 + 1 + dry.order          # h, u_m, discharge, moments
    assert len(figures.histories_cursor(dry, 0.01)) == panels


def test_hyperbolicity_views(store):
    view = store.hyperbolicity(_key(store, "Dry_N2"))
    cells, tolerance = view.run.hyperbolicity.cells, view.run.hyperbolicity.tolerance
    t = float(view.derived["time"].iloc[-1])

    series = figures.hyperbolicity_series_figure(view.derived, tolerance, t)
    assert [shape.to_plotly_json() for shape in series.layout.shapes] == \
        figures.hyperbolicity_series_cursor(view.derived, t)

    grid = figures.hyperbolicity_map_figure(cells, tolerance, "classification", t).data[0].z
    assert np.isnan(np.asarray(grid, dtype=float)).any(), "dry cells must stay blank"

    row, eigenvalues = figures.spectrum_at(cells, t, None)
    rows = cells[cells["time"] == row["time"]]
    assert row["max_abs_imag_eig"] == np.nanmax(rows["max_abs_imag_eig"])
    assert eigenvalues.size == dry_size(view.run.order)
    row, _ = figures.spectrum_at(cells, t, 0.95)
    assert abs(row["x"] - 0.95) <= np.diff(np.unique(cells["x"])).max()


def dry_size(order):
    return 2 + order


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------

def test_moments_a_run_lacks_are_left_out(dry, plain):
    items = [compare.Compared(dry, "N2"), compare.Compared(plain, "N0")]
    figure = compare.fields_figure(items, None)
    assert len([trace for trace in figure.data if trace.legendgroup == "N2"]) == 4
    assert len([trace for trace in figure.data if trace.legendgroup == "N0"]) == 2

    _, notes = compare.difference_figure(items, 0, "a2", None)
    assert any("N0 has no a2" in note for note in notes)


def test_a_run_minus_itself_is_zero(dry):
    items = [compare.Compared(dry, "a"), compare.Compared(dry, "b")]
    figure, notes = compare.difference_figure(items, 0, "h", float(dry.snapshots.times[2]))
    assert notes == []
    assert np.max(np.abs(np.asarray(figure.data[0].y))) == 0.0


def test_different_grids_are_interpolated_and_say_so(dry, lake):
    items = [compare.Compared(lake, "lake"), compare.Compared(dry, "dry")]
    _, notes = compare.difference_figure(items, 0, "h", None)
    assert any("interpolated" in note for note in notes)
    _, notes = compare.difference_figure(items[::-1], 0, "h", None)
    assert any("does not span" in note for note in notes)


def test_common_time_range(dry, lake, plain):
    items = [compare.Compared(dry, "a"), compare.Compared(lake, "b"),
             compare.Compared(plain, "c")]
    start, end = compare.common_time_range(items)
    assert start == 0.0
    assert end == min(dry.snapshots.times[-1], lake.snapshots.times[-1])
    assert compare.common_time_range([compare.Compared(plain, "c")]) is None


def test_run_styles_follow_the_order_not_the_position():
    styles = theme.run_styles([1, 2, 1], theme.LIGHT)
    assert styles[0].color == styles[2].color != styles[1].color
    assert styles[0].dash != styles[2].dash
    # Dropping the first run does not repaint the second.
    assert theme.run_styles([2, 1], theme.LIGHT)[0].color == styles[1].color


def test_point_series_reports_its_sampling(dry, plain):
    items = [compare.Compared(dry, "dry"), compare.Compared(plain, "plain")]
    figure, notes = compare.point_series_figure(items, None, "q")
    assert len(figure.data) == 1
    assert any("stored no field history" in note for note in notes)
    assert any("snapshots of" in note for note in notes)


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

def test_app_builds(store):
    app = viewer_app.create_app(store=store)
    layout = app.layout()
    assert layout is not None and len(app.callback_map) > 15


def test_run_list_filters(store):
    shown = json.dumps(viewer_app.view_run_list(store, "lake", None, []),
                       default=lambda c: c.to_plotly_json())
    assert "Lake_N1" in shown and "Dry_N2" not in shown


def test_time_grid(store, dry):
    key = _key(store, "Dry_N2")
    assert viewer_app.time_grid(store, "single", key, []) == list(dry.snapshots.times)
    assert viewer_app.time_grid(store, "single", _key(store, "Plain_N0"), []) == [0.3]
    both = viewer_app.time_grid(store, "compare", None, [key, _key(store, "Lake_N1")])
    assert both == sorted(both) and len(both) >= len(dry.snapshots.times)


def test_overview_and_errors(store):
    text = json.dumps(viewer_app.view_overview(store, _key(store, "Dry_N2")).to_plotly_json(),
                      default=lambda c: c.to_plotly_json())
    assert "Dry_N2" in text and "Roe" in text
    run, error = viewer_app.load(store, "nope::nope")
    assert run is None and "reload" in error


def test_report_text_becomes_html():
    parts = viewer_app.lines_to_html(["SCHEME-LEVEL COUNTERS", "",
                                      "  scheme                Roe",
                                      Prose("A paragraph.")])
    kinds = [type(part).__name__ for part in parts]
    assert kinds == ["H4", "Table", "P"]


def _post(app, output, inputs, changed):
    spec = app.callback_map[output]
    def dep(d):
        return {"id": d["id"], "property": d["property"],
                "value": inputs.get(f"{d['id']}.{d['property']}")}
    targets = [{"id": o.rsplit(".", 1)[0], "property": o.rsplit(".", 1)[1]}
               for o in output.strip(".").split("...")]
    payload = {"output": output,
               "outputs": targets if output.startswith("..") else targets[0],
               "inputs": [dep(d) for d in spec["inputs"]],
               "state": [dep(d) for d in spec["state"]],
               "changedPropIds": changed}
    response = app.server.test_client().post("/_dash-update-component", json=payload)
    assert response.status_code == 200, response.data[:500]
    return response.get_json()["response"]


def test_a_time_only_change_is_sent_as_a_cursor_patch(store, dry):
    app = viewer_app.create_app(store=store)
    inputs = {"selected.data": _key(store, "Dry_N2"), "tabs.value": "space-time",
              "space-time-field.value": "h", "theme.data": "light",
              "current-t.data": float(dry.snapshots.times[2])}
    full = _post(app, "space-time-graph.figure", inputs, ["tabs.value"])
    assert full["space-time-graph"]["figure"]["data"][0]["type"] == "heatmap"

    inputs["current-t.data"] = float(dry.snapshots.times[4])
    patch = _post(app, "space-time-graph.figure", inputs, ["current-t.data"])
    operations = patch["space-time-graph"]["figure"]["operations"]
    assert [op["location"] for op in operations] == [["layout", "shapes"]]
    assert operations[0]["params"]["value"][0]["y0"] == float(dry.snapshots.times[4])


def test_inactive_tabs_do_no_work(store):
    app = viewer_app.create_app(store=store)
    response = app.server.test_client().post("/_dash-update-component", json={
        "output": "fields-graph.figure",
        "outputs": {"id": "fields-graph", "property": "figure"},
        "inputs": [{"id": "selected", "property": "data", "value": _key(store, "Dry_N2")},
                   {"id": "current-t", "property": "data", "value": 0.01},
                   {"id": "fields-mode", "property": "value", "value": "t"},
                   {"id": "tabs", "property": "value", "value": "overview"},
                   {"id": "theme", "property": "data", "value": "light"}],
        "state": [], "changedPropIds": ["current-t.data"]})
    # `no_update`: answered, but with nothing for the figure.
    assert response.status_code in (200, 204)
    if response.status_code == 200:
        assert "fields-graph" not in response.get_json()["response"]


# ---------------------------------------------------------------------------
# playback: frames are patches of fixed figures
# ---------------------------------------------------------------------------

def _structure(figure):
    return [(trace.type, trace.legendgroup, trace.xaxis if "xaxis" in trace else None)
            for trace in figure.data]


def _single_builders(run):
    return {
        "fields": lambda t, frame: figures.fields_figure(run, t, frame=frame),
        "profile_lines": lambda t, frame: figures.profile_lines_figure(run, t, frame=frame),
        "profile_map": lambda t, frame: figures.profile_map_figure(run, t, frame=frame),
        "profile_deviation": lambda t, frame: figures.profile_map_figure(
            run, t, kind="deviation", frame=frame),
        "wet_dry_depth": lambda t, frame: figures.wet_dry_depth_figure(run, t, frame=frame),
        "topography": lambda t, frame: figures.topography_figure(run, t, frame=frame),
    }


@pytest.mark.parametrize("name", ["Dry_N2", "Lake_N1", "Plain_N0"])
def test_a_frame_has_the_trace_list_of_the_full_figure(store, name):
    """A patch writes trace i's data into trace i: the lists must line up.

    The full figure at one time and a frame at another must have the same
    traces in the same order, or a frame would draw the wrong data.
    """
    run = store.run(_key(store, name))
    times = [None] if run.snapshots is None else [
        float(run.snapshots.times[0]), float(run.snapshots.times[-1])]
    for view, build in _single_builders(run).items():
        full = build(times[0], False)
        frame = build(times[-1], True)
        assert figures.is_placeholder(full) == figures.is_placeholder(frame), view
        assert _structure(full) == _structure(frame), view


def test_a_compare_frame_has_the_trace_list_of_the_full_figure(dry, lake, plain):
    items = [compare.Compared(dry, "dry"), compare.Compared(lake, "lake"),
             compare.Compared(plain, "plain")]
    early, late = float(dry.snapshots.times[1]), float(dry.snapshots.times[-1])
    pairs = [
        (compare.fields_figure(items, early), compare.fields_figure(items, late, frame=True)),
        (compare.difference_figure(items, 0, "h", early)[0],
         compare.difference_figure(items, 0, "h", late, frame=True)[0]),
        (compare.profiles_figure(items, 0.5, early),
         compare.profiles_figure(items, 0.5, late, frame=True)),
        (compare.point_series_figure(items, None, "h", early)[0],
         compare.point_series_figure(items, None, "h", late, frame=True)[0]),
    ]
    for full, frame in pairs:
        assert _structure(full) == _structure(frame)


def test_axes_are_fixed_over_the_whole_run(dry):
    early, late = float(dry.snapshots.times[0]), float(dry.snapshots.times[-1])
    ranges = [[axis.range for axis in (figures.fields_figure(dry, t).layout[name]
                                       for name in ("yaxis", "yaxis2", "yaxis3", "yaxis4"))]
              for t in (early, late)]
    assert ranges[0] == ranges[1]
    assert all(value is not None for value in ranges[0])
    for (column, _), (low, high) in zip(figures.field_columns(dry), ranges[0]):
        values = figures.field_values(dry, column)
        assert low <= values.min() and values.max() <= high, column

    maps = [figures.profile_map_figure(dry, t).data[0] for t in (early, late)]
    assert (maps[0].zmin, maps[0].zmax) == (maps[1].zmin, maps[1].zmax)
    assert maps[0].zmin is not None


def test_padded_range_never_collapses():
    assert figures.padded_range([0.0, 0.0]) == [-1.0, 1.0]
    low, high = figures.padded_range([2.0, 2.0])
    assert low < 2.0 < high
    assert figures.padded_range([np.nan]) is None


def test_the_default_profile_cell_does_not_move_with_time(dry):
    cell = figures.profile_cell(dry, None)
    titles = {figures.profile_lines_figure(dry, float(t)).layout.annotations[1].text
              for t in dry.snapshots.times}
    assert len(titles) == 1, "the cell, and so the title, must not change with t"
    assert cell == figures.profile_cell(dry, None)


ALLOWED_FRAME_PATHS = ("data", ("layout", "title"), ("layout", "annotations"),
                       ("layout", "shapes"))


def _frame_locations(response, component):
    operations = response[component]["figure"]["operations"]
    return [tuple(op["location"]) for op in operations]


def _only_frame_paths(locations):
    return all(location[0] == "data" or location[:2] in ALLOWED_FRAME_PATHS
               for location in locations)


@pytest.mark.parametrize("tab, output, component", [
    ("fields", "fields-graph.figure", "fields-graph"),
    ("topography", "..topography-graph.figure...topography-caption.children..",
     "topography-graph"),
])
def test_a_frame_is_a_data_patch_that_never_touches_the_axes(store, lake, tab, output,
                                                             component):
    app = viewer_app.create_app(store=store)
    inputs = {"selected.data": _key(store, "Lake_N1"), "tabs.value": tab,
              "fields-mode.value": "t", "theme.data": "light",
              "current-t.data": float(lake.snapshots.times[1])}
    full = _post(app, output, inputs, ["tabs.value"])
    assert "data" in full[component]["figure"], "a tab switch sends the whole figure"

    inputs["current-t.data"] = float(lake.snapshots.times[-1])
    locations = _frame_locations(_post(app, output, inputs, ["current-t.data"]), component)
    assert locations and _only_frame_paths(locations)
    assert not any("axis" in str(part) for location in locations for part in location)

    inputs["selected.data"] = _key(store, "Dry_N2")
    rebuilt = _post(app, output, inputs, ["selected.data"])
    assert "data" in rebuilt[component]["figure"], "a run change sends the whole figure"


def test_profile_frames_patch_all_three_figures(store, dry):
    app = viewer_app.create_app(store=store)
    output = [key for key in app.callback_map if key.startswith("..profile-lines-graph")][0]
    inputs = {"selected.data": _key(store, "Dry_N2"), "tabs.value": "profiles",
              "theme.data": "light", "current-t.data": float(dry.snapshots.times[2])}
    _post(app, output, inputs, ["tabs.value"])
    inputs["current-t.data"] = float(dry.snapshots.times[5])
    response = _post(app, output, inputs, ["current-t.data"])
    for component in ("profile-lines-graph", "profile-map-graph", "profile-deviation-graph"):
        assert _only_frame_paths(_frame_locations(response, component))
    # Heatmap frames travel as typed arrays, not decimal text.
    map_ops = response["profile-map-graph"]["figure"]["operations"]
    z = next(op["params"]["value"] for op in map_ops if op["location"][-1] == "z")
    assert set(z) >= {"dtype", "bdata"}

    # A new x is a new cell with new axis ranges: rebuilt, not patched.
    inputs["cursor-x.data"] = 0.1
    rebuilt = _post(app, output, inputs, ["cursor-x.data"])
    assert "data" in rebuilt["profile-lines-graph"]["figure"]


def test_compare_frames_are_patches(store, dry):
    app = viewer_app.create_app(store=store)
    output = [key for key in app.callback_map if key.startswith("..compare-histories")][0]
    keys = [_key(store, "Dry_N2"), _key(store, "Lake_N1")]
    inputs = {"mode.value": "compare", "compare-set.data": keys, "theme.data": "light",
              "compare-reference.value": 0, "compare-field.value": "h",
              "compare-point-field.value": "h", "current-t.data": float(dry.snapshots.times[1])}
    _post(app, output, inputs, ["compare-set.data"])
    inputs["current-t.data"] = float(dry.snapshots.times[3])
    response = _post(app, output, inputs, ["current-t.data"])
    for component in ("compare-fields-graph", "compare-difference-graph",
                      "compare-profiles-graph", "compare-point-graph"):
        assert _only_frame_paths(_frame_locations(response, component)), component


def test_transport_steps_clamp_and_stop_at_the_end():
    grid = [0.0, 0.1, 0.2, 0.3]
    transport = viewer_app.transport
    assert transport("step-forward", 1, grid, 0.1, False) == (2, 0.2, False)
    assert transport("step-forward", 3, grid, 0.3, False) == (3, 0.3, False)
    assert transport("step-back", 0, grid, 0.0, False) == (0, 0.0, False)
    # Stepping pauses playback.
    assert transport("step-back", 2, grid, 0.2, True) == (1, 0.1, False)
    # Ticks advance and stop on the last step instead of wrapping.
    assert transport("player", 1, grid, 0.1, True) == (2, 0.2, True)
    assert transport("player", 2, grid, 0.2, True) == (3, 0.3, False)
    assert transport("player", 3, grid, 0.3, False) == (3, 0.3, False)
    # Play at the end starts over; play while playing pauses.
    assert transport("play", 3, grid, 0.3, False) == (0, 0.0, True)
    assert transport("play", 1, grid, 0.1, True) == (1, 0.1, False)
    # A map click jumps to the nearest stored time; a new grid keeps the time.
    assert transport("space-time-graph", 0, grid, 0.0, False, 0.21) == (2, 0.2, False)
    assert transport(None, 0, [0.0, 0.25, 0.5], 0.3, False) == (1, 0.25, False)
    assert transport(None, 0, grid, None, False) == (3, 0.3, False)


def test_play_interval():
    assert viewer_app.play_interval("2×", "single") == 150
    assert viewer_app.play_interval("2×", "compare") == viewer_app.COMPARE_MIN_INTERVAL_MS
    assert viewer_app.play_interval("0.25×", "compare") == 1200
    assert viewer_app.play_interval(None, "single") == viewer_app.PLAY_INTERVAL_MS


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if hasattr(child, "to_plotly_json"):
            yield from _walk(child)


def test_no_loading_overlay_hides_the_plots(store):
    """The regression this change fixes: `dcc.Loading` blanking a plot per frame."""
    loaders = [node for node in _walk(viewer_app.build_layout(store))
               if isinstance(node, dcc.Loading)]
    assert loaders
    for loader in loaders:
        assert loader.overlay_style.get("visibility") == "visible"
        assert loader.delay_show >= 200


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

def test_cli_rejects_a_missing_directory(tmp_path, capsys):
    assert viewer_cli.main([str(tmp_path / "nope"), "--no-browser"]) == 1
    assert "No such directory" in capsys.readouterr().err


def test_cli_rejects_a_directory_without_runs(tmp_path, capsys):
    assert viewer_cli.main([str(tmp_path), "--no-browser"]) == 1
    assert "no run output" in capsys.readouterr().err

