"""The post-processing suite (RESTRUCTURE_PLAN.md Step 8.5, Phase 2).

Three things are worth more than the rest here, because each pins a defect that
`processing/plotter.py` has and this subpackage exists not to inherit:

* **The loader survives what is actually on disk.** Every one of the 40
  hyperbolicity CSVs in `results/` is 1 byte, and the 20 thesis runs have no
  metadata sidecar. A loader that assumed readable, complete input would fail on
  essentially every real run.
* **Memory does not scale with run length.** The largest field history is 185 MB
  / 1.54M rows. `plotter.py` reads it whole and then pivots it.
* **The velocity profile is not re-derived.** `plotter.py` inlines the shifted
  Legendre basis and stops at `a3`, so an N=6 run renders as if it were N=3 with
  no warning. The page must agree with the package's own N-agnostic routine
  exactly, which only holds if it calls it.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import textwrap
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from swme import cli, report
from swme.report import assemble, data as report_data, history as report_history
from swme.report.style import DEFAULT_STYLE


TINY = textwrap.dedent("""
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
      resolution_x: 30
    numerics:
      order: 2
      t_end: 0.02
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
      store_history: true
      store_hyperbolicity: true
      history_stride: 2
      hyperbolicity_stride: 4
""")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    """A complete run on disk: CSVs, hyperbolicity diagnostics and a sidecar."""
    directory = tmp_path_factory.mktemp("tiny")
    (directory / "case.yaml").write_text(TINY)
    cli.run(directory / "case.yaml", output_dir=str(directory))
    return directory


@pytest.fixture(scope="module")
def tiny_run(tiny_dir):
    return report.from_directory(tiny_dir)


@pytest.fixture(scope="module")
def tiny_sim_run(tmp_path_factory):
    """The same case, but loaded in-process rather than from disk."""
    directory = tmp_path_factory.mktemp("tiny_sim")
    (directory / "case.yaml").write_text(TINY)
    config = cli.load_config(directory / "case.yaml")
    sim = cli.build_simulation(config)
    final = sim.run_simulation(0.02)
    return report.from_simulation(sim, final)


# ---------------------------------------------------------------------------
# discovery and loading
# ---------------------------------------------------------------------------

def test_prefix_is_inferred_from_the_final_state_file(tiny_run):
    assert tiny_run.prefix == "swme_N2"
    assert tiny_run.order == 2
    assert tiny_run.moment_columns == ("a1", "a2")
    assert tiny_run.source == "directory"


def test_missing_run_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="_final.csv"):
        report.from_directory(tmp_path)
    with pytest.raises(FileNotFoundError, match="No such directory"):
        report.from_directory(tmp_path / "nope")


def test_two_runs_in_one_directory_ask_which(tmp_path, tiny_dir):
    for name in ("swme_N2_final.csv", "hswme_N2_final.csv"):
        (tmp_path / name).write_text((tiny_dir / "swme_N2_final.csv").read_text())
    with pytest.raises(ValueError, match="more than one run"):
        report.from_directory(tmp_path)
    # ... and naming one resolves it.
    assert report.from_directory(tmp_path, prefix="hswme_N2").order == 2


def test_one_byte_hyperbolicity_files_are_treated_as_absent(tmp_path, tiny_dir):
    """The regression that matters most for real data.

    All 40 hyperbolicity CSVs in `results/` are a bare newline - stale output of
    a defect fixed in Step 7. They must read as "not captured", not as a parse
    error and not as an empty-but-valid result.
    """
    for name in ("swme_N2_final.csv", "swme_N2_summary_history.csv"):
        (tmp_path / name).write_text((tiny_dir / name).read_text())
    for name in ("recharge_hyperbolicity_history.csv",
                 "recharge_hyperbolicity_summary.csv"):
        (tmp_path / name).write_text("\n")

    run = report.from_directory(tmp_path)

    assert run.hyperbolicity is None
    assert any("empty" in warning for warning in run.warnings)


def test_legacy_hyperbolicity_filenames_are_still_read(tmp_path, tiny_dir):
    """Runs written before defect D6 used a hardcoded `recharge_` stem
    regardless of model, so both spellings must be accepted forever."""
    (tmp_path / "swme_N2_final.csv").write_text(
        (tiny_dir / "swme_N2_final.csv").read_text())
    (tmp_path / "recharge_hyperbolicity_summary.csv").write_text(
        (tiny_dir / "swme_N2_hyperbolicity_summary.csv").read_text())

    run = report.from_directory(tmp_path)
    assert run.hyperbolicity is not None
    assert run.hyperbolicity.summary is not None


def test_absent_sidecar_degrades_and_says_so(tmp_path, tiny_dir):
    (tmp_path / "swme_N2_final.csv").write_text(
        (tiny_dir / "swme_N2_final.csv").read_text())
    run = report.from_directory(tmp_path)

    assert run.meta.is_empty
    assert run.scheme_counters is None
    assert run.bed_elevation is None
    assert any("sidecar" in warning for warning in run.warnings)


def test_sidecar_supplies_what_the_csvs_cannot(tiny_run):
    assert tiny_run.meta.scheme == "Roe"
    assert tiny_run.scheme_counters is not None
    assert tiny_run.scheme_counters.scheme == "Roe"
    assert tiny_run.thresholds is not None
    assert tiny_run.thresholds.h_dry == 1e-4


def test_gapped_moment_columns_are_rejected():
    frame = pd.DataFrame(columns=["x", "h", "u_m", "a1", "a3"])
    with pytest.raises(ValueError, match="not contiguous"):
        report_data.moment_columns_of(frame)


def test_moment_columns_sort_numerically_not_lexically():
    frame = pd.DataFrame(columns=["x", "h", "u_m"] + [f"a{i}" for i in range(1, 12)])
    assert report_data.moment_columns_of(frame)[-1] == "a11"


def test_order_zero_is_legal():
    frame = pd.DataFrame(columns=["x", "h", "u_m"])
    assert report_data.moment_columns_of(frame) == ()


# ---------------------------------------------------------------------------
# the two loaders must agree
# ---------------------------------------------------------------------------

def test_both_loaders_describe_the_same_run(tiny_run, tiny_sim_run):
    assert tiny_sim_run.order == tiny_run.order
    assert tiny_sim_run.moment_columns == tiny_run.moment_columns
    assert np.allclose(tiny_sim_run.final_array(), tiny_run.final_array(),
                       rtol=1e-10, atol=1e-12)
    assert np.array_equal(tiny_sim_run.snapshots.steps, tiny_run.snapshots.steps)
    assert np.allclose(tiny_sim_run.snapshots.values, tiny_run.snapshots.values,
                       rtol=1e-10, atol=1e-12)


def test_summary_history_matches_the_writer(tiny_run, tiny_sim_run):
    """`report.data` recomputes the per-step statistics rather than importing
    the CLI, so that `swme.report` never depends on `swme.cli`. This is what
    catches the two implementations drifting apart."""
    from_disk = tiny_run.summary_history
    in_process = tiny_sim_run.summary_history

    assert list(from_disk.columns) == list(in_process.columns)
    for column in from_disk.columns:
        assert np.allclose(from_disk[column], in_process[column],
                           rtol=1e-10, atol=1e-12), column


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------

def test_select_steps_keeps_the_endpoints_and_spreads_evenly():
    steps = np.arange(0, 1000, 3)
    picked = report_history.select_steps(steps, 17)

    assert len(picked) <= 17
    assert picked[0] == steps[0] and picked[-1] == steps[-1]
    gaps = np.diff(picked)
    assert gaps.max() - gaps.min() <= 3          # evenly spaced up to rounding


def test_select_steps_is_a_no_op_when_under_the_cap():
    steps = [0, 5, 9]
    assert list(report_history.select_steps(steps, 100)) == steps


@pytest.mark.parametrize("chunk_rows", [7, 31, 10_000_000])
def test_streaming_is_independent_of_chunk_boundaries(tiny_dir, chunk_rows):
    """A single step's rows can straddle a chunk boundary; the result must not
    depend on where the boundary fell."""
    run = report.from_directory(tiny_dir, chunk_rows=chunk_rows)
    reference = report.from_directory(tiny_dir, chunk_rows=10_000_000)

    assert np.array_equal(run.snapshots.steps, reference.snapshots.steps)
    assert np.array_equal(run.snapshots.values, reference.snapshots.values)


def test_streamed_snapshots_match_a_naive_read(tiny_dir, tiny_run):
    whole = pd.read_csv(tiny_dir / "swme_N2_field_history.csv")
    columns = ["x", "h", "u_m", "a1", "a2"]
    for index, step in enumerate(tiny_run.snapshots.steps):
        expected = whole[whole["step"] == step].sort_values("x")[columns].to_numpy()
        assert np.array_equal(tiny_run.snapshots.values[index], expected)


def test_peak_memory_grows_far_slower_than_the_file(tmp_path):
    """The property the whole two-pass design exists for.

    The claim is sub-linearity, not flatness: the snapshot cube and the parser
    chunk are both fixed by their budgets, but pandas' own read machinery does
    scale a little with file size. What must not happen is proportional growth,
    which is what makes `plotter.py`'s whole-file `read_csv` unusable here.

    Measured for real: the 185 MB / 1.54M-row history in `results/` loads at a
    ~30 MB peak, i.e. about 6x smaller than the file rather than larger.
    """
    def write(path, n_steps, n_cells=40):
        x = np.linspace(-1.0, 1.0, n_cells)
        frames = [
            pd.DataFrame({"step": s, "time": 0.01 * s, "x": x,
                          "h": 1.0 + 0.1 * s, "u_m": 0.0, "a1": 0.0})
            for s in range(n_steps)
        ]
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)

    def peak_for(n_steps):
        path = tmp_path / f"h{n_steps}.csv"
        write(path, n_steps)
        steps = report_history.select_steps(range(n_steps), 20)
        tracemalloc.start()
        report_history.stream_field_history(
            path, steps_wanted=steps, columns=("x", "h", "u_m", "a1"),
            chunk_rows=500)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    small, large = peak_for(100), peak_for(1000)

    # Ten times the rows, at the same snapshot budget and chunk size. Anything
    # near 10x would mean the file is being materialised.
    assert large < small * 3.0, (small, large)
    # And in absolute terms the peak stays within a few MB, far under the
    # ~2 MB of CSV that 1000 steps of 40 cells occupies on disk.
    assert large < 8_000_000, large


def test_missing_step_in_the_field_history_is_reported(tmp_path, tiny_dir):
    path = tmp_path / "history.csv"
    frame = pd.read_csv(tiny_dir / "swme_N2_field_history.csv")
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="no rows for step"):
        report_history.stream_field_history(
            path, steps_wanted=[10 ** 9], columns=("x", "h", "u_m", "a1", "a2"))


# ---------------------------------------------------------------------------
# the velocity profile is the package's, not a copy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [0, 1, 2, 3, 6])
def test_profile_matches_the_package_implementation_exactly(order):
    """Exact equality, not a tolerance.

    Anything less would pass against a hand-rolled reimplementation that agreed
    to a few digits - which is precisely the defect in `processing/`, where six
    copies of this maths are each capped at `a2` or `a3` and truncate a
    higher-order run in silence.
    """
    from swme import pde as pde_module

    rng = np.random.default_rng(20260817 + order)
    n_cells = 12
    values = np.column_stack([
        np.linspace(-1.0, 1.0, n_cells),
        rng.uniform(0.5, 2.0, n_cells),
        rng.uniform(-0.5, 0.5, n_cells),
        *[rng.uniform(-0.3, 0.3, n_cells) for _ in range(order)],
    ])
    z = np.linspace(0.0, 1.0, 37)

    reference = pde_module.SWME1D(
        'lakeAtRest', 0.0, 1.0, False, False
    ).compute_vertical_velocity_profile(order, values, z)

    assert np.array_equal(report_data.vertical_velocity_profile(order, values, z),
                          reference)


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

def test_every_page_returns_a_figure_and_writes_nothing(tiny_run, tmp_path):
    before = set(os.listdir(tmp_path))
    for spec, figure in assemble.iter_figures(tiny_run):
        assert isinstance(figure, Figure), spec.key
        assert figure.get_axes(), spec.key
        figure.clear()
    assert set(os.listdir(tmp_path)) == before


@pytest.mark.parametrize("order", [0, 1, 3])
def test_pages_render_at_any_moment_order(tmp_path, order):
    config = tmp_path / "case.yaml"
    config.write_text(TINY.replace("order: 2", f"order: {order}"))
    cli.run(config, output_dir=str(tmp_path))

    run = report.from_directory(tmp_path)
    assert run.order == order
    for spec, figure in assemble.iter_figures(run):
        assert isinstance(figure, Figure), spec.key
        figure.clear()


def test_page_selection_follows_the_data(tiny_run):
    keys = [spec.key for spec in assemble.selected_pages(tiny_run)]

    assert keys[0] == "cover"
    assert "hyperbolicity" in keys, "the page must always be present"
    assert "wet_dry" in keys, "this case dries out"
    assert "topography" not in keys, "flat bed"
    # Order is the static DEFAULT_PAGES order, never data-dependent.
    assert keys == [spec.key for spec in assemble.DEFAULT_PAGES
                    if spec.key in set(keys)]


def test_time_history_page_states_what_is_missing_rather_than_going_blank(tiny_run):
    stripped = report_data.RunData(
        final=tiny_run.final, order=tiny_run.order,
        moment_columns=tiny_run.moment_columns, meta=tiny_run.meta,
        source="directory", summary_history=None, snapshots=None,
    )
    figure = assemble.DEFAULT_PAGES[2].render(stripped, DEFAULT_STYLE)
    text = " ".join(t.get_text() for t in figure.findobj(match=lambda o: hasattr(o, "get_text")))

    assert "store_history" in text
    figure.clear()


def test_hyperbolicity_page_never_implies_an_all_clear(tiny_run):
    """Tier 1 must be text, not a bare number, and must never read as
    reassurance for a scheme that measured nothing."""
    from swme.report import hyperbolicity as hyper_pages

    unknown = report_data.RunData(
        final=tiny_run.final, order=tiny_run.order,
        moment_columns=tiny_run.moment_columns, meta=report_data.RunMetadata(),
        source="directory")
    lines = " ".join(hyper_pages.scheme_counter_lines(unknown))
    assert "not recorded" in lines
    assert "0 of 0" not in lines

    lf = report_data.RunData(
        final=tiny_run.final, order=tiny_run.order,
        moment_columns=tiny_run.moment_columns, meta=tiny_run.meta,
        source="directory",
        scheme_counters=report_data.SchemeCounters(scheme="LF"))
    lines = " ".join(hyper_pages.scheme_counter_lines(lf))
    assert "never" in lines and "eigendecompose" in lines
    assert "all clear" not in lines.lower()


def test_scheme_counter_semantics():
    from swme.report.data import SchemeCounters

    assert SchemeCounters(scheme="Roe", spectra_examined=5).records_per_interface == 1
    assert SchemeCounters(scheme="Osher", spectra_examined=5).records_per_interface == 5
    assert SchemeCounters(scheme="LF").is_vacuous
    assert SchemeCounters(scheme="PRICE").is_vacuous
    assert SchemeCounters(scheme="Roe", spectra_examined=0).is_vacuous
    assert not SchemeCounters(scheme="Roe", spectra_examined=9).is_vacuous


def test_topography_page_appears_only_with_a_bed(tmp_path):
    config = tmp_path / "topo.yaml"
    config.write_text(
        TINY.replace("initial_condition: damBreak_dryBed",
                     "initial_condition: lakeAtRest")
        + textwrap.dedent("""
            topography:
              bed_profile: gaussian_bump
              amplitude: 0.2
              center: 0.0
              width: 0.2
              reference_water_level: 1.0
        """))
    cli.run(config, output_dir=str(tmp_path))

    run = report.from_directory(tmp_path)
    assert run.has_topography
    assert run.bed_elevation is not None and len(run.bed_elevation) == run.n_cells
    assert "topography" in [spec.key for spec in assemble.selected_pages(run)]


# ---------------------------------------------------------------------------
# hyperbolicity: recomputation and the maps page
# ---------------------------------------------------------------------------

def _cell_frame(magnitudes, *, step=0, time=0.0, legacy=False):
    """A per-cell frame; `legacy=True` writes the pre-Phase-1 `is_hyperbolic`.

    Before Step 8.5 a dry cell was written with `is_hyperbolic = 0`, asserting
    that its spectrum left the real axis when in fact none was computed.
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    evaluated = np.isfinite(magnitudes)
    if legacy:
        flag = np.where(evaluated & (magnitudes <= 1e-10), 1, 0)
    else:
        flag = np.where(evaluated, (magnitudes <= 1e-10).astype(float), np.nan)
    return pd.DataFrame({
        "step": step, "time": time,
        "cell_index": np.arange(len(magnitudes)),
        "x": np.linspace(-1.0, 1.0, len(magnitudes)),
        "max_abs_imag_eig": magnitudes,
        "is_hyperbolic": flag,
    })


def test_recompute_separates_dry_cells_from_genuine_loss():
    """The D3/D4 recomputation, on the exact mix that used to be misreported:
    8 dry cells, 1 genuinely complex, 31 clean."""
    from swme.report import hyperbolicity as hyper

    magnitudes = np.zeros(40)
    magnitudes[:8] = np.nan          # dry / not evaluated
    magnitudes[20] = 0.5             # genuine loss
    derived = hyper.recompute_summary(_cell_frame(magnitudes))

    row = derived.iloc[0]
    assert row["num_not_evaluated_cells"] == 8
    assert row["num_nonhyperbolic_cells"] == 1
    assert row["num_evaluated_cells"] == 32
    assert row["worst_cell_index"] == 20
    assert row["max_abs_imag_eig"] == 0.5
    # Denominator is evaluated cells, not the whole mesh.
    assert row["fraction_nonhyperbolic_cells"] == pytest.approx(1 / 32)


def test_recompute_is_correct_on_legacy_per_cell_files():
    """The fixes are not retroactive: every per-cell CSV already in `results/`
    keeps the old `is_hyperbolic = 0` for dry cells. The recomputation must
    ignore that column and read the NaN magnitudes instead."""
    from swme.report import hyperbolicity as hyper

    magnitudes = np.zeros(40)
    magnitudes[:8] = np.nan
    magnitudes[20] = 0.5
    legacy = hyper.recompute_summary(_cell_frame(magnitudes, legacy=True))
    modern = hyper.recompute_summary(_cell_frame(magnitudes, legacy=False))

    assert legacy.iloc[0]["num_nonhyperbolic_cells"] == 1
    assert legacy.iloc[0]["num_not_evaluated_cells"] == 8
    # The legacy flag column says 9 cells are "not hyperbolic"; the honest
    # count is 1, and both formats must land on the same answer.
    assert (_cell_frame(magnitudes, legacy=True)["is_hyperbolic"] == 0).sum() == 9
    pd.testing.assert_frame_equal(legacy, modern)


def test_worst_spectrum_is_not_hidden_by_a_preceding_dry_cell():
    """The D3 ordering hazard: a bad cell sitting after a NaN one."""
    from swme.report import hyperbolicity as hyper

    magnitudes = np.array([0.0, np.nan, 0.5, np.nan, 0.1])
    row = hyper.recompute_summary(_cell_frame(magnitudes)).iloc[0]

    assert row["worst_cell_index"] == 2
    assert row["max_abs_imag_eig"] == 0.5


def test_all_dry_step_reports_no_worst_cell():
    from swme.report import hyperbolicity as hyper

    row = hyper.recompute_summary(_cell_frame([np.nan] * 6)).iloc[0]

    assert row["num_evaluated_cells"] == 0
    assert row["worst_cell_index"] == -1
    assert np.isnan(row["max_abs_imag_eig"])


def test_worst_time_survives_a_nan_poisoned_column():
    """A legacy summary has NaN in `max_abs_imag_eig` at every step containing a
    dry cell; `idxmax` would pick the wrong row in silence."""
    from swme.report import hyperbolicity as hyper

    summary = pd.DataFrame({
        "time": [0.0, 1.0, 2.0, 3.0],
        "max_abs_imag_eig": [0.1, np.nan, 0.9, np.nan],
        "num_nonhyperbolic_cells": [0, 0, 2, 0],
    })

    assert hyper.worst_time(summary) == 2.0
    assert hyper.first_nonhyperbolic_time(summary) == 2.0
    assert hyper.worst_time(pd.DataFrame({
        "time": [0.0], "max_abs_imag_eig": [np.nan]})) is None


def test_maps_page_needs_the_per_cell_frame(tiny_run):
    from swme.report import hyperbolicity as hyper

    assert hyper.has_cell_spectra(tiny_run)
    assert "hyperbolicity_maps" in [s.key for s in assemble.selected_pages(tiny_run)]

    without = report_data.RunData(
        final=tiny_run.final, order=tiny_run.order,
        moment_columns=tiny_run.moment_columns, meta=tiny_run.meta,
        source="directory")
    keys = [s.key for s in assemble.selected_pages(without)]
    assert "hyperbolicity" in keys, "the overview page is always present"
    assert "hyperbolicity_maps" not in keys


def test_maps_page_renders_all_panels(tiny_run):
    from swme.report import hyperbolicity as hyper

    figure = hyper.page_hyperbolicity_maps(tiny_run, DEFAULT_STYLE)
    assert len(figure.get_axes()) >= 3          # three panels plus colorbars
    figure.clear()


def test_model_level_text_reports_dry_cells_as_such(tiny_run):
    """The wet-dry run has many dry cells and no genuine loss. The page must
    say so in those words, not report the dry cells as a finding."""
    from swme.report import hyperbolicity as hyper

    text = " ".join(hyper.model_level_lines(tiny_run))
    assert "not evaluated" in text
    assert "non-hyperbolic cells  0" in text
    assert "carry no moments" in text


def test_genuine_loss_is_reported_as_a_finding(tiny_run):
    """The branch no real shipped config reaches.

    Every thesis run is clean - 14.4M states replayed with zero loss - so the
    "we found something" path has to be exercised synthetically or not at all.
    N >= 2 SWME does lose hyperbolicity, in a narrow wedge of moment ratios,
    so this is a state the model genuinely admits.
    """
    from swme.report import hyperbolicity as hyper

    frames = []
    for step, time in enumerate([0.0, 0.5, 1.0]):
        magnitudes = np.zeros(20)
        if step >= 1:
            magnitudes[7:9] = 0.35          # a wet cell leaves the real axis
        frames.append(_cell_frame(magnitudes, step=step, time=time))
    cells = pd.concat(frames, ignore_index=True)

    run = report_data.RunData(
        final=tiny_run.final, order=tiny_run.order,
        moment_columns=tiny_run.moment_columns, meta=tiny_run.meta,
        source="directory",
        hyperbolicity=report_data.HyperbolicityData(cells=cells, tolerance=1e-10))

    derived = hyper.recompute_summary(cells)
    assert list(derived["num_nonhyperbolic_cells"]) == [0, 2, 2]
    assert hyper.first_nonhyperbolic_time(derived) == 0.5

    text = " ".join(hyper.model_level_lines(run))
    assert "Genuine loss of hyperbolicity" in text
    assert "t = 0.5" in text
    assert "HSWME" in text, "the reader should be told the direct check"

    for page in (hyper.page_hyperbolicity, hyper.page_hyperbolicity_maps):
        figure = page(run, DEFAULT_STYLE)
        assert figure.get_axes()
        figure.clear()


def test_recorded_summary_without_cells_is_labelled_as_uncorrectable(tiny_run):
    """When only the summary survives, its numbers cannot be fixed - so the
    page has to say that rather than present them as clean."""
    from swme.report import hyperbolicity as hyper

    summary_only = report_data.RunData(
        final=tiny_run.final, order=tiny_run.order,
        moment_columns=tiny_run.moment_columns, meta=tiny_run.meta,
        source="directory",
        hyperbolicity=report_data.HyperbolicityData(
            summary=tiny_run.hyperbolicity.summary, cells=None))

    text = " ".join(hyper.model_level_lines(summary_only))
    assert "D4" in text and "D3" in text
    figure = hyper.page_hyperbolicity(summary_only, DEFAULT_STYLE)
    figure.clear()


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------

def test_build_report_writes_a_multi_page_pdf(tiny_run, tmp_path):
    result = assemble.build_report(tiny_run, tmp_path / "report.pdf")

    assert result.path.exists()
    assert result.path.read_bytes()[:4] == b"%PDF"
    assert result.path.stat().st_size > 10_000
    assert result.page_keys == tuple(
        spec.key for spec in assemble.selected_pages(tiny_run))
    assert set(result.page_keys) & set(result.skipped_keys) == set()


def test_a_failing_page_leaves_no_partial_pdf(tiny_run, tmp_path):
    """A truncated report.pdf sitting beside the CSVs looks like a finished
    artifact, which is worse than no report at all."""
    def explode(run, style):
        raise RuntimeError("page failed")

    pages = (assemble.DEFAULT_PAGES[0],
             assemble.PageSpec("boom", "Boom", explode))
    target = tmp_path / "report.pdf"

    with pytest.raises(RuntimeError, match="page failed"):
        assemble.build_report(tiny_run, target, pages=pages)

    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_report_can_be_restricted_to_chosen_pages(tiny_run, tmp_path):
    only = tuple(spec for spec in assemble.DEFAULT_PAGES
                 if spec.key in {"cover", "velocity_profiles"})
    result = assemble.build_report(tiny_run, tmp_path / "small.pdf", pages=only)
    assert result.page_keys == ("cover", "velocity_profiles")


# ---------------------------------------------------------------------------
# import hygiene
# ---------------------------------------------------------------------------

def test_importing_the_report_package_does_not_pull_in_pyplot():
    """Step 8 moved matplotlib off the import path of a scripted run; the
    report must not put it back."""
    code = (
        "import sys; import swme.report as r; "
        "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot imported'; "
        "assert 'matplotlib' not in sys.modules, 'matplotlib imported'; "
        "print('ok')"
    )
    import subprocess
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_modules_have_no_import_time_side_effects(tmp_path, monkeypatch):
    """`processing/plotter.py` reads a relative config path and creates a
    directory at import. Nothing here may do either."""
    import matplotlib

    monkeypatch.chdir(tmp_path)
    before = dict(matplotlib.rcParams)

    for name in ("swme.report.data", "swme.report.history", "swme.report.style",
                 "swme.report.pages", "swme.report.assemble",
                 "swme.report.hyperbolicity", "swme.report"):
        importlib.reload(importlib.import_module(name))

    assert Path.cwd() == tmp_path
    assert list(tmp_path.iterdir()) == []
    assert dict(matplotlib.rcParams) == before


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_cli_report_flag_is_opt_in(tmp_path):
    config = tmp_path / "case.yaml"
    config.write_text(TINY)

    cli.run(config, output_dir=str(tmp_path))
    assert not (tmp_path / "report.pdf").exists()

    cli.run(config, output_dir=str(tmp_path), report=True)
    assert (tmp_path / "report.pdf").read_bytes()[:4] == b"%PDF"


def test_report_cli_rebuilds_from_a_directory(tiny_dir, tmp_path):
    from swme.report import cli as report_cli

    assert report_cli.main([str(tiny_dir), "-o", str(tmp_path / "out.pdf")]) == 0
    assert (tmp_path / "out.pdf").read_bytes()[:4] == b"%PDF"


def test_report_cli_reports_a_missing_directory(tmp_path, capsys):
    from swme.report import cli as report_cli

    assert report_cli.main([str(tmp_path / "absent")]) == 1
    assert "No such directory" in capsys.readouterr().err


def test_sidecar_round_trips_through_run_metadata(tiny_dir):
    payload = json.loads((tiny_dir / "swme_N2_run.json").read_text())
    meta = report_data.RunMetadata.from_dict(payload)

    assert meta.scheme == "Roe"
    assert meta.order == 2
    assert meta.domain == (-1.0, 1.0)
    assert meta.raw == payload
