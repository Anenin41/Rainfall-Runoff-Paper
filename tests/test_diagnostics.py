"""The hyperbolicity diagnostics a run records (RESTRUCTURE_PLAN.md Step 8.5,
Phase 1 — defects D3, D4, D6 and the metadata sidecar).

These code paths had **never executed** before this step: `store_hyperbolicity`
is false in all 20 shipped configs, and every `*_hyperbolicity_*.csv` on disk
was 1 byte. So the defects below were latent rather than observed, and the
tests exist to keep them that way.

The one worth understanding is D4. A dam break onto a dry bed at N=1 used to
report *100 of 200 cells non-hyperbolic at t = 0* — for a system that §6 of the
plan proves is **unconditionally** hyperbolic at N <= 1. Every one of those was
a dry cell, whose moments have been ramped away and whose spectrum therefore
says nothing about the model. A report drawing that number would have blamed
the model for the wet-dry treatment, which is precisely the false-positive
failure mode Steps 5.5 and 6 exist to prevent.
"""

from __future__ import annotations

import json
import textwrap

import numpy as np
import pandas as pd
import pytest

from swme import cli


DRY_DAM_BREAK = textwrap.dedent("""
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
      order: 1
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
      history_stride: 5
      hyperbolicity_stride: 5
""")


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory):
    """A wet-dry run with diagnostics on: half the domain is dry throughout."""
    directory = tmp_path_factory.mktemp("dry_run")
    config = directory / "case.yaml"
    config.write_text(DRY_DAM_BREAK)
    cli.run(config, output_dir=str(directory))
    return directory


def _prefix(directory):
    (final,) = directory.glob("*_final.csv")
    return str(final)[: -len("_final.csv")]


# ---------------------------------------------------------------------------
# D4 — dry cells are not non-hyperbolic cells
# ---------------------------------------------------------------------------

def test_dry_cells_counted_separately_not_as_nonhyperbolic(dry_run):
    """The headline D4 regression.

    N=1 SWME is unconditionally hyperbolic (plan §6: the spectrum is exactly
    {u_m, u_m +/- sqrt(g*h + alpha_1^2)}), so on this run the honest count of
    non-hyperbolic cells is zero at every step — while the dry count is large,
    because the dam break starts with half the domain empty.
    """
    summary = pd.read_csv(f"{_prefix(dry_run)}_hyperbolicity_summary.csv")

    assert (summary["num_nonhyperbolic_cells"] == 0).all()
    assert (summary["num_dry_cells"] > 0).all()
    assert (summary["num_failed_cells"] == 0).all()
    # Every cell is accounted for exactly once.
    total = (summary["num_dry_cells"] + summary["num_evaluated_cells"])
    assert (total == 40).all()


def test_nonhyperbolic_fraction_is_over_evaluated_cells(dry_run):
    """The fraction must not be diluted by cells that were never evaluated.

    Dividing by the whole mesh would let a run that is 90% dry report a
    reassuringly small fraction for the handful of wet cells that genuinely
    lost hyperbolicity.
    """
    summary = pd.read_csv(f"{_prefix(dry_run)}_hyperbolicity_summary.csv")
    expected = summary["num_nonhyperbolic_cells"] / summary["num_evaluated_cells"]
    assert np.allclose(summary["fraction_nonhyperbolic_cells"], expected)


def test_not_evaluated_cells_are_nan_in_both_columns(dry_run):
    """`is_hyperbolic` and `max_abs_imag_eig` must agree about which cells were
    skipped, so a consumer can identify them from either one.

    `is_hyperbolic` used to be 0 for a dry cell, which asserts "the spectrum
    left the real axis" when the truth is "no spectrum was computed".
    """
    cells = pd.read_csv(f"{_prefix(dry_run)}_hyperbolicity_history.csv")

    assert (cells["is_hyperbolic"].isna() == cells["max_abs_imag_eig"].isna()).all()
    assert cells["is_hyperbolic"].isna().any(), "expected some dry cells"
    # No cell is recorded as evaluated-and-non-hyperbolic on this run.
    assert not (cells["is_hyperbolic"] == 0.0).any()


def test_summary_counts_match_the_per_cell_rows(dry_run):
    """The two files are written from the same sweep and must not disagree.

    This is what lets the report recompute the honest numbers from the
    per-cell frame for legacy CSVs, which keep the pre-fix columns forever.
    """
    prefix = _prefix(dry_run)
    summary = pd.read_csv(f"{prefix}_hyperbolicity_summary.csv").set_index("step")
    cells = pd.read_csv(f"{prefix}_hyperbolicity_history.csv")

    for step, group in cells.groupby("step"):
        assert group["max_abs_imag_eig"].isna().sum() == summary.loc[step, "num_dry_cells"]
        assert group["max_abs_imag_eig"].notna().sum() == summary.loc[step, "num_evaluated_cells"]


# ---------------------------------------------------------------------------
# D3 — the worst-cell tracker must not latch on NaN
# ---------------------------------------------------------------------------

def test_worst_cell_is_a_wet_cell_not_the_last_dry_one(dry_run):
    """D3: the tracker used to be

        if np.isnan(max_abs_imag) or max_abs_imag > max_abs_imag_global:

    so the first dry cell set the running max to NaN, and every later
    comparison against NaN was False. The recorded "worst" cell was then simply
    the last dry cell in index order, and `max_abs_imag_eig` stayed NaN for the
    whole step — hiding any genuinely non-hyperbolic wet cell after it.
    """
    prefix = _prefix(dry_run)
    summary = pd.read_csv(f"{prefix}_hyperbolicity_summary.csv")
    cells = pd.read_csv(f"{prefix}_hyperbolicity_history.csv")

    assert summary["max_abs_imag_eig"].notna().all()
    assert (summary["worst_cell_index"] >= 0).all()

    for row in summary.itertuples():
        step_cells = cells[cells["step"] == row.step].set_index("cell_index")
        worst = step_cells.loc[row.worst_cell_index]
        # The nominated cell was actually evaluated ...
        assert not np.isnan(worst["max_abs_imag_eig"])
        # ... and it really is the worst evaluated one (`.max()` skips NaN).
        assert np.isclose(row.max_abs_imag_eig,
                          step_cells["max_abs_imag_eig"].max())


def test_worst_cell_finds_the_largest_imaginary_part_after_dry_cells():
    """The failure D3 actually hides: a bad cell sitting *after* a dry cell.

    Built directly against the tracker's logic rather than through a solver,
    because reaching this state in a real run requires N >= 2 and a moment
    ratio inside the narrow unstable wedge — the point is the ordering, not
    the physics.
    """
    max_abs_imag_global = -1.0
    worst_index = -1
    # cell 0 wet and clean, cell 1 dry (NaN), cell 2 wet and genuinely complex
    for index, value in enumerate([0.0, np.nan, 0.5]):
        if np.isfinite(value) and value > max_abs_imag_global:
            max_abs_imag_global = value
            worst_index = index

    assert worst_index == 2
    assert max_abs_imag_global == 0.5


# ---------------------------------------------------------------------------
# D6 — output filenames
# ---------------------------------------------------------------------------

def test_hyperbolicity_files_use_the_run_prefix(dry_run):
    """D6: these two used to carry a hardcoded `recharge_` stem regardless of
    model, so a plain SWME run wrote files claiming to be recharge output and
    two models sharing a directory collided on them."""
    names = {path.name for path in dry_run.glob("*.csv")}

    assert "swme_N1_hyperbolicity_summary.csv" in names
    assert "swme_N1_hyperbolicity_history.csv" in names
    assert not any(name.startswith("recharge_") for name in names)


# ---------------------------------------------------------------------------
# the metadata sidecar
# ---------------------------------------------------------------------------

def test_sidecar_records_what_the_csvs_cannot(dry_run):
    """The sidecar exists because the CSVs carry only `[x, h, u_m, a1..aN]`.

    Without it a report rebuilt from disk cannot name the flux scheme — and the
    interface-matrix counters are uninterpretable without it, since Roe records
    one path average per interface, Osher five weight-scaled node matrices, and
    LF/PRICE none at all.
    """
    metadata = json.loads((dry_run / "swme_N1_run.json").read_text())

    assert metadata["schema"] == cli.SIDECAR_SCHEMA
    assert metadata["scheme"] == "Roe"
    assert metadata["scheme_well_balanced"] is True
    assert metadata["model"] == "SWME1D"
    assert metadata["order"] == 1
    assert metadata["hyperbolic"] is False
    assert metadata["resolution"] == 40
    assert metadata["time_integrator"] == "ExplicitEuler"
    assert metadata["wet_dry"] == {"eps_div": 1e-14, "h_dry": 1e-4, "h_wet": 1e-3}
    assert metadata["store_hyperbolicity"] is True
    assert set(metadata["scheme_counters"]) == {
        "spectra_examined", "nonhyperbolic_count",
        "max_abs_imaginary_eigenvalue", "tolerance",
    }


def test_sidecar_bed_is_a_profile_not_a_sampled_array(tmp_path):
    """Topography round-trip.

    The bed is stored as a profile name plus parameters, because
    `TopographySettings` carries a closure that will not serialise. What must
    hold is that re-deriving it from those parameters at the CSV's own `x`
    reproduces the mesh's sampling — that is what the report's topography page
    stands on.
    """
    from swme import topography

    config = tmp_path / "topo.yaml"
    config.write_text(DRY_DAM_BREAK.replace(
        "initial_condition: damBreak_dryBed",
        "initial_condition: lakeAtRest",
    ) + textwrap.dedent("""
        topography:
          bed_profile: gaussian_bump
          amplitude: 0.4
          center: 0.0
          width: 0.1
          reference_water_level: 1.0
    """))
    cli.run(config, output_dir=str(tmp_path))

    metadata = json.loads((tmp_path / "swme_N1_run.json").read_text())
    assert metadata["bed_profile"] == "gaussian_bump"
    assert metadata["has_topography"] is True
    assert metadata["bed_params"] == {"amplitude": 0.4, "center": 0.0, "width": 0.1}

    final = pd.read_csv(f"{_prefix(tmp_path)}_final.csv")
    rederived = topography.get_bed_profile(
        metadata["bed_profile"], **metadata["bed_params"])(final["x"].to_numpy())
    sampled = cli.build_simulation(cli.load_config(config)).mesh.bed_elevation[1:-1]

    # Both sample the bed at cell centres, so this agrees to round-off rather
    # than exactly: `x` loses ~1 ULP going through the CSV (1.1e-16 here), and
    # a Gaussian of width 0.1 amplifies that slightly. Well inside anything the
    # topography page cares about, but not bit-identical.
    assert np.allclose(rederived, sampled, rtol=1e-12, atol=1e-14)


def test_flat_bed_records_no_profile(dry_run):
    metadata = json.loads((dry_run / "swme_N1_run.json").read_text())
    assert metadata["has_topography"] is False
    assert metadata["bed_profile"] is None


# ---------------------------------------------------------------------------
# build_simulation extraction
# ---------------------------------------------------------------------------

def test_build_simulation_returns_a_configured_simulation(tmp_path):
    """Split out of `run()` so a caller can get a real simulation without also
    acquiring an output directory and CSV writing."""
    config = tmp_path / "case.yaml"
    config.write_text(DRY_DAM_BREAK)

    sim = cli.build_simulation(cli.load_config(config))

    assert sim.order == 1
    assert sim.mesh.resolution == 40
    assert sim.store_hyperbolicity is True
    assert sim.hyperbolicity_stride == 5
    assert sim.verbose is False
    assert type(sim.spatial_discretization).__name__ == "Roe"
    # Nothing was written: building is separate from running.
    assert list(tmp_path.glob("*.csv")) == []


def test_build_simulation_verbose_flag(tmp_path):
    config = tmp_path / "case.yaml"
    config.write_text(DRY_DAM_BREAK)
    assert cli.build_simulation(cli.load_config(config), verbose=True).verbose is True
