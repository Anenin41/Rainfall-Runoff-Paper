"""The hyperbolicity pages, and the recomputation that makes them trustworthy.

This is the part of the report most able to do damage, because a rendered
figure carries authority a warning line does not. Two separate hazards, and
each shapes the code below.

**Hazard one: the scheme counters do not measure the model.** The always-on
counters eigendecompose the *path-averaged interface matrix*
`sum_k w_k A(psi(s_k))`, not `A(U)` at any state. `A` is nonlinear in `U`, so
that average is not `A(anything)` and need not be hyperbolic even when every
matrix being averaged is. A dam break onto a dry bed trips it at 78 of 12 462
interfaces at **N = 0** - plain shallow water, unconditionally hyperbolic, no
moments to destabilise - and the count is identical at N=1, N=2 and HSWME,
which is what gives it away. They also mean different things per scheme: Roe
records one path average per interface, Osher five weight-scaled single-node
matrices, LF and PRICE nothing at all. So tier 1 is text, never a bare number,
and never renders a zero as reassurance.

**Hazard two: the recorded summary is wrong on older runs, permanently.**
Defects D3 and D4 were fixed in Phase 1, but that is not retroactive: every
`*_hyperbolicity_summary.csv` already in `results/` counts dry cells as
non-hyperbolic and has a NaN-latched worst-cell record. So the report derives
its own numbers from the per-cell frame via `recompute_summary` whenever that
frame is available, and says so when it cannot.

The recomputation needs nothing new on disk. A cell that was not evaluated -
dry, or a failed eigensolve - is exactly a row whose `max_abs_imag_eig` is NaN,
which is true in both the old and new formats.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pandas as pd

from .data import RunData
from .style import ReportStyle

INTERFACE_CAVEAT = (
    "These counters eigendecompose the path-averaged interface matrix, not "
    "A(U) at any state. That average is not A(anything), and it leaves the "
    "real axis at a strong enough jump even for models that are hyperbolic at "
    "every point - a wet-dry front does so at N = 0, i.e. for plain shallow "
    "water. A non-zero count here is therefore not evidence that the model "
    "lost hyperbolicity."
)

ENABLE_HINT = (
    "Set postprocessing.store_hyperbolicity: true (and hyperbolicity_stride) "
    "in the config and re-run to capture the cell-by-cell spectrum of A(U)."
)

STALE_SUMMARY_NOTE = (
    "Counts taken from the recorded summary, which on runs written before "
    "Step 8.5 includes dry cells among the non-hyperbolic ones (defect D4) and "
    "may carry a NaN-latched worst cell (D3). The per-cell frame is absent, so "
    "this cannot be corrected here."
)


# ---------------------------------------------------------------------------
# recomputation
# ---------------------------------------------------------------------------

def recompute_summary(cells: pd.DataFrame, tolerance: float = 1e-10) -> pd.DataFrame:
    """Derive honest per-step counts from the per-cell spectra.

    Independent of whether the run predates the D3/D4 fixes, because it reads
    only the raw per-cell quantities: a cell that was never evaluated is
    exactly one whose `max_abs_imag_eig` is NaN.

    Returns one row per recorded step with `num_nonhyperbolic_cells` (finite
    spectra that left the real axis - the only figure that says anything about
    the model), `num_not_evaluated_cells` (dry or failed), `num_evaluated_cells`,
    `fraction_nonhyperbolic_cells` over the evaluated ones, and the worst
    *finite* spectrum with its location.
    """
    rows = []
    for (step, time), group in cells.groupby(["step", "time"], sort=True):
        magnitude = group["max_abs_imag_eig"].to_numpy(dtype=np.float64)
        evaluated = np.isfinite(magnitude)
        n_evaluated = int(evaluated.sum())
        bad = evaluated & (magnitude > tolerance)

        if n_evaluated:
            best = int(np.nanargmax(np.where(evaluated, magnitude, -np.inf)))
            worst_value = float(magnitude[best])
            worst_index = int(group["cell_index"].to_numpy()[best])
            worst_x = float(group["x"].to_numpy()[best])
        else:
            worst_value, worst_index, worst_x = np.nan, -1, np.nan

        rows.append({
            "step": int(step),
            "time": float(time),
            "num_nonhyperbolic_cells": int(bad.sum()),
            "num_not_evaluated_cells": int((~evaluated).sum()),
            "num_evaluated_cells": n_evaluated,
            "fraction_nonhyperbolic_cells": (
                float(bad.sum()) / n_evaluated if n_evaluated else np.nan),
            "max_abs_imag_eig": worst_value,
            "worst_cell_index": worst_index,
            "worst_x": worst_x,
        })
    return pd.DataFrame(rows)


def first_nonhyperbolic_time(summary: pd.DataFrame) -> float | None:
    """Time of the first step with a genuinely non-hyperbolic cell, if any."""
    bad = summary[summary["num_nonhyperbolic_cells"] > 0]
    return float(bad["time"].iloc[0]) if len(bad) else None


def worst_time(summary: pd.DataFrame) -> float | None:
    """Time of the largest finite `|Im(lambda)|`.

    Uses `nanargmax` rather than `idxmax`: on a legacy summary the column is
    NaN-poisoned by defect D3, and `idxmax` would silently pick the wrong row.
    """
    magnitude = summary["max_abs_imag_eig"].to_numpy(dtype=np.float64)
    if not np.isfinite(magnitude).any():
        return None
    return float(summary["time"].to_numpy()[int(np.nanargmax(
        np.where(np.isfinite(magnitude), magnitude, -np.inf)))])


def _space_time(cells: pd.DataFrame, column: str):
    """(x, t, values) for a space-time map of one per-cell column.

    `pivot_table` rather than `pivot`, which raises outright on a duplicated
    (time, x) pair - possible whenever two steps share a time.
    """
    table = cells.pivot_table(index="time", columns="x", values=column,
                              aggfunc="max")
    return (table.columns.to_numpy(), table.index.to_numpy(), table.to_numpy())


# ---------------------------------------------------------------------------
# tier 1: the scheme counters, as prose
# ---------------------------------------------------------------------------

def _wrap(text: str, width: int = 66) -> list[str]:
    return ["  " + line for line in textwrap.wrap(text, width)]


def scheme_counter_lines(run: RunData) -> list[str]:
    """Counters plus what they do and do not mean.

    Never a bare number, and never "0 of 0" rendered as reassurance.
    """
    counters = run.scheme_counters
    if counters is None or counters.scheme is None:
        return [
            "SCHEME-LEVEL COUNTERS", "",
            "  Scheme not recorded, so the interface-matrix counters cannot be",
            "  interpreted and are not shown. Roe records one path-averaged",
            "  matrix per interface, Osher five weight-scaled single-node",
            "  matrices, and LF and PRICE never eigendecompose at all - the",
            "  same integer means a different thing in each case.",
        ]

    lines = ["SCHEME-LEVEL COUNTERS", "", f"  scheme                {counters.scheme}"]

    if not counters.eigendecomposes:
        lines += ["", *_wrap(
            f"{counters.scheme} applies a scalar or polynomial viscosity and never "
            "eigendecomposes the interface matrix. It records no spectra at all, "
            "so its counters are structurally zero and carry no information "
            "about this run. This is an absence of measurement, not a clean "
            "result.")]
        return lines

    lines += [
        f"  spectra recorded      {counters.spectra_examined}",
        f"  left the real axis    {counters.nonhyperbolic_count}",
        f"  largest |Im(lambda)|  {counters.max_abs_imaginary_eigenvalue:.3e}",
        f"  tolerance             {counters.tolerance:.1e}",
    ]
    if counters.spectra_examined == 0:
        lines += ["", "  No spectra were recorded, so nothing was measured."]
        return lines

    if counters.scheme == "Osher":
        lines += ["", *_wrap(
            "Osher records five spectra per interface, one per quadrature node, "
            "and each is a weight-scaled single-node matrix rather than the path "
            "average. Its counts and magnitudes are therefore not comparable "
            "with Roe's: the weights (0.118 to 0.284) scale the imaginary parts "
            "down, making the fixed tolerance up to 8.4x stricter in the units "
            "of A.")]
    lines += ["", *_wrap(INTERFACE_CAVEAT)]
    return lines


def model_level_lines(run: RunData) -> list[str]:
    """Tier 2 as prose: what the per-cell spectrum of `A(U)` actually showed."""
    hyper = run.hyperbolicity
    lines = ["MODEL-LEVEL SPECTRUM OF A(U)", ""]

    if hyper is None or not hyper.has_cells:
        if hyper is not None and hyper.summary is not None:
            return lines + ["  Per-cell frame absent; see the recorded summary "
                            "above.", "", *_wrap(STALE_SUMMARY_NOTE)]
        return lines + _wrap("Not captured. " + ENABLE_HINT)

    derived = recompute_summary(hyper.cells, hyper.tolerance)
    total_bad = int(derived["num_nonhyperbolic_cells"].sum())
    peak = float(np.nanmax(derived["max_abs_imag_eig"].to_numpy()))
    first = first_nonhyperbolic_time(derived)

    lines += [
        f"  steps recorded        {len(derived)}",
        f"  cells per step        {int(derived['num_evaluated_cells'].iloc[0]) + int(derived['num_not_evaluated_cells'].iloc[0])}",
        f"  non-hyperbolic cells  {total_bad}",
        f"  not evaluated (dry)   {int(derived['num_not_evaluated_cells'].sum())}",
        f"  largest |Im(lambda)|  {peak:.3e}" if np.isfinite(peak) else
        "  largest |Im(lambda)|  none evaluated",
        f"  tolerance             {hyper.tolerance:.1e}",
    ]
    if total_bad == 0:
        lines += ["", *_wrap(
            "No evaluated cell left the real axis at any recorded step. Dry "
            "cells are counted separately and are not evidence of anything - "
            "they carry no moments, so their spectrum says nothing about the "
            "model.")]
    else:
        lines += ["", *_wrap(
            f"Genuine loss of hyperbolicity, first at t = {first:.4g}. SWME is "
            "only unconditionally hyperbolic for N <= 1; HSWME always is, so "
            "re-running with hyperbolic: true is the direct check.")]
    return lines


# ---------------------------------------------------------------------------
# page 6: overview
# ---------------------------------------------------------------------------

def page_hyperbolicity(run: RunData, style: ReportStyle):
    """Tier 1 (scheme counters) and tier 2 (the model's own spectrum)."""
    figure, axes = style.new_page(3, 2, height_ratios=[1.35, 1.0, 1.0])
    style.page_header(figure, "Hyperbolicity",
                      "scheme-level counters and the model-level spectrum")

    style.text_block(axes[0, 0], scheme_counter_lines(run))
    style.text_block(axes[0, 1], model_level_lines(run))

    hyper = run.hyperbolicity
    derived = None
    if hyper is not None and hyper.has_cells:
        derived = recompute_summary(hyper.cells, hyper.tolerance)
    elif hyper is not None and hyper.summary is not None:
        derived = _adapt_recorded_summary(hyper.summary)

    if derived is None:
        style.placeholder(axes[1, 0], "Model-level spectrum not captured",
                          ENABLE_HINT)
        for position in ((1, 1), (2, 0), (2, 1)):
            axes[position].axis("off")
        return figure

    time = derived["time"].to_numpy()
    recomputed = hyper.has_cells

    # 1/7 - largest |Im(lambda)| against the tolerance.
    ax = axes[1, 0]
    magnitude = derived["max_abs_imag_eig"].to_numpy(dtype=np.float64)
    ax.plot(time, np.where(np.isfinite(magnitude), magnitude, np.nan),
            linewidth=style.line_width)
    ax.axhline(hyper.tolerance, color="tab:red", linestyle="--", linewidth=1.0,
               label="tolerance")
    if np.nanmax(magnitude) > 0:
        ax.set_yscale("symlog", linthresh=max(hyper.tolerance, 1e-16))
    style.axis(ax, "$t$", r"largest $|\mathrm{Im}\,\lambda|$",
               "worst spectrum per step", legend=True)

    # 2/7 - how many cells, split by population.
    ax = axes[1, 1]
    ax.plot(time, derived["num_nonhyperbolic_cells"], linewidth=style.line_width,
            label="non-hyperbolic")
    if "num_not_evaluated_cells" in derived:
        ax.plot(time, derived["num_not_evaluated_cells"], linewidth=1.0,
                linestyle="--", color="0.55", label="dry / not evaluated")
    style.axis(ax, "$t$", "cells", "affected cells per step", legend=True)
    style.caption(ax,
                  "Dry cells are drawn separately and deliberately: they carry "
                  "no moments, so counting them as non-hyperbolic conflates the "
                  "wet-dry treatment with a defect of the model."
                  if recomputed else STALE_SUMMARY_NOTE)

    # 3/7 - fraction, over evaluated cells only.
    ax = axes[2, 0]
    ax.plot(time, derived["fraction_nonhyperbolic_cells"],
            linewidth=style.line_width)
    style.axis(ax, "$t$", "fraction", "of evaluated cells")

    # 4/7 - where the worst cell sits.
    ax = axes[2, 1]
    worst_x = derived["worst_x"].to_numpy(dtype=np.float64)
    ax.plot(time, worst_x, ".", markersize=3)
    style.axis(ax, "$t$", "$x$ of worst cell", "location of the worst spectrum")
    return figure


def _adapt_recorded_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Present a recorded summary under the recomputed column names.

    Used only when the per-cell frame is missing, so the numbers cannot be
    corrected. Both the pre- and post-Phase-1 column sets are accepted.
    """
    frame = summary.copy()
    if "num_dry_cells" in frame.columns:
        frame["num_not_evaluated_cells"] = (
            frame["num_dry_cells"] + frame.get("num_failed_cells", 0))
    return frame


# ---------------------------------------------------------------------------
# page 7: the maps
# ---------------------------------------------------------------------------

def page_hyperbolicity_maps(run: RunData, style: ReportStyle):
    """Space-time and snapshot views of the per-cell spectrum."""
    figure, axes = style.new_page(3, 1, height_ratios=[1.0, 1.0, 1.0])
    hyper = run.hyperbolicity
    cells = hyper.cells
    derived = recompute_summary(cells, hyper.tolerance)
    style.page_header(
        figure, "Hyperbolicity in space and time",
        f"per-cell spectrum of A(U), {len(derived)} recorded steps")

    # 5/7 - space-time magnitude, floored then log-scaled.
    ax = axes[0, 0]
    x, t, values = _space_time(cells, "max_abs_imag_eig")
    norm = style.heatmap_norm(values)
    mesh = ax.pcolormesh(x, t, np.maximum(values, style.heatmap_floor),
                         cmap="magma", norm=norm, shading="nearest",
                         rasterized=True)
    figure.colorbar(mesh, ax=ax, label=r"$|\mathrm{Im}\,\lambda|$")
    style.axis(ax, "$x$", "$t$", "magnitude of the imaginary part")
    style.caption(ax, "Blank cells were never evaluated (dry, or a failed "
                      "eigensolve); they are not zeros.")

    # 6/7 - the same thing as a three-state classification.
    ax = axes[1, 0]
    magnitude = cells["max_abs_imag_eig"].to_numpy(dtype=np.float64)
    state = np.where(~np.isfinite(magnitude), np.nan,
                     np.where(magnitude > hyper.tolerance, 0.0, 1.0))
    classified = cells.assign(_state=state)
    x, t, values = _space_time(classified, "_state")
    mesh = ax.pcolormesh(x, t, values, cmap="RdYlGn", vmin=0.0, vmax=1.0,
                         shading="nearest", rasterized=True)
    bar = figure.colorbar(mesh, ax=ax, ticks=[0.0, 1.0])
    bar.ax.set_yticklabels(["lost", "hyperbolic"], fontsize=style.tick_size)
    style.axis(ax, "$x$", "$t$", "cell classification (white = not evaluated)")

    # 7/7 - profiles at the times that matter.
    ax = axes[2, 0]
    interesting = [("first step", float(derived["time"].iloc[0]))]
    peak = worst_time(derived)
    if peak is not None:
        interesting.append(("worst spectrum", peak))
    first_bad = first_nonhyperbolic_time(derived)
    if first_bad is not None:
        interesting.append(("first loss", first_bad))

    for label, moment in interesting:
        index = int(np.argmin(np.abs(derived["time"].to_numpy() - moment)))
        step = int(derived["step"].iloc[index])
        rows = cells[cells["step"] == step].sort_values("x")
        ax.plot(rows["x"], rows["max_abs_imag_eig"], linewidth=style.line_width,
                label=f"{label} ($t = {moment:.4g}$)")
    ax.axhline(hyper.tolerance, color="0.5", linestyle="--", linewidth=0.9,
               label="tolerance")
    if np.nanmax(magnitude) > 0:
        ax.set_yscale("symlog", linthresh=max(hyper.tolerance, 1e-16))
    style.axis(ax, "$x$", r"$|\mathrm{Im}\,\lambda|$",
               "spectrum across the domain", legend=True)
    return figure


def has_cell_spectra(run: RunData) -> bool:
    return run.hyperbolicity is not None and run.hyperbolicity.has_cells
