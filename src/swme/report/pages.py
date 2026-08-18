"""One function per report page.

Every page has the same shape:

    def page_<name>(run: RunData, style: ReportStyle) -> Figure

It builds a figure, returns it, and does nothing else - no `savefig`, no
`show`, no path argument, no directory creation, no module state. Saving is the
assembler's job, which is what makes a multi-page document possible at all;
none of `processing/plotter.py`'s fourteen plotting functions returns anything,
they all save-and-close one file each.

A page also never raises because data is missing. Either its `available`
predicate in `assemble.py` excluded it, or it draws an explicit placeholder
saying what is absent and how to capture it.

Layout is likewise the page's own business and is decided before anything is
drawn: a page that needs half the height uses half the height rather than
stretching two panels over A4, and a page whose text length is data-dependent
(the cover, whose warning list is not known until the run is read) measures the
text first and sizes its blocks to it. Nothing is laid out by eye.
"""

from __future__ import annotations

import numpy as np

from .data import RunData
from .style import Prose, ReportStyle


def _fmt(value, spec: str = ".6g") -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return format(float(value), spec)
    return str(value)


def _panel_grid(style: ReportStyle, panels: int, **kwargs):
    """A 2-column grid for `panels` plots, no taller than it needs to be.

    Two or more rows fill the page. A single row does not: stretched over the
    full height of A4 a lone pair of panels comes out at a 1:3 aspect, which is
    how an N = 0 run used to be drawn.
    """
    rows = int(np.ceil(panels / 2))
    return style.new_page(rows, 2, bottom=0.615 if rows == 1 else 0.075,
                          **kwargs)


def _blank_unused(axes, used: int) -> None:
    rows, cols = axes.shape
    for position in range(used, rows * cols):
        axes[position // cols, position % cols].axis("off")


# ---------------------------------------------------------------------------
# 1. cover
# ---------------------------------------------------------------------------

def page_cover(run: RunData, style: ReportStyle):
    """Three stacked blocks, each exactly as tall as the text inside it.

    The blocks used to have fixed height ratios, which left a hand's width of
    white space between sections on a short run and pushed the notes off the
    page on a run with several warnings. Here the text is wrapped to the known
    column width first, so the line count - and therefore the layout - is a
    measured quantity.
    """
    meta = run.meta
    left = [
        "MODEL",
        f"  type             {_fmt(meta.model)}",
        f"  closure          {'HSWME (hyperbolic)' if meta.hyperbolic else 'SWME'}",
        f"  moment order N   {run.order}",
        f"  initial cond.    {_fmt(meta.initial_condition)}",
        f"  viscosity        {_fmt(meta.viscosity)}",
        f"  slip length      {_fmt(meta.slip_length)}",
        "",
        "DISCRETISATION",
        f"  flux scheme      {_fmt(meta.scheme)}",
        f"  well balanced    {_fmt(meta.scheme_well_balanced)}",
        f"  time integrator  {_fmt(meta.time_integrator)}",
        f"  cells            {run.n_cells}",
        f"  domain           {_fmt(meta.domain)}",
        f"  boundary         {_fmt(meta.boundary_condition)}",
        f"  t_end            {_fmt(meta.t_end)}",
    ]
    if meta.model == "RechargeSWME1D":
        left += ["", "RECHARGE",
                 f"  infiltration     {_fmt(meta.infiltration_type)}",
                 f"  rainfall rate    {_fmt(meta.rainfall_rate)}"]

    thresholds = run.effective_thresholds
    right = [
        "WET-DRY",
        f"  h_dry            {_fmt(thresholds.h_dry)}",
        f"  h_wet            {_fmt(thresholds.h_wet)}",
        f"  min depth        {_fmt(run.min_depth)}",
        f"  went dry         {_fmt(run.went_dry)}",
        "",
        "DATA AVAILABLE",
    ]
    for label, present, note in run.inventory():
        right.append(f"  {'x' if present else '-'} {label:<20} {note}")
    right += ["",
              f"  source           {run.source}",
              f"  wall time        {_fmt(meta.elapsed_seconds, '.3g')} s",
              f"  generated        {_fmt(meta.generated_at)}"]

    notes = ["NOTES"]
    notes += ([Prose(f"- {line}") for line in run.warnings]
              or [Prose("- All expected outputs were present.")])

    # Measure against the column the blocks will actually occupy, so the line
    # counts - and with them the layout - are the ones that get drawn.
    margins, top = dict(left=0.11, right=0.95), 0.895
    width = style.column_width(**margins)
    blocks = [style.measure(left, width),
              style.measure(right, width),
              style.measure(notes, width, size=style.caption_size + 0.5,
                            family="sans-serif")]
    heights = [height for _, _, height in blocks]
    gap = 0.038                                    # between sections
    bottom = max(0.05, top - sum(heights) - 2 * gap)

    figure, axes = style.new_page(
        3, 1, height_ratios=heights, bottom=bottom, top=top,
        hspace=3.0 * gap / max(sum(heights), 1e-6), **margins)
    style.page_header(figure, run.title, "Run report")
    for index, (lines, size, _) in enumerate(blocks):
        style.text_block(axes[index, 0], lines, size=size,
                         family="sans-serif" if index == 2 else "monospace")
    return figure


# ---------------------------------------------------------------------------
# 2. final state
# ---------------------------------------------------------------------------

def page_final_state(run: RunData, style: ReportStyle):
    """`h`, `u_m` and every moment at t_end.

    The panel count follows N with no upper bound, unlike `swme/plotting.py`'s
    fixed 3x3 grid, which silently overflows from N = 6.
    """
    panels = 2 + run.order
    figure, axes = _panel_grid(style, panels)
    style.page_header(figure, "Final state",
                      f"t = {_fmt(run.meta.t_end)}, {run.n_cells} cells")

    x = run.x
    series = [("h", "water depth $h$"), ("u_m", "mean velocity $u_m$")]
    series += [(name, rf"moment $\alpha_{{{index}}}$")
               for index, name in enumerate(run.moment_columns, start=1)]

    for position, (column, label) in enumerate(series):
        ax = axes[position // 2, position % 2]
        ax.plot(x, run.final[column].to_numpy(), linewidth=style.line_width)
        if column == "h" and run.has_topography:
            ax.plot(x, run.bed_elevation, color="0.45", linewidth=1.0,
                    label="bed $Z$")
        style.axis(ax, "$x$", label, legend=column == "h" and run.has_topography)

    _blank_unused(axes, len(series))
    return figure


# ---------------------------------------------------------------------------
# 3. time histories
# ---------------------------------------------------------------------------

def page_time_histories(run: RunData, style: ReportStyle):
    panels = 2 + run.order
    summary = run.summary_history
    if summary is None:
        figure, axes = style.new_page(1, 1, bottom=0.60)
        style.page_header(figure, "Time histories", "spatial statistics per step")
        style.placeholder(
            axes[0, 0], "Time histories not stored",
            "Set postprocessing.store_history: true (and history_stride) "
            "in the config and re-run.")
        return figure

    figure, axes = _panel_grid(style, panels)
    style.page_header(figure, "Time histories", "spatial statistics per step")
    time = summary["time"].to_numpy()

    ax = axes[0, 0]
    ax.plot(time, summary["mean_h"], linewidth=style.line_width, label="mean")
    if {"min_h", "max_h"} <= set(summary.columns):
        ax.fill_between(time, summary["min_h"], summary["max_h"],
                        alpha=0.20, linewidth=0, label="min-max")
    style.axis(ax, "$t$", "water depth $h$", legend=True)

    ax = axes[0, 1]
    ax.plot(time, summary["mean_u_m"], linewidth=style.line_width)
    # The CSV records only a mean for u_m - no min/max columns are written.
    style.axis(ax, "$t$", "mean velocity $u_m$")

    for index, name in enumerate(run.moment_columns, start=1):
        position = 1 + index
        ax = axes[position // 2, position % 2]
        ax.plot(time, summary[f"mean_{name}"], linewidth=style.line_width,
                label="mean")
        if {f"min_{name}", f"max_{name}"} <= set(summary.columns):
            ax.fill_between(time, summary[f"min_{name}"], summary[f"max_{name}"],
                            alpha=0.20, linewidth=0, label="min-max")
        style.axis(ax, "$t$", rf"moment $\alpha_{{{index}}}$", legend=True)

    _blank_unused(axes, panels)
    return figure


# ---------------------------------------------------------------------------
# 4. space-time maps
# ---------------------------------------------------------------------------

def page_space_time(run: RunData, style: ReportStyle):
    # The right margin holds a colourbar's tick labels and its rotated label,
    # which is half an inch of type that has to land on the page.
    figure, axes = style.new_page(2, 1, hspace=0.30, bottom=0.10, right=0.90)
    snapshots = run.snapshots
    style.page_header(
        figure, "Space-time evolution",
        f"{len(snapshots.steps)} of {snapshots.total_available} stored steps")

    x = snapshots.x
    for row, (name, label) in enumerate([("h", "water depth $h$"),
                                         ("u_m", "mean velocity $u_m$")]):
        ax = axes[row, 0]
        mesh = ax.pcolormesh(x, snapshots.times, snapshots.field(name),
                             cmap=style.colormap, shading="nearest",
                             rasterized=True)
        style.axis(ax, "$x$", "$t$", label)
        style.colorbar(ax, mesh, label)
    return figure


# ---------------------------------------------------------------------------
# 5. vertical velocity profiles
# ---------------------------------------------------------------------------

def page_velocity_profiles(run: RunData, style: ReportStyle):
    """`u(z) = u_m + sum_i alpha_i phi_i(z)` at a few positions and times.

    The whole point of a moment model is this profile, and it is computed by
    `SWME1D.compute_vertical_velocity_profile` rather than re-derived here -
    the six hand-rolled copies in `processing/` are each capped at `a_2` or
    `a_3` and truncate a higher-order run in silence.
    """
    figure, axes = style.new_page(2, 2, wspace=0.42, hspace=0.34, right=0.92)
    style.page_header(figure, "Vertical velocity profiles",
                      r"$u(z) = u_m + \sum_i \alpha_i\,\phi_i(z)$, "
                      r"$\phi_i(z) = P_i(1-2z)$")

    z = np.linspace(0.0, 1.0, 100)
    final = run.final_array()
    profiles = run.velocity_profile(final, z)
    x = run.x

    # Sample positions across the wet part of the domain.
    wet = np.flatnonzero(run.final["h"].to_numpy() > run.effective_thresholds.h_wet)
    pool = wet if wet.size else np.arange(len(x))
    picks = np.unique(np.linspace(0, pool.size - 1,
                                  min(style.max_profile_curves, pool.size))
                      .round().astype(int))
    cells = pool[picks]

    ax = axes[0, 0]
    for cell in cells:
        ax.plot(profiles[cell], z, linewidth=style.line_width,
                label=f"$x = {x[cell]:.3g}$")
    style.axis(ax, "$u(z)$", "$z$ (0 = bed, 1 = surface)",
               "profiles at $t_{end}$", legend=True)

    # The same cells as a space-z map at the final time.
    ax = axes[0, 1]
    mesh = ax.pcolormesh(x, z, profiles.T, cmap=style.colormap,
                         shading="nearest", rasterized=True)
    style.axis(ax, "$x$", "$z$", "$u(x, z)$ at $t_{end}$")
    style.colorbar(ax, mesh, "$u$")

    # Deviation from plug flow: what the moments actually buy.
    ax = axes[1, 0]
    deviation = profiles - run.final["u_m"].to_numpy()[:, None]
    if run.order == 0:
        style.placeholder(
            ax, "No vertical structure at N = 0",
            "The N = 0 model carries no moments, so u(z) is the constant "
            "u_m and the deviation is identically zero.")
    else:
        limit = float(np.nanmax(np.abs(deviation))) or 1.0
        mesh = ax.pcolormesh(x, z, deviation.T, cmap="coolwarm",
                             vmin=-limit, vmax=limit, shading="nearest",
                             rasterized=True)
        style.axis(ax, "$x$", "$z$", "departure from plug flow")
        style.colorbar(ax, mesh, "$u - u_m$")

    # Time evolution of the profile at mid-domain, when history exists.
    ax = axes[1, 1]
    if run.snapshots is None:
        style.placeholder(
            ax, "No field history",
            "Profiles over time need postprocessing.store_history: true.")
    else:
        snapshots = run.snapshots
        cell = int(cells[len(cells) // 2])
        for index in style.curve_times(snapshots.times):
            values = snapshots.values[index]
            curve = run.velocity_profile(values, z)[cell]
            ax.plot(curve, z, linewidth=style.line_width,
                    label=f"$t = {snapshots.times[index]:.3g}$")
        style.axis(ax, "$u(z)$", "$z$",
                   f"at $x = {x[cell]:.3g}$ over time", legend=True)
    return figure


# ---------------------------------------------------------------------------
# 8. wet-dry
# ---------------------------------------------------------------------------

def page_wet_dry(run: RunData, style: ReportStyle):
    # `right` leaves room for the widest colourbar tick label, "transition".
    figure, axes = style.new_page(3, 1, height_ratios=[1.0, 1.0, 0.65],
                                  hspace=0.30, right=0.89)
    thresholds = run.effective_thresholds
    style.page_header(
        figure, "Wet-dry behaviour",
        f"h_dry = {thresholds.h_dry:.3g}, h_wet = {thresholds.h_wet:.3g}")

    x = run.x
    h = run.final["h"].to_numpy()

    ax = axes[0, 0]
    ax.semilogy(x, np.maximum(h, 1e-300), linewidth=style.line_width, label="$h$")
    ax.axhline(thresholds.h_wet, color="tab:blue", linestyle="--", linewidth=1.0,
               label="$h_{wet}$")
    ax.axhline(thresholds.h_dry, color="tab:red", linestyle=":", linewidth=1.0,
               label="$h_{dry}$")
    # A dry cell holds exactly 0, and an unbounded log axis then runs from
    # 1e-315 to the free surface: 300 decades of nothing, in which the two
    # thresholds the panel is about are a single line. The view stops three
    # decades below h_dry and the empty cells fall off the bottom.
    ax.set_ylim(thresholds.h_dry * 1e-3,
                max(1.6 * float(np.max(h)), 10.0 * thresholds.h_wet))
    style.axis(ax, "$x$", "depth (log)", "final depth against the thresholds",
               legend=True)
    style.caption(
        ax, "Below h_dry a cell carries no moments and its velocity is driven "
            "to zero; between the thresholds the moments ramp linearly. A "
            "vacuum front always sits below h_dry, which is why the computed "
            "front lags the exact one (see RESTRUCTURE_PLAN.md section 6). "
            "Cells holding exactly zero cannot be drawn on a log axis and run "
            "off the bottom of it.")

    ax = axes[1, 0]
    if run.snapshots is not None:
        depths = run.snapshots.field("h")
        state = np.full(depths.shape, 2.0)                 # wet
        state[depths < thresholds.h_wet] = 1.0             # transition
        state[depths <= thresholds.h_dry] = 0.0            # dry
        mesh = ax.pcolormesh(run.snapshots.x, run.snapshots.times, state,
                             cmap="RdYlBu", vmin=0, vmax=2, shading="nearest",
                             rasterized=True)
        style.axis(ax, "$x$", "$t$", "wet-dry state over time")
        bar = style.colorbar(ax, mesh, ticks=[0, 1, 2])
        bar.ax.set_yticklabels(["dry", "transition", "wet"],
                               fontsize=style.tick_size)
    else:
        style.placeholder(ax, "No field history",
                          "The wet-dry map over time needs store_history: true.")

    counts = {
        "dry (h <= h_dry)": int(np.sum(h <= thresholds.h_dry)),
        "transition": int(np.sum((h > thresholds.h_dry) & (h < thresholds.h_wet))),
        "wet (h >= h_wet)": int(np.sum(h >= thresholds.h_wet)),
    }
    lines = ["FINAL-STATE CELL COUNTS"]
    lines += [f"  {label:<22} {count:>6}  ({100 * count / run.n_cells:.1f} %)"
              for label, count in counts.items()]
    lines += ["", f"  minimum depth reached  {run.min_depth:.6g}"]
    if run.meta.mass_created_by_clamping:
        lines += [f"  mass added by clamping {run.meta.mass_created_by_clamping:.3e}"]
    style.text_block(axes[2, 0], lines)
    return figure


# ---------------------------------------------------------------------------
# 9. topography
# ---------------------------------------------------------------------------

def page_topography(run: RunData, style: ReportStyle):
    """Bed, free surface, and the departure from a lake at rest.

    Plots `h + Z` rather than bare `h`, following `swme/plotting.py`: over a
    non-flat bed a lake at rest looks like an inverted bump in `h` alone, which
    reads as a solver failure when it is exactly the correct answer.
    """
    figure, axes = style.new_page(3, 1, hspace=0.30)
    style.page_header(figure, "Topography and well-balancing",
                      f"bed profile: {run.meta.bed_profile}")

    x = run.x
    bed = run.bed_elevation
    h = run.final["h"].to_numpy()
    surface = h + bed

    ax = axes[0, 0]
    ax.fill_between(x, 0.0, bed, color="0.75", linewidth=0, label="bed $Z$")
    ax.fill_between(x, bed, surface, alpha=0.45, linewidth=0, label="water")
    ax.plot(x, surface, linewidth=style.line_width, label="free surface $h+Z$")
    style.axis(ax, "$x$", "elevation", "final free surface over the bed",
               legend=True)

    ax = axes[1, 0]
    reference = run.meta.reference_water_level
    if reference is None:
        style.placeholder(
            ax, "No reference water level recorded",
            "The h + Z - H residual needs topography.reference_water_level "
            "from the run's sidecar.")
    else:
        residual = surface - reference
        ax.plot(x, residual, linewidth=style.line_width)
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        style.axis(ax, "$x$", "$h + Z - H$",
                   f"departure from the still water level $H = {reference:g}$")
        peak = float(np.max(np.abs(residual)))
        balanced = run.meta.scheme_well_balanced
        style.caption(
            ax,
            f"max |h + Z - H| = {peak:.3e}. For a lake-at-rest case this is the "
            "C-property residual and should sit at round-off. It only can if the "
            "scheme's viscosity polynomial satisfies P(0) = 0 - true for Roe and "
            f"Osher, false for LF and PRICE. This run used "
            f"{run.meta.scheme or 'an unrecorded scheme'} "
            f"(well balanced: {'yes' if balanced else 'no' if balanced is not None else 'unknown'}).")

    ax = axes[2, 0]
    ax.plot(x, bed, color="0.45", linewidth=style.line_width)
    style.axis(ax, "$x$", "bed $Z$",
               "bed as re-derived from the recorded profile")
    style.caption(
        ax, "The CSVs carry no bed column; this is the sidecar's profile name "
            "and parameters evaluated at the CSV's own cell centres, which "
            "reproduces the mesh sampling to round-off.")
    return figure
