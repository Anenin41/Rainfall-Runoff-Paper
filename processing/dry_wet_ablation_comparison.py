#!/usr/bin/env python3
"""
Dry-vs-wet moment-cascade comparison plots for the constant rainfall/exfiltration
ablation test.

Run:
    python dry_wet_ablation_comparison.py

Expected root folder:
    /home/anenin/Documents/Git/thesis/model/processing/Dry_Wet_Test

Expected structure:
    Dry_Wet_Test/
    ├── Dry_N1/recharge_swme_N1_constant_field_history.csv (source-free case)
    ├── Dry_N2/recharge_swme_N2_constant_field_history.csv (source-free case)
    ├── Wet_N1/recharge_swme_N1_constant_field_history.csv (source-active case)
    └── Wet_N2/recharge_swme_N2_constant_field_history.csv (source-active case)

Main output folder:
    Dry_Wet_Test/Dry_Wet_Comparison_Figures/

Generated figure groups:
    1. Bulk source-free/source-active ablation comparison, split into four figures.
    2. Source-active-minus-source-free bulk differences, split into four figures.
    3. N=2-minus-N=1 bulk differences, split into four figures.
    4. Moment amplitude comparison, split into four figures.
    5. Source-active-minus-source-free moment differences, split into four figures.
    6. Spatial profile comparisons for source-free and source-active cases, split into h and u_m.
    7. Source-active-minus-source-free spatial differences, split into h and u_m.
    8. Reconstructed vertical velocity profile comparison.
    9. Optional hyperbolicity diagnostics, split into two figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Literal, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# Absolute paths and user settings
# =============================================================================

ROOT_DIR = Path("/home/anenin/Documents/Git/thesis/model/processing/Dry_Wet_Test")
OUTPUT_DIR = ROOT_DIR / "Dry_Wet_Comparison_Figures"

REGIMES = ("Dry", "Wet")
ORDERS = (1, 2)

# The nearest stored simulation time is used internally.
# Titles display these requested times.
SELECTED_TIMES = (0.0, 0.5, 1.0)

# For vertical-profile comparison, use one common x-location per time for all curves.
# Recommended:
#   "wet_N2_peak_height" because it follows the active source-active N=2 pulse.
# Alternatives:
#   "fixed_x" or "wet_N2_peak_velocity".
PROFILE_X_MODE: Literal[
    "wet_N2_peak_height",
    "wet_N2_peak_velocity",
    "fixed_x",
] = "fixed_x"

FIXED_PROFILE_X = 0.5

INCLUDE_HYPERBOLICITY = False


# =============================================================================
# Plot styling
# =============================================================================

plt.style.use("tableau-colorblind10")


def plot_model_curve(
    ax: plt.Axes,
    x,
    y,
    order: int,
    label: str,
) -> None:
    """
    Plot one model-order curve using the thesis styling.

    N=1: orange, dash-dot, linewidth 2.0.
    N=2: black, dashed, linewidth 1.5.
    """
    if order == 1:
        color = "#E69F00"      # orange
        linestyle = "-."
        linewidth = 2.0
        zorder = 2

    elif order == 2:
        color = "#000000"      # black
        linestyle = "--"
        linewidth = 1.5
        zorder = 3

    else:
        color = None
        linestyle = "-"
        linewidth = 2.0
        zorder = order

    ax.plot(
        x,
        y,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=zorder,
    )


def plot_regime_curve(
    ax: plt.Axes,
    x,
    y,
    regime: str,
    label: str,
) -> None:
    """
    Plot one regime curve when the comparison is source-free versus source-active only.
    """
    if regime == "Dry":
        color = "#0072B2"      # blue
        linestyle = "--"
        linewidth = 2.0
        zorder = 1

    elif regime == "Wet":
        color = "#000000"      # black
        linestyle = "-"
        linewidth = 1.5
        zorder = 2

    else:
        color = None
        linestyle = "-"
        linewidth = 2.0
        zorder = 1

    ax.plot(
        x,
        y,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=zorder,
    )


def plot_regime_order_curve(
    ax: plt.Axes,
    x,
    y,
    regime: str,
    order: int,
    label: str,
) -> None:
    """
    Plot a curve where both regime and model order matter.

    Colour encodes model order:
        N=1: orange.
        N=2: black.

    Line style encodes regime:
        Source-free (Dry): dashed.
        Source-active (Wet): solid.

    This keeps the source-free/source-active distinction visible without losing the model-order
    colour convention used throughout the thesis figures.
    """
    if order == 1:
        color = "#E69F00"
        linewidth = 2.5
        zorder_base = 2

    elif order == 2:
        color = "#000000"
        linewidth = 1.5
        zorder_base = 3

    else:
        color = None
        linewidth = 2.0
        zorder_base = order

    if regime == "Dry":
        linestyle = "--"
        zorder = zorder_base

    elif regime == "Wet":
        linestyle = "-"
        zorder = zorder_base + 2

    else:
        linestyle = "-"
        zorder = zorder_base

    ax.plot(
        x,
        y,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        zorder=zorder,
    )


def plot_zero_line(ax: plt.Axes) -> None:
    """
    Draw a secondary zero-reference line.
    """
    ax.axhline(
        0.0,
        color="0.45",
        linestyle=":",
        linewidth=1.1,
        zorder=0,
    )


def as_flat_axes(axes) -> np.ndarray:
    """
    Return a matplotlib axes object or axes array as a flat numpy array.
    """
    return np.asarray(axes).ravel()


def force_axis_tick_labels(axes) -> None:
    """
    Force tick labels to appear on every subplot, even when axes are shared.
    """
    for ax in as_flat_axes(axes):
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=True,
            labelleft=True,
        )


def apply_common_xlim(axes, pad_fraction: float = 0.0) -> None:
    """
    Apply one common x-axis range to all axes.
    """
    flat_axes = as_flat_axes(axes)

    xmin = min(ax.get_xlim()[0] for ax in flat_axes)
    xmax = max(ax.get_xlim()[1] for ax in flat_axes)

    if pad_fraction > 0.0:
        width = xmax - xmin
        xmin -= pad_fraction * width
        xmax += pad_fraction * width

    for ax in flat_axes:
        ax.set_xlim(xmin, xmax)

    force_axis_tick_labels(flat_axes)


def apply_common_ylim(axes, pad_fraction: float = 0.0) -> None:
    """
    Apply one common y-axis range to all axes.
    """
    flat_axes = as_flat_axes(axes)

    ymin = min(ax.get_ylim()[0] for ax in flat_axes)
    ymax = max(ax.get_ylim()[1] for ax in flat_axes)

    if pad_fraction > 0.0:
        height = ymax - ymin
        ymin -= pad_fraction * height
        ymax += pad_fraction * height

    for ax in flat_axes:
        ax.set_ylim(ymin, ymax)

    force_axis_tick_labels(flat_axes)


# =============================================================================
# General helpers
# =============================================================================

def regime_display_name(regime: str) -> str:
    """
    Return the display name for a regime.
    
    Dry -> Source-free
    Wet -> Source-active
    """
    if regime == "Dry":
        return "Source-free"
    elif regime == "Wet":
        return "Source-active"
    else:
        return regime


def run_label(regime: str, order: int) -> str:
    """
    Return a compact label for plot legends.
    """
    display_regime = regime_display_name(regime)
    return f"{display_regime}, N={order}"


def run_folder(regime: str, order: int) -> str:
    """
    Return the folder name for a given regime/order pair.
    """
    return f"{regime}_N{order}"


def field_history_path(root_dir: Path, regime: str, order: int) -> Path:
    """
    Return path to the field-history CSV for one regime/order pair.
    """
    return (
        root_dir
        / run_folder(regime, order)
        / f"recharge_swme_N{order}_constant_field_history.csv"
    )


def summary_history_path(root_dir: Path, regime: str, order: int) -> Path:
    """
    Return path to the optional summary-history CSV for one regime/order pair.
    """
    return (
        root_dir
        / run_folder(regime, order)
        / f"recharge_swme_N{order}_constant_summary_history.csv"
    )


def hyperbolicity_summary_path(root_dir: Path, regime: str, order: int) -> Path:
    """
    Return path to the optional hyperbolicity summary CSV.
    """
    return root_dir / run_folder(regime, order) / "recharge_hyperbolicity_summary.csv"


def save_fig(fig: plt.Figure, path: Path) -> None:
    """
    Save a Matplotlib figure and close it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def nearest_time(df: pd.DataFrame, target_time: float) -> float:
    """
    Return the simulation time closest to the requested target time.
    """
    times = np.asarray(sorted(df["time"].dropna().unique()), dtype=float)

    if times.size == 0:
        raise ValueError("No time values found in dataframe.")

    return float(times[np.argmin(np.abs(times - target_time))])


def numeric_clean(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """
    Convert selected columns to numeric when they exist.
    """
    out = df.copy()

    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def snapshot_at_time(df: pd.DataFrame, target_time: float) -> pd.DataFrame:
    """
    Return the spatial snapshot closest to the requested time.
    """
    t = nearest_time(df, target_time)
    return df[np.isclose(df["time"], t)].sort_values("x").copy()


# =============================================================================
# Loading
# =============================================================================

def load_field_history(root_dir: Path, regime: str, order: int) -> pd.DataFrame:
    """
    Load one field-history CSV and attach metadata columns.

    Required columns:
        time, x, h, u_m

    Optional columns:
        a1, a2
    """
    path = field_history_path(root_dir, regime, order)

    if not path.exists():
        raise FileNotFoundError(f"Missing field-history CSV:\n{path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    df = numeric_clean(
        df,
        ["step", "time", "x", "h", "u_m", "a1", "a2"],
    )

    required = {"time", "x", "h", "u_m"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["time", "x", "h", "u_m"]).copy()
    df["regime"] = regime
    df["N"] = order
    df["run"] = run_label(regime, order)

    return df


def load_all_histories(root_dir: Path) -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Load all source-free/source-active and N=1/N=2 field histories.
    """
    histories: Dict[Tuple[str, int], pd.DataFrame] = {}

    for regime in REGIMES:
        for order in ORDERS:
            histories[(regime, order)] = load_field_history(
                root_dir=root_dir,
                regime=regime,
                order=order,
            )

    return histories


# =============================================================================
# Bulk statistics
# =============================================================================

def compute_bulk_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bulk statistics from field-history data.

    The mean discharge is computed as the spatial mean of h*u_m:
        q_bar = mean_x(h u_m)
    """
    work = df.copy()
    work["q"] = work["h"] * work["u_m"]

    stats = (
        work.groupby("time", as_index=False)
        .agg(
            h_mean=("h", "mean"),
            h_min=("h", "min"),
            h_max=("h", "max"),
            u_mean=("u_m", "mean"),
            q_mean=("q", "mean"),
        )
        .sort_values("time")
    )

    stats["h_range"] = stats["h_max"] - stats["h_min"]

    return stats


def build_bulk_table(
    histories: Dict[Tuple[str, int], pd.DataFrame],
) -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Compute bulk statistics for every run.
    """
    return {
        key: compute_bulk_statistics(df)
        for key, df in histories.items()
    }


def interp_to_reference(
    reference_time: np.ndarray,
    source: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """
    Interpolate a source series onto a reference time grid.
    """
    source = source.sort_values("time")

    return np.interp(
        reference_time,
        source["time"].to_numpy(),
        source[column].to_numpy(),
    )


# =============================================================================
# Bulk comparison figures
# =============================================================================

def plot_single_bulk_ablation_quantity(
    bulk: Dict[Tuple[str, int], pd.DataFrame],
    quantity: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot one source-free/source-active and N=1/N=2 bulk quantity as a standalone figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for regime in REGIMES:
        for order in ORDERS:
            stats = bulk[(regime, order)]

            plot_regime_order_curve(
                ax=ax,
                x=stats["time"],
                y=stats[quantity],
                regime=regime,
                order=order,
                label=run_label(regime, order),
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_bulk_ablation(
    bulk: Dict[Tuple[str, int], pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Plot source-free/source-active and N=1/N=2 bulk response as separate figures.
    """
    plot_single_bulk_ablation_quantity(
        bulk=bulk,
        quantity="h_mean",
        ylabel=r"$\overline{h}$",
        output_path=out_dir / "Dry_Wet_bulk_mean_height_ablation.pdf",
    )

    plot_single_bulk_ablation_quantity(
        bulk=bulk,
        quantity="h_range",
        ylabel=r"$\max_x h - \min_x h$",
        output_path=out_dir / "Dry_Wet_bulk_height_spread_ablation.pdf",
    )

    plot_single_bulk_ablation_quantity(
        bulk=bulk,
        quantity="u_mean",
        ylabel=r"$\overline{u_m}$",
        output_path=out_dir / "Dry_Wet_bulk_mean_velocity_ablation.pdf",
    )

    plot_single_bulk_ablation_quantity(
        bulk=bulk,
        quantity="q_mean",
        ylabel=r"$\overline{q}=\overline{h u_m}$",
        output_path=out_dir / "Dry_Wet_bulk_mean_discharge_ablation.pdf",
    )


def plot_single_recharge_minus_dry_bulk_quantity(
    bulk: Dict[Tuple[str, int], pd.DataFrame],
    quantity: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot one source-active-minus-source-free bulk difference as a standalone figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for order in ORDERS:
        source_free = bulk[("Dry", order)].sort_values("time")
        source_active = bulk[("Wet", order)].sort_values("time")

        t = source_free["time"].to_numpy()
        source_active_interp = interp_to_reference(t, source_active, quantity)
        diff = source_active_interp - source_free[quantity].to_numpy()

        plot_model_curve(
            ax=ax,
            x=t,
            y=diff,
            order=order,
            label=f"Source-Active - Source-Free, N={order}",
        )

    plot_zero_line(ax)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_recharge_minus_dry_bulk(
    bulk: Dict[Tuple[str, int], pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Plot source-active-minus-source-free bulk differences separately for each quantity.
    """
    plot_single_recharge_minus_dry_bulk_quantity(
        bulk=bulk,
        quantity="h_mean",
        ylabel=r"$\Delta\overline{h}$",
        output_path=out_dir / "Dry_Wet_bulk_wet_minus_dry_mean_height.pdf",
    )

    plot_single_recharge_minus_dry_bulk_quantity(
        bulk=bulk,
        quantity="h_range",
        ylabel=r"$\Delta(\max_x h - \min_x h)$",
        output_path=out_dir / "Dry_Wet_bulk_wet_minus_dry_height_spread.pdf",
    )

    plot_single_recharge_minus_dry_bulk_quantity(
        bulk=bulk,
        quantity="u_mean",
        ylabel=r"$\Delta\overline{u_m}$",
        output_path=out_dir / "Dry_Wet_bulk_wet_minus_dry_mean_velocity.pdf",
    )

    plot_single_recharge_minus_dry_bulk_quantity(
        bulk=bulk,
        quantity="q_mean",
        ylabel=r"$\Delta\overline{q}$",
        output_path=out_dir / "Dry_Wet_bulk_wet_minus_dry_mean_discharge.pdf",
    )


def plot_single_order_difference_bulk_quantity(
    bulk: Dict[Tuple[str, int], pd.DataFrame],
    quantity: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot one N=2-minus-N=1 bulk difference as a standalone figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for regime in REGIMES:
        n1 = bulk[(regime, 1)].sort_values("time")
        n2 = bulk[(regime, 2)].sort_values("time")

        t = n1["time"].to_numpy()
        n2_interp = interp_to_reference(t, n2, quantity)
        diff = n2_interp - n1[quantity].to_numpy()

        plot_regime_curve(
            ax=ax,
            x=t,
            y=diff,
            regime=regime,
            label=f"N=2 - N=1, {regime}",
        )

    plot_zero_line(ax)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_order_difference_bulk(
    bulk: Dict[Tuple[str, int], pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Plot N=2-minus-N=1 bulk differences for source-free and source-active regimes.
    """
    plot_single_order_difference_bulk_quantity(
        bulk=bulk,
        quantity="h_mean",
        ylabel=r"$\Delta_N\overline{h}$",
        output_path=out_dir / "Dry_Wet_bulk_N2_minus_N1_mean_height.pdf",
    )

    plot_single_order_difference_bulk_quantity(
        bulk=bulk,
        quantity="h_range",
        ylabel=r"$\Delta_N(\max_x h - \min_x h)$",
        output_path=out_dir / "Dry_Wet_bulk_N2_minus_N1_height_spread.pdf",
    )

    plot_single_order_difference_bulk_quantity(
        bulk=bulk,
        quantity="u_mean",
        ylabel=r"$\Delta_N\overline{u_m}$",
        output_path=out_dir / "Dry_Wet_bulk_N2_minus_N1_mean_velocity.pdf",
    )

    plot_single_order_difference_bulk_quantity(
        bulk=bulk,
        quantity="q_mean",
        ylabel=r"$\Delta_N\overline{q}$",
        output_path=out_dir / "Dry_Wet_bulk_N2_minus_N1_mean_discharge.pdf",
    )


# =============================================================================
# Moment statistics
# =============================================================================

def compute_moment_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute L2/Euclidean and maximum absolute moment amplitudes over time.

    The L2 norm is the discrete Euclidean norm over the stored spatial cells.
    Since all runs use the same grid, this is suitable for model-order and
    source-free/source-active comparison.
    """
    rows: list[dict[str, float]] = []

    for time_value, group in df.groupby("time"):
        row: dict[str, float] = {"time": float(time_value)}

        if "a1" in group.columns:
            a1 = group["a1"].to_numpy(dtype=float)
            row["a1_l2"] = float(np.linalg.norm(a1, ord=2))
            row["a1_max_abs"] = float(np.max(np.abs(a1)))

        if "a2" in group.columns:
            a2 = group["a2"].to_numpy(dtype=float)
            row["a2_l2"] = float(np.linalg.norm(a2, ord=2))
            row["a2_max_abs"] = float(np.max(np.abs(a2)))

        rows.append(row)

    return pd.DataFrame(rows).sort_values("time")


def build_moment_table(
    histories: Dict[Tuple[str, int], pd.DataFrame],
) -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Compute moment statistics for every run.
    """
    return {
        key: compute_moment_statistics(df)
        for key, df in histories.items()
    }


def plot_single_moment_ablation_quantity(
    moments: Dict[Tuple[str, int], pd.DataFrame],
    quantity: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot one source-free/source-active moment statistic as a standalone figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for regime in REGIMES:
        for order in ORDERS:
            stats = moments[(regime, order)]

            if quantity not in stats.columns:
                continue

            plot_regime_order_curve(
                ax=ax,
                x=stats["time"],
                y=stats[quantity],
                regime=regime,
                order=order,
                label=run_label(regime, order),
            )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_moment_ablation(
    moments: Dict[Tuple[str, int], pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Plot moment-amplitude statistics for all available runs.
    """
    plot_single_moment_ablation_quantity(
        moments=moments,
        quantity="a1_l2",
        ylabel=r"$\|\alpha_1\|_2$",
        output_path=out_dir / "Dry_Wet_moment_a1_l2_ablation.pdf",
    )

    plot_single_moment_ablation_quantity(
        moments=moments,
        quantity="a1_max_abs",
        ylabel=r"$\max_x |\alpha_1|$",
        output_path=out_dir / "Dry_Wet_moment_a1_max_abs_ablation.pdf",
    )

    plot_single_moment_ablation_quantity(
        moments=moments,
        quantity="a2_l2",
        ylabel=r"$\|\alpha_2\|_2$",
        output_path=out_dir / "Dry_Wet_moment_a2_l2_ablation.pdf",
    )

    plot_single_moment_ablation_quantity(
        moments=moments,
        quantity="a2_max_abs",
        ylabel=r"$\max_x |\alpha_2|$",
        output_path=out_dir / "Dry_Wet_moment_a2_max_abs_ablation.pdf",
    )


def plot_single_moment_wet_minus_dry_quantity(
    moments: Dict[Tuple[str, int], pd.DataFrame],
    quantity: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot one source-active-minus-source-free moment statistic as a standalone figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for order in ORDERS:
        source_free = moments[("Dry", order)].sort_values("time")
        source_active = moments[("Wet", order)].sort_values("time")

        if quantity not in source_free.columns or quantity not in source_active.columns:
            continue

        t = source_free["time"].to_numpy()
        source_active_interp = interp_to_reference(t, source_active, quantity)
        diff = source_active_interp - source_free[quantity].to_numpy()

        plot_model_curve(
            ax=ax,
            x=t,
            y=diff,
            order=order,
            label=f"Source-active - Source-free, N={order}",
        )

    plot_zero_line(ax)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_moment_wet_minus_dry(
    moments: Dict[Tuple[str, int], pd.DataFrame],
    out_dir: Path,
) -> None:
    """
    Plot source-active-minus-source-free differences in moment statistics.
    """
    plot_single_moment_wet_minus_dry_quantity(
        moments=moments,
        quantity="a1_l2",
        ylabel=r"$\Delta\|\alpha_1\|_2$",
        output_path=out_dir / "Dry_Wet_moment_wet_minus_dry_a1_l2.pdf",
    )

    plot_single_moment_wet_minus_dry_quantity(
        moments=moments,
        quantity="a1_max_abs",
        ylabel=r"$\Delta\max_x |\alpha_1|$",
        output_path=out_dir / "Dry_Wet_moment_wet_minus_dry_a1_max_abs.pdf",
    )

    plot_single_moment_wet_minus_dry_quantity(
        moments=moments,
        quantity="a2_l2",
        ylabel=r"$\Delta\|\alpha_2\|_2$",
        output_path=out_dir / "Dry_Wet_moment_wet_minus_dry_a2_l2.pdf",
    )

    plot_single_moment_wet_minus_dry_quantity(
        moments=moments,
        quantity="a2_max_abs",
        ylabel=r"$\Delta\max_x |\alpha_2|$",
        output_path=out_dir / "Dry_Wet_moment_wet_minus_dry_a2_max_abs.pdf",
    )


# =============================================================================
# Spatial profile figures
# =============================================================================

def plot_single_spatial_profile_by_regime(
    histories: Dict[Tuple[str, int], pd.DataFrame],
    regime: str,
    variable: str,
    ylabel: str,
    output_path: Path,
    selected_times: Iterable[float] = SELECTED_TIMES,
) -> None:
    """
    Plot one spatial variable for N=1 and N=2 within one regime.

    The x- and y-axis ranges are shared across the selected-time subplots.
    """
    selected_times = tuple(selected_times)
    ncols = len(selected_times)

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5.0 * ncols, 3.8),
        sharex=True,
        sharey=True,
    )

    if ncols == 1:
        axes = np.asarray([axes])

    for col, target_time in enumerate(selected_times):
        ax = axes[col]

        for order in ORDERS:
            snap = snapshot_at_time(histories[(regime, order)], target_time)

            plot_model_curve(
                ax=ax,
                x=snap["x"],
                y=snap[variable],
                order=order,
                label=f"N={order}",
            )

        ax.set_title(f"t = {target_time:.3f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    apply_common_xlim(axes)
    apply_common_ylim(axes)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_spatial_profiles_by_regime(
    histories: Dict[Tuple[str, int], pd.DataFrame],
    regime: str,
    out_dir: Path,
    selected_times: Iterable[float] = SELECTED_TIMES,
) -> None:
    """
    Plot h(x) and u_m(x) for N=1 and N=2 within one regime.
    """
    plot_single_spatial_profile_by_regime(
        histories=histories,
        regime=regime,
        variable="h",
        ylabel=r"$h$",
        output_path=out_dir / f"Dry_Wet_spatial_height_profiles_{regime}.pdf",
        selected_times=selected_times,
    )

    plot_single_spatial_profile_by_regime(
        histories=histories,
        regime=regime,
        variable="u_m",
        ylabel=r"$u_m$",
        output_path=out_dir / f"Dry_Wet_spatial_velocity_profiles_{regime}.pdf",
        selected_times=selected_times,
    )


def plot_single_spatial_wet_minus_dry(
    histories: Dict[Tuple[str, int], pd.DataFrame],
    variable: str,
    ylabel: str,
    output_path: Path,
    selected_times: Iterable[float] = SELECTED_TIMES,
) -> None:
    """
    Plot one source-active-minus-source-free spatial difference for N=1 and N=2.

    The x- and y-axis ranges are shared across the selected-time subplots.
    """
    selected_times = tuple(selected_times)
    ncols = len(selected_times)

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5.0 * ncols, 3.8),
        sharex=True,
        sharey=True,
    )

    if ncols == 1:
        axes = np.asarray([axes])

    for col, target_time in enumerate(selected_times):
        ax = axes[col]

        for order in ORDERS:
            dry = snapshot_at_time(histories[("Dry", order)], target_time)
            wet = snapshot_at_time(histories[("Wet", order)], target_time)

            x_ref = dry["x"].to_numpy(dtype=float)
            wet_interp = np.interp(
                x_ref,
                wet["x"].to_numpy(dtype=float),
                wet[variable].to_numpy(dtype=float),
            )

            diff = wet_interp - dry[variable].to_numpy(dtype=float)

            plot_model_curve(
                ax=ax,
                x=x_ref,
                y=diff,
                order=order,
                label=f"N={order}",
            )

        plot_zero_line(ax)

        ax.set_title(f"t = {target_time:.3f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    apply_common_xlim(axes)
    apply_common_ylim(axes)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_spatial_wet_minus_dry(
    histories: Dict[Tuple[str, int], pd.DataFrame],
    out_dir: Path,
    selected_times: Iterable[float] = SELECTED_TIMES,
) -> None:
    """
    Plot source-active-minus-source-free spatial differences for h and u_m.
    """
    plot_single_spatial_wet_minus_dry(
        histories=histories,
        variable="h",
        ylabel=r"$h_{\mathrm{source-active}}-h_{\mathrm{source-free}}$",
        output_path=out_dir / "Dry_Wet_spatial_wet_minus_dry_height.pdf",
        selected_times=selected_times,
    )

    plot_single_spatial_wet_minus_dry(
        histories=histories,
        variable="u_m",
        ylabel=r"$u_{m,\mathrm{source-active}}-u_{m,\mathrm{source-free}}$",
        output_path=out_dir / "Dry_Wet_spatial_wet_minus_dry_velocity.pdf",
        selected_times=selected_times,
    )

def plot_single_spatial_n2_dry_wet(
    histories: Dict[Tuple[str, int], pd.DataFrame],
    variable: str,
    ylabel: str,
    output_path: Path,
    selected_times: Iterable[float] = SELECTED_TIMES,
) -> None:
    """
    Plot one spatial variable for the N=2 source-free and source-active runs.

    This directly compares the second-order model with and without
    rainfall-runoff source terms.

    The x- and y-axis ranges are shared across the selected-time subplots.
    """
    selected_times = tuple(selected_times)
    ncols = len(selected_times)

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5.0 * ncols, 3.8),
        sharex=True,
        sharey=True,
    )

    if ncols == 1:
        axes = np.asarray([axes])

    order = 2

    for col, target_time in enumerate(selected_times):
        ax = axes[col]

        for regime in REGIMES:
            snap = snapshot_at_time(histories[(regime, order)], target_time)

            plot_regime_curve(
                ax=ax,
                x=snap["x"],
                y=snap[variable],
                regime=regime,
                label=f"{regime_display_name(regime)}, N=2",
            )

        ax.set_title(f"t = {target_time:.3f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    apply_common_xlim(axes)
    apply_common_ylim(axes)
    fig.tight_layout()
    save_fig(fig, output_path)


def plot_spatial_profiles_n2_dry_wet(
    histories: Dict[Tuple[str, int], pd.DataFrame],
    out_dir: Path,
    selected_times: Iterable[float] = SELECTED_TIMES,
) -> None:
    """
    Plot N=2 source-free/source-active spatial profile comparison.

    Creates two separate figures:
        1. h(x) for source-free N=2 and source-active N=2.
        2. u_m(x) for source-free N=2 and source-active N=2.
    """
    plot_single_spatial_n2_dry_wet(
        histories=histories,
        variable="h",
        ylabel=r"$h$",
        output_path=out_dir / "Dry_Wet_spatial_height_profiles_N2_dry_wet.pdf",
        selected_times=selected_times,
    )

    plot_single_spatial_n2_dry_wet(
        histories=histories,
        variable="u_m",
        ylabel=r"$u_m$",
        output_path=out_dir / "Dry_Wet_spatial_velocity_profiles_N2_dry_wet.pdf",
        selected_times=selected_times,
    )


# =============================================================================
# Vertical velocity reconstruction
# =============================================================================

def phi1(zeta: np.ndarray) -> np.ndarray:
    """
    First shifted Legendre basis function on [0,1].
    """
    return 1.0 - 2.0 * zeta


def phi2(zeta: np.ndarray) -> np.ndarray:
    """
    Second shifted Legendre basis function on [0,1].
    """
    return 1.0 - 6.0 * zeta + 6.0 * zeta**2


def reconstruct_vertical_velocity(
    row: pd.Series,
    zeta: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct u(zeta) from u_m, a1 and a2 when available.
    """
    profile = np.full_like(zeta, float(row["u_m"]), dtype=float)

    if "a1" in row.index and pd.notna(row["a1"]):
        profile += float(row["a1"]) * phi1(zeta)

    if "a2" in row.index and pd.notna(row["a2"]):
        profile += float(row["a2"]) * phi2(zeta)

    return profile


def common_profile_x(
    histories: Dict[Tuple[str, int], pd.DataFrame],
    target_time: float,
    mode: str = PROFILE_X_MODE,
    fixed_x: float = FIXED_PROFILE_X,
) -> float:
    """
    Choose one physical x-location for vertical-profile comparison.

    wet_N2_peak_height:
        Select x where the Wet N=2 height is maximal at this time.
    wet_N2_peak_velocity:
        Select x where the Wet N=2 mean velocity is maximal at this time.
    fixed_x:
        Use the fixed value FIXED_PROFILE_X.
    """
    if mode == "fixed_x":
        return float(fixed_x)

    reference = snapshot_at_time(histories[("Wet", 2)], target_time)

    if mode == "wet_N2_peak_height":
        return float(reference.loc[reference["h"].idxmax(), "x"])

    if mode == "wet_N2_peak_velocity":
        return float(reference.loc[reference["u_m"].idxmax(), "x"])

    raise ValueError(f"Unknown PROFILE_X_MODE: {mode}")


def row_at_time_and_x(
    df: pd.DataFrame,
    target_time: float,
    target_x: float,
) -> pd.Series:
    """
    Return the row closest to target_x at the nearest stored time.
    """
    snap = snapshot_at_time(df, target_time)
    distances = np.abs(snap["x"].to_numpy(dtype=float) - target_x)
    return snap.iloc[int(np.argmin(distances))]


VERTICAL_PROFILE_STYLE = {
    ("Dry", 1): {
        "color": "#F0B84D",      # light adjacent orange
        "linestyle": "-",
        "linewidth": 2.75,
        "zorder": 3,
    },
    ("Wet", 1): {
        "color": "#D98200",      # darker adjacent orange
        "linestyle": "--",
        "linewidth": 2.5,
        "zorder": 4,
    },
    ("Dry", 2): {
        "color": "#6A6A6A",      # dark grey adjacent to black
        "linestyle": "-.",
        "linewidth": 2.0,
        "zorder": 5,
    },
    ("Wet", 2): {
        "color": "#000000",      # keep thesis black
        "linestyle": ":",
        "linewidth": 2.75,
        "zorder": 6,
    },
}

def plot_vertical_profile_curve(
    ax: plt.Axes,
    x,
    y,
    regime: str,
    order: int,
    label: str,
) -> None:
    """
    Plot vertical-profile curves with a local palette.

    This keeps the thesis colour logic:
        N=1 remains in the orange family.
        N=2 remains in the black/grey family.

    The local shades are used only for the vertical-profile figure where
    the four curves overlap strongly.
    """
    style = VERTICAL_PROFILE_STYLE[(regime, order)]

    ax.plot(
        x,
        y,
        label=label,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=style["linewidth"],
        zorder=style["zorder"],
    )


def plot_vertical_profiles(
    histories: Dict[Tuple[str, int], pd.DataFrame],
    out_dir: Path,
    selected_times: Iterable[float] = SELECTED_TIMES,
    ) -> None:
    """
    Plot reconstructed vertical velocity profiles for source-free/source-active and N=1/N=2.

    The y-axis is common because all panels use zeta in [0,1]. The x-axis is
    also shared so profile magnitudes are comparable across selected times.

    This function uses a local colour override so the overlapping curves remain
    distinguishable without changing the global thesis colour scheme.
    """
    selected_times = tuple(selected_times)
    zeta = np.linspace(0.0, 1.0, 250)

    ncols = len(selected_times)

    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(5.0 * ncols, 5.0),
        sharex=True,
        sharey=True,
    )

    if ncols == 1:
        axes = np.asarray([axes])

    for col, target_time in enumerate(selected_times):
        x_ref = common_profile_x(histories, target_time)
        ax = axes[col]

        for regime in REGIMES:
            for order in ORDERS:
                row = row_at_time_and_x(
                    histories[(regime, order)],
                    target_time,
                    x_ref,
                )

                profile = reconstruct_vertical_velocity(row, zeta)

                plot_vertical_profile_curve(
                    ax=ax,
                    x=profile,
                    y=zeta,
                    regime=regime,
                    order=order,
                    label=(
                        f"{regime_display_name(regime)}, "
                        f"N={order}, "
                        f"x={row['x']:.3f}"
                    ),
                )

        ax.set_title(f"t = {target_time:.3f}")
        ax.set_xlabel(r"$u(\zeta)$")
        ax.set_ylabel(r"$\zeta$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, framealpha=0.95)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    apply_common_xlim(axes)
    apply_common_ylim(axes)

    fig.tight_layout()
    save_fig(fig, out_dir / "Dry_Wet_vertical_profiles_comparison.pdf")

# =============================================================================
# Hyperbolicity diagnostics
# =============================================================================

def load_hyperbolicity_summary(
    root_dir: Path,
    regime: str,
    order: int,
) -> pd.DataFrame | None:
    """
    Load a hyperbolicity summary if available, otherwise return None.
    """
    path = hyperbolicity_summary_path(root_dir, regime, order)

    if not path.exists():
        return None

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    df = numeric_clean(
        df,
        [
            "time",
            "num_nonhyperbolic_cells",
            "nonhyperbolic_cells",
            "max_abs_imag_eig",
            "max_imag",
            "max_imag_eig",
        ],
    )

    if "time" not in df.columns or df.empty:
        return None

    return df.sort_values("time")


def find_first_existing_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    """
    Return the first column name that exists in df from a candidate list.
    """
    for col in candidates:
        if col in df.columns:
            return col

    return None


def load_hyperbolicity_tables(
    root_dir: Path,
) -> Dict[Tuple[str, int], pd.DataFrame]:
    """
    Load all available hyperbolicity summary tables.
    """
    tables: Dict[Tuple[str, int], pd.DataFrame] = {}

    for regime in REGIMES:
        for order in ORDERS:
            df = load_hyperbolicity_summary(root_dir, regime, order)

            if df is not None:
                tables[(regime, order)] = df

    return tables


def plot_single_hyperbolicity_quantity(
    tables: Dict[Tuple[str, int], pd.DataFrame],
    column_candidates: Iterable[str],
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot one hyperbolicity diagnostic as a standalone figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    any_plotted = False

    for regime in REGIMES:
        for order in ORDERS:
            key = (regime, order)

            if key not in tables:
                continue

            df = tables[key]
            col = find_first_existing_column(df, column_candidates)

            if col is None:
                continue

            plot_regime_order_curve(
                ax=ax,
                x=df["time"],
                y=df[col],
                regime=regime,
                order=order,
                label=run_label(regime, order),
            )

            any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print(f"[skip] no usable hyperbolicity column found for {ylabel}")
        return

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    save_fig(fig, output_path)


def plot_hyperbolicity_summary(root_dir: Path, out_dir: Path) -> None:
    """
    Plot optional hyperbolicity diagnostics if the summary files exist.
    """
    tables = load_hyperbolicity_tables(root_dir)

    if not tables:
        print("[skip] no usable hyperbolicity summary files found")
        return

    plot_single_hyperbolicity_quantity(
        tables=tables,
        column_candidates=["num_nonhyperbolic_cells", "nonhyperbolic_cells"],
        ylabel="Count",
        output_path=out_dir / "Dry_Wet_hyperbolicity_nonhyperbolic_cell_count.pdf",
    )

    plot_single_hyperbolicity_quantity(
        tables=tables,
        column_candidates=["max_abs_imag_eig", "max_imag_eig", "max_imag"],
        ylabel=r"$\max |\operatorname{Im}(\lambda)|$",
        output_path=out_dir / "Dry_Wet_hyperbolicity_max_imaginary_eigenvalue.pdf",
    )


# =============================================================================
# Main generation wrapper
# =============================================================================

def validate_required_files(root_dir: Path) -> None:
    """
    Fail early if required folders or field-history files are missing.
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist:\n{root_dir}")

    missing = []

    for regime in REGIMES:
        for order in ORDERS:
            path = field_history_path(root_dir, regime, order)

            if not path.exists():
                missing.append(path)

    if missing:
        msg = (
            "Missing required field-history CSV files:\n"
            + "\n".join(str(p) for p in missing)
        )

        raise FileNotFoundError(msg)


def generate_all_figures() -> None:
    """
    Generate all source-free/source-active ablation figures.
    """
    validate_required_files(ROOT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[root] {ROOT_DIR}")
    print(f"[out ] {OUTPUT_DIR}")

    histories = load_all_histories(ROOT_DIR)
    bulk = build_bulk_table(histories)
    moments = build_moment_table(histories)

    plot_bulk_ablation(bulk, OUTPUT_DIR)
    plot_recharge_minus_dry_bulk(bulk, OUTPUT_DIR)
    plot_order_difference_bulk(bulk, OUTPUT_DIR)

    plot_moment_ablation(moments, OUTPUT_DIR)
    plot_moment_wet_minus_dry(moments, OUTPUT_DIR)

    for regime in REGIMES:
        plot_spatial_profiles_by_regime(
            histories=histories,
            regime=regime,
            out_dir=OUTPUT_DIR,
            selected_times=SELECTED_TIMES,
        )

    plot_spatial_wet_minus_dry(
        histories=histories,
        out_dir=OUTPUT_DIR,
        selected_times=SELECTED_TIMES,
    )

    plot_spatial_profiles_n2_dry_wet(
        histories=histories,
        out_dir=OUTPUT_DIR,
        selected_times=SELECTED_TIMES,
    )

    plot_vertical_profiles(
        histories=histories,
        out_dir=OUTPUT_DIR,
        selected_times=SELECTED_TIMES,
    )

    if INCLUDE_HYPERBOLICITY:
        plot_hyperbolicity_summary(ROOT_DIR, OUTPUT_DIR)

    print("\n[done] Source-free/source-active comparison figures generated.")


if __name__ == "__main__":
    generate_all_figures()