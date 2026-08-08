#!/usr/bin/env python3
"""
Non-wrapping pulse model-comparison plots.

No argparse version.
Edit ROOT_DIR and OUTPUT_DIR below if your folder location changes, then run:

    python non_wrapping_pulse_model_comparison.py

Expected folder structure
-------------------------
ROOT_DIR should point to the Non_Wrapping_Pulse folder:

    /home/anenin/Documents/Git/thesis/model/processing/Non_Wrapping_Pulse

and this folder should contain:

    Non_Wrapping_Pulse/
    ├── Non_Wrapping_Pulse_N0/
    │   └── recharge_swme_N0_constant_field_history.csv
    ├── Non_Wrapping_Pulse_N1/
    │   └── recharge_swme_N1_constant_field_history.csv
    └── Non_Wrapping_Pulse_N2/
        └── recharge_swme_N2_constant_field_history.csv

Optional hyperbolicity files:
    recharge_hyperbolicity_summary.csv
    recharge_hyperbolicity_history.csv

Outputs
-------
Figures are saved in:

    ROOT_DIR / "Non_Wrapping_Pulse_Comparison_Figures"

Generated figures:
    1. Non_Wrapping_Pulse_bulk_comparison.pdf
    2. Non_Wrapping_Pulse_spatial_profiles_comparison.pdf
    3. Non_Wrapping_Pulse_moment_statistics.pdf
    4. Non_Wrapping_Pulse_vertical_profiles_comparison.pdf
    5. Non_Wrapping_Pulse_hyperbolicity_summary.pdf, optional

Notes
-----
This script reads the CSV datasets only. It does not use the already generated
per-order PNG figures.

The test is assumed to use the constant rainfall/exfiltration or constant
rainfall/infiltration source model, hence the CSV stem uses 'constant' rather
than 'horton'.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')


# =============================================================================
# Absolute paths and settings
# =============================================================================

ROOT_DIR = Path("/home/anenin/Documents/Git/thesis/model/processing/Non_Wrapping_Pulse")

OUTPUT_DIR = ROOT_DIR / "Non_Wrapping_Pulse_Comparison_Figures"

ORDERS = (0, 1, 2)

# For your non-wrapping short-time test.
# Change if your final time is different.
SELECTED_TIMES = (0.0, 0.5, 1.0)

# Set True if you want the hyperbolicity summary figure.
INCLUDE_HYPERBOLICITY = False

# File stem used by the constant recharge test.
SOURCE_TAG = "constant"


# =============================================================================
# Shared loading utilities
# =============================================================================

def _nearest_time(df: pd.DataFrame, target_time: float) -> float:
    """
    Return the available simulation time closest to a requested target time.

    Parameters
    ----------
    df : pandas.DataFrame
        Field-history dataframe containing a 'time' column.
    target_time : float
        Requested physical time.

    Returns
    -------
    float
        Closest available time in the dataframe.
    """
    times = np.asarray(sorted(df["time"].unique()), dtype=float)

    if times.size == 0:
        raise ValueError("The dataframe contains no available time values.")

    return float(times[np.argmin(np.abs(times - target_time))])


def _order_folder_name(order: int) -> str:
    """
    Return the folder name for one moment order.

    Parameters
    ----------
    order : int
        Moment order N.

    Returns
    -------
    str
        Folder name containing the datasets for this order.
    """
    return f"Non_Wrapping_Pulse_N{order}"


def _get_order_history_path(root_dir: str | Path, order: int) -> Path:
    """
    Construct the expected field-history path for a given moment order.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root Non_Wrapping_Pulse directory.
    order : int
        Moment order N.

    Returns
    -------
    pathlib.Path
        Path to the corresponding field-history CSV.
    """
    root_dir = Path(root_dir)

    return (
        root_dir
        / _order_folder_name(order)
        / f"recharge_swme_N{order}_{SOURCE_TAG}_field_history.csv"
    )


def _get_hyperbolicity_summary_path(root_dir: str | Path, order: int) -> Path:
    """
    Construct the expected hyperbolicity-summary path.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root Non_Wrapping_Pulse directory.
    order : int
        Moment order N.

    Returns
    -------
    pathlib.Path
        Path to recharge_hyperbolicity_summary.csv.
    """
    root_dir = Path(root_dir)

    return (
        root_dir
        / _order_folder_name(order)
        / "recharge_hyperbolicity_summary.csv"
    )


def _load_order_history(root_dir: str | Path, order: int) -> pd.DataFrame:
    """
    Load one field-history CSV and attach a model-order column.

    Expected columns are at least:
        time, x, h, u_m

    For N >= 1, the file may also contain:
        a1

    For N >= 2, the file may also contain:
        a2

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root Non_Wrapping_Pulse result directory.
    order : int
        Moment order N.

    Returns
    -------
    pandas.DataFrame
        Loaded and lightly cleaned dataframe.
    """
    path = _get_order_history_path(root_dir, order)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find field-history file:\n{path}\n\n"
            f"Expected folder: {_order_folder_name(order)}"
        )

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    df["N"] = order

    required = {"time", "x", "h", "u_m"}
    missing = required.difference(df.columns)

    if missing:
        raise ValueError(
            f"File {path} is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    for col in ["time", "x", "h", "u_m", "a1", "a2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time", "x", "h", "u_m"]).copy()

    return df


def load_non_wrapping_pulse_histories(
    root_dir: str | Path,
    orders: tuple[int, ...] = ORDERS,
) -> dict[int, pd.DataFrame]:
    """
    Load all non-wrapping pulse field-history datasets.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root Non_Wrapping_Pulse result directory.
    orders : tuple of int
        Moment orders to load.

    Returns
    -------
    dict[int, pandas.DataFrame]
        Dictionary mapping N to its field-history dataframe.
    """
    histories: dict[int, pd.DataFrame] = {}

    for order in orders:
        histories[order] = _load_order_history(root_dir, order)

    return histories


def _save_figure(fig: plt.Figure, output_path: str | Path) -> None:
    """
    Save a matplotlib figure and close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    output_path : str or pathlib.Path
        Output figure path.

    Returns
    -------
    None
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")

def _as_flat_axes(axes):
    """
    Return axes as a flat numpy array.
    """
    return np.asarray(axes).ravel()


def _force_axis_tick_labels(ax):
    """
    Force both x- and y-axis tick labels to be visible.

    This is useful when matplotlib suppresses labels because axes are shared.
    """
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=True,
        labelleft=True,
    )


def _apply_common_xlim(axes, pad_fraction: float = 0.0):
    """
    Apply one common x-axis range to all axes and show x tick labels everywhere.
    """
    flat_axes = _as_flat_axes(axes)

    xmin = min(ax.get_xlim()[0] for ax in flat_axes)
    xmax = max(ax.get_xlim()[1] for ax in flat_axes)

    if pad_fraction > 0.0:
        width = xmax - xmin
        xmin -= pad_fraction * width
        xmax += pad_fraction * width

    for ax in flat_axes:
        ax.set_xlim(xmin, xmax)
        _force_axis_tick_labels(ax)


def _apply_common_ylim(axes, pad_fraction: float = 0.0):
    """
    Apply one common y-axis range to all axes and show y tick labels everywhere.
    """
    flat_axes = _as_flat_axes(axes)

    ymin = min(ax.get_ylim()[0] for ax in flat_axes)
    ymax = max(ax.get_ylim()[1] for ax in flat_axes)

    if pad_fraction > 0.0:
        height = ymax - ymin
        ymin -= pad_fraction * height
        ymax += pad_fraction * height

    for ax in flat_axes:
        ax.set_ylim(ymin, ymax)
        _force_axis_tick_labels(ax)

def _show_all_tick_labels(axes) -> None:
    """
    Force x- and y-axis tick labels to be visible on all subplots,
    even when sharex/sharey is used.
    """
    for ax in np.asarray(axes).ravel():
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=True,
            labelleft=True,
        )

# =============================================================================
# Plotting Utilities
# =============================================================================
def plot_model_curve(ax, x, y, order, label):
    """
    Plot one model-order curve with distinguishable line style.
    
    Colours are left to plt.style.use('tableau-colorblind10'). 
    Only linestyle, width, market, and z-order are controlled here.
    """
    if order == 0:
        color = "#0072B2"  # blue
        linestyle = "-"
        linewidth = 3.0
        zorder = 1
    elif order == 1:
        color = "#E69F00"  # orange
        linestyle = "-."
        linewidth = 2.0
        zorder = 2
    elif order == 2:
        color = "#000000"  # black
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

# =============================================================================
# Figure 1: bulk response comparison
# =============================================================================

def compute_bulk_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bulk spatial statistics from a field-history dataframe.

    Computes:
        mean(h),
        min(h),
        max(h),
        max(h)-min(h),
        mean(u_m),
        mean(q), where q = h*u_m.

    Parameters
    ----------
    df : pandas.DataFrame
        Field-history dataframe containing at least time, x, h, u_m.

    Returns
    -------
    pandas.DataFrame
        Time-indexed bulk statistics.
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
    )

    stats["h_range"] = stats["h_max"] - stats["h_min"]

    return stats

def plot_single_bulk_quantity(histories, quantity, ylabel, output_path):
    """
    Plot one bulk quantity comparison for all model orders.

    Parameters
    ----------
    histories : dict[int, pandas.DataFrame]
        Dictionary mapping N to its field-history dataframe.
    quantity : str
        Column name of the bulk quantity to plot.
    ylabel : str
        Y-axis label for the plot.
    output_path : str or pathlib.Path
        Output figure path.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for order, df in histories.items():
        stats = compute_bulk_statistics(df)
        label = f"N={order}"

        plot_model_curve(
            ax=ax,
            x=stats["time"],
            y=stats[quantity],
            order=order,
            label=label,
        )

    # Keep x-axis on every separate figure
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    _save_figure(fig, output_path)

def plot_non_wrapping_pulse_bulk_comparison(
    root_dir: str | Path,
    output_path: str | Path,
    orders: tuple[int, ...] = ORDERS,
) -> None:
    """
    Plot bulk response comparison for N=0, N=1, and N=2.

    Panels:
        1. Mean water height over time.
        2. Height spread max(h)-min(h) over time.
        3. Mean velocity over time.
        4. Mean discharge q = h*u_m over time.
    """
    histories = load_non_wrapping_pulse_histories(root_dir, orders)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

    ax_h_mean = axes[0, 0]
    ax_h_range = axes[0, 1]
    ax_u_mean = axes[1, 0]
    ax_q_mean = axes[1, 1]

    for order, df in histories.items():
        stats = compute_bulk_statistics(df)
        label = f"N={order}"

        plot_model_curve(
            ax = ax_h_mean,
            x = stats["time"],
            y = stats["h_mean"],
            order = order,
            label = f"N={order}"
        )
        plot_model_curve(
            ax = ax_h_range,
            x = stats["time"],
            y = stats["h_range"],
            order = order,
            label = f"N={order}"
        )
        plot_model_curve(
            ax = ax_u_mean,
            x = stats["time"],
            y = stats["u_mean"],
            order = order,
            label = f"N={order}"
        )
        plot_model_curve(
            ax = ax_q_mean,
            x = stats["time"],
            y = stats["q_mean"],
            order = order,
            label = f"N={order}"
        )

    ax_h_mean.set_ylabel(r"$\overline{h}$")
    ax_h_mean.grid(True, alpha=0.3)
    ax_h_mean.legend()

    ax_h_range.set_ylabel(r"$\max_x h - \min_x h$")
    ax_h_range.grid(True, alpha=0.3)
    ax_h_range.legend()

    ax_u_mean.set_xlabel("Time")
    ax_u_mean.set_ylabel(r"$\overline{u_m}$")
    ax_u_mean.grid(True, alpha=0.3)
    ax_u_mean.legend()

    ax_q_mean.set_xlabel("Time")
    ax_q_mean.set_ylabel(r"$\overline{q}=\overline{h u_m}$")
    ax_q_mean.grid(True, alpha=0.3)
    ax_q_mean.legend()

    _apply_common_xlim(axes)

    for ax in _as_flat_axes(axes):
        ax.set_xlabel("Time [s]")

    fig.tight_layout()
    _save_figure(fig, output_path)


# =============================================================================
# Figure 2: spatial profiles at selected times
# =============================================================================

def plot_non_wrapping_pulse_spatial_profiles(
    root_dir: str | Path,
    selected_times: tuple[float, ...],
    output_path_h: str | Path,
    output_path_u: str | Path,
    orders: tuple[int, ...] = ORDERS,
) -> None:
    """
    Plot spatial profiles of h(x) and u_m(x) for N=0, N=1, and N=2.

    This creates two separate figures:

        1. h(x) profiles at selected times.
        2. u_m(x) profiles at selected times.

    Each figure contains one row of subfigures, one column per selected time.
    The y-axis is shared across all subfigures in each figure.
    """
    histories = load_non_wrapping_pulse_histories(root_dir, orders)

    reference_df = histories[orders[0]]
    available_times = [_nearest_time(reference_df, t) for t in selected_times]

    ncols = len(available_times)

    # -------------------------------------------------------------------------
    # Figure 1: h(x) profiles
    # -------------------------------------------------------------------------
    fig_h, axes_h = plt.subplots(
        1,
        ncols,
        figsize=(5.0 * ncols, 3.8),
        sharex=False,
        sharey=True,
    )

    if ncols == 1:
        axes_h = np.asarray([axes_h])

    for col, time_value in enumerate(available_times):
        ax = axes_h[col]

        for order, df in histories.items():
            local_time = _nearest_time(df, time_value)
            snap = df[np.isclose(df["time"], local_time)].copy()
            snap = snap.sort_values("x")

            plot_model_curve(
                ax=ax,
                x=snap["x"],
                y=snap["h"],
                order=order,
                label=f"N={order}",
            )

        ax.set_title(f"t = {time_value:.3f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$h$")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    axes_h[0].set_ylabel(r"$h$")
    
    _show_all_tick_labels(axes_h)

    fig_h.tight_layout()
    _save_figure(fig_h, output_path_h)

    # -------------------------------------------------------------------------
    # Figure 2: u_m(x) profiles
    # -------------------------------------------------------------------------
    fig_u, axes_u = plt.subplots(
        1,
        ncols,
        figsize=(5.0 * ncols, 3.8),
        sharex=False,
        sharey=True,
    )

    if ncols == 1:
        axes_u = np.asarray([axes_u])

    for col, time_value in enumerate(available_times):
        ax = axes_u[col]

        for order, df in histories.items():
            local_time = _nearest_time(df, time_value)
            snap = df[np.isclose(df["time"], local_time)].copy()
            snap = snap.sort_values("x")

            plot_model_curve(
                ax=ax,
                x=snap["x"],
                y=snap["u_m"],
                order=order,
                label=f"N={order}",
            )

        ax.set_title(f"t = {time_value:.3f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u_m$")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    axes_u[0].set_ylabel(r"$u_m$")
    
    _show_all_tick_labels(axes_u)

    fig_u.tight_layout()
    _save_figure(fig_u, output_path_u)

# =============================================================================
# Figure 2b: spatial moment profiles at selected times
# =============================================================================

def _moment_profile_values(
    snap: pd.DataFrame,
    moment_name: str,
) -> np.ndarray:
    """
    Return spatial moment values for one snapshot.

    If the requested moment column is absent, return zeros. This is useful for:
        N=0: a1 = 0, a2 = 0
        N=1: a2 = 0
    """
    if moment_name in snap.columns:
        return snap[moment_name].to_numpy(dtype=float)

    return np.zeros(len(snap), dtype=float)


def plot_non_wrapping_pulse_spatial_moment_profiles(
    root_dir: str | Path,
    selected_times: tuple[float, ...],
    output_path_a1: str | Path,
    output_path_a2: str | Path,
    orders: tuple[int, ...] = ORDERS,
) -> None:
    """
    Plot spatial profiles of a1(x) and a2(x) for N=0, N=1, and N=2.

    This creates two separate figures:

        1. a1(x) profiles at selected times.
        2. a2(x) profiles at selected times.

    For models where a moment is not present, the corresponding profile is
    plotted as zero. This makes the hierarchy explicit:
        N=0: a1 = 0, a2 = 0
        N=1: a2 = 0
        N=2: a1 and a2 are both available.
    """
    histories = load_non_wrapping_pulse_histories(root_dir, orders)

    reference_df = histories[orders[0]]
    available_times = [_nearest_time(reference_df, t) for t in selected_times]

    ncols = len(available_times)

    # -------------------------------------------------------------------------
    # Figure 1: a1(x) profiles
    # -------------------------------------------------------------------------
    fig_a1, axes_a1 = plt.subplots(
        1,
        ncols,
        figsize=(5.0 * ncols, 3.8),
        sharex=False,
        sharey=True,
    )

    if ncols == 1:
        axes_a1 = np.asarray([axes_a1])

    for col, time_value in enumerate(available_times):
        ax = axes_a1[col]

        for order in (1, 2):
            if order not in histories:
                continue

            df = histories[order]
            local_time = _nearest_time(df, time_value)
            snap = df[np.isclose(df["time"], local_time)].copy()
            snap = snap.sort_values("x")

            y = _moment_profile_values(snap, "a1")

            plot_model_curve(
                ax=ax,
                x=snap["x"],
                y=y,
                order=order,
                label=f"N={order}",
            )

        ax.axhline(0.0, color="0.65", linewidth=0.8, zorder=0)
        ax.set_title(f"t = {time_value:.3f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\alpha_1$")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    _show_all_tick_labels(axes_a1)

    fig_a1.tight_layout()
    _save_figure(fig_a1, output_path_a1)

    # -------------------------------------------------------------------------
    # Figure 2: a2(x) profiles
    # -------------------------------------------------------------------------
    fig_a2, axes_a2 = plt.subplots(
        1,
        ncols,
        figsize=(5.0 * ncols, 3.8),
        sharex=False,
        sharey=True,
    )

    if ncols == 1:
        axes_a2 = np.asarray([axes_a2])

    for col, time_value in enumerate(available_times):
        ax = axes_a2[col]

        for order in (2,):
            if order not in histories:
                continue

            df = histories[order]
            local_time = _nearest_time(df, time_value)
            snap = df[np.isclose(df["time"], local_time)].copy()
            snap = snap.sort_values("x")

            y = _moment_profile_values(snap, "a2")

            plot_model_curve(
                ax=ax,
                x=snap["x"],
                y=y,
                order=order,
                label=f"N={order}",
            )

        ax.axhline(0.0, color="0.65", linewidth=0.8, zorder=0)
        ax.set_title(f"t = {time_value:.3f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\alpha_2$")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    _show_all_tick_labels(axes_a2)

    fig_a2.tight_layout()
    _save_figure(fig_a2, output_path_a2)

# =============================================================================
# Figure 3: moment dynamics
# =============================================================================

def compute_moment_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Euclidean and maximum absolute moment amplitudes over time.

    Computes:
        ||alpha_1||_2,
        max_x |alpha_1|,
        ||alpha_2||_2,
        max_x |alpha_2|.

    If a moment column is absent, the corresponding quantity is set to zero.
    This makes the model hierarchy explicit:
        N=0: alpha_1 = 0, alpha_2 = 0
        N=1: alpha_2 = 0
        N=2: alpha_1 and alpha_2 available.
    """
    rows: list[dict[str, float]] = []

    for time_value, group in df.groupby("time"):
        row: dict[str, float] = {"time": float(time_value)}

        if "a1" in group.columns:
            a1 = group["a1"].to_numpy(dtype=float)
            row["a1_l2"] = float(np.linalg.norm(a1, ord=2))
            row["a1_max_abs"] = float(np.max(np.abs(a1)))
        else:
            row["a1_l2"] = 0.0
            row["a1_max_abs"] = 0.0

        if "a2" in group.columns:
            a2 = group["a2"].to_numpy(dtype=float)
            row["a2_l2"] = float(np.linalg.norm(a2, ord=2))
            row["a2_max_abs"] = float(np.max(np.abs(a2)))
        else:
            row["a2_l2"] = 0.0
            row["a2_max_abs"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows).sort_values("time")

def plot_single_moment_quantity(
    histories: dict[int, pd.DataFrame],
    quantity: str,
    ylabel: str,
    output_path: str | Path,
    relevant_orders: tuple[int, ...],
) -> None:
    """
    Plot one moment statistic as a standalone figure.

    Only relevant model orders are plotted:
        alpha_1: N=1 and N=2
        alpha_2: N=2 only
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for order in relevant_orders:
        if order not in histories:
            continue

        df = histories[order]
        stats = compute_moment_statistics(df)

        plot_model_curve(
            ax=ax,
            x=stats["time"],
            y=stats[quantity],
            order=order,
            label=f"N={order}",
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    _save_figure(fig, output_path)

def plot_non_wrapping_pulse_moment_statistics(
    root_dir: str | Path,
    output_dir: str | Path,
    orders: tuple[int, ...] = ORDERS,
) -> None:
    """
    Plot moment-amplitude statistics for N=0, N=1, and N=2.

    Instead of one 2x2 figure, this creates four separate figures:
        1. Euclidean norm of alpha_1.
        2. Maximum absolute alpha_1.
        3. Euclidean norm of alpha_2.
        4. Maximum absolute alpha_2.
    """
    histories = load_non_wrapping_pulse_histories(root_dir, orders)
    output_dir = Path(output_dir)

    plot_single_moment_quantity(
    histories=histories,
    quantity="a1_l2",
    ylabel=r"$\|\alpha_1\|_2$",
    output_path=output_dir / "Non_Wrapping_Pulse_a1_l2_norm_comparison.pdf",
    relevant_orders=(1, 2),
    )

    plot_single_moment_quantity(
    histories=histories,
    quantity="a1_max_abs",
    ylabel=r"$\max_x |\alpha_1|$",
    output_path=output_dir / "Non_Wrapping_Pulse_a1_max_abs_comparison.pdf",
    relevant_orders=(1, 2),
    )

    plot_single_moment_quantity(
        histories=histories,
        quantity="a2_l2",
        ylabel=r"$\|\alpha_2\|_2$",
        output_path=output_dir / "Non_Wrapping_Pulse_a2_l2_norm_comparison.pdf",
        relevant_orders=(2,),
    )

    plot_single_moment_quantity(
        histories=histories,
        quantity="a2_max_abs",
        ylabel=r"$\max_x |\alpha_2|$",
        output_path=output_dir / "Non_Wrapping_Pulse_a2_max_abs_comparison.pdf",
        relevant_orders=(2,),
    )

# =============================================================================
# Figure 4: reconstructed vertical velocity profiles
# =============================================================================

def _phi_1(zeta: np.ndarray) -> np.ndarray:
    """
    First shifted Legendre basis function.

    phi_1(zeta) = 1 - 2*zeta.
    """
    return 1.0 - 2.0 * zeta


def _phi_2(zeta: np.ndarray) -> np.ndarray:
    """
    Second shifted Legendre basis function.

    phi_2(zeta) = 1 - 6*zeta + 6*zeta^2.
    """
    return 1.0 - 6.0 * zeta + 6.0 * zeta**2


def reconstruct_vertical_velocity(row: pd.Series, zeta: np.ndarray) -> np.ndarray:
    """
    Reconstruct u(zeta) from one dataframe row.

    N=0:
        u(zeta) = u_m

    N=1:
        u(zeta) = u_m + alpha_1 phi_1(zeta)

    N=2:
        u(zeta) = u_m + alpha_1 phi_1(zeta) + alpha_2 phi_2(zeta)
    """
    profile = np.full_like(zeta, float(row["u_m"]), dtype=float)

    if "a1" in row.index and pd.notna(row["a1"]):
        profile += float(row["a1"]) * _phi_1(zeta)

    if "a2" in row.index and pd.notna(row["a2"]):
        profile += float(row["a2"]) * _phi_2(zeta)

    return profile


def _select_profile_row(
    df: pd.DataFrame,
    time_value: float,
    selection: str = "peak_velocity",
    fixed_x: float = 0.5,
) -> pd.Series:
    """
    Select one spatial row for vertical profile reconstruction.

    selection='peak_velocity':
        use the cell with maximum u_m.

    selection='fixed_x':
        use the cell closest to fixed_x.
    """
    local_time = _nearest_time(df, time_value)
    snap = df[np.isclose(df["time"], local_time)].copy()

    if snap.empty:
        raise ValueError(f"No snapshot found near time {time_value}.")

    if selection == "peak_velocity":
        idx = snap["u_m"].idxmax()
        return snap.loc[idx]

    if selection == "fixed_x":
        distances = np.abs(snap["x"].to_numpy(dtype=float) - fixed_x)
        idx = snap.index[int(np.argmin(distances))]
        return snap.loc[idx]

    raise ValueError(
        "Unknown selection rule. Use selection='peak_velocity' or selection='fixed_x'."
    )


def plot_non_wrapping_pulse_vertical_profiles(
    root_dir: str | Path,
    selected_times: tuple[float, ...],
    output_path: str | Path,
    orders: tuple[int, ...] = ORDERS,
    selection: str = "peak_velocity",
    fixed_x: float = 0.5,
    n_zeta: int = 200,
) -> None:
    """
    Plot reconstructed vertical velocity profiles for N=0, N=1, and N=2.

    Each subplot corresponds to one selected time. In each subplot, the
    reconstructed profile u(zeta) is overlaid for all selected model orders.
    """
    histories = load_non_wrapping_pulse_histories(root_dir, orders)
    reference_df = histories[orders[0]]
    available_times = [_nearest_time(reference_df, t) for t in selected_times]

    zeta = np.linspace(0.0, 1.0, n_zeta)

    ncols = len(available_times)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=True)

    if ncols == 1:
        axes = np.asarray([axes])

    for col, time_value in enumerate(available_times):
        ax = axes[col]

        for order, df in histories.items():
            row = _select_profile_row(
                df,
                time_value=time_value,
                selection=selection,
                fixed_x=fixed_x,
            )

            profile = reconstruct_vertical_velocity(row, zeta)
            label = f"N={order}, x={row['x']:.3f}"

            plot_model_curve(
                ax=ax,
                x=profile,
                y=zeta,
                order=order,
                label=label,
            )

        ax.set_title(f"t = {time_value:.3f}")
        ax.set_xlabel(r"$u(\zeta)$")
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    _apply_common_ylim(axes)

    for ax in _as_flat_axes(axes):
        ax.set_xlabel(r"$u(\zeta)$")
        ax.set_ylabel(r"$\zeta$")
    
    fig.tight_layout()

    _save_figure(fig, output_path)


# =============================================================================
# Optional: hyperbolicity summary
# =============================================================================

def plot_hyperbolicity_summary_comparison(
    root_dir: str | Path,
    output_path: str | Path,
    orders: tuple[int, ...] = ORDERS,
) -> None:
    """
    Plot hyperbolicity summary diagnostics for all available orders.

    This is optional and only useful if store_hyperbolicity=True was active.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    any_loaded = False

    for order in orders:
        path = _get_hyperbolicity_summary_path(root_dir, order)

        if not path.exists():
            continue

        df = pd.read_csv(path)
        df.columns = [str(col).strip() for col in df.columns]

        if df.empty:
            continue

        label = f"N={order}"

        if "num_nonhyperbolic_cells" in df.columns:
            axes[0].plot(df["time"], df["num_nonhyperbolic_cells"], label=label)

        if "max_abs_imag_eig" in df.columns:
            axes[1].plot(df["time"], df["max_abs_imag_eig"], label=label)

        any_loaded = True

    if not any_loaded:
        print("[skip] No hyperbolicity summary files were found.")
        plt.close(fig)
        return

    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Time")
    axes[1].set_ylabel(r"$\max |\operatorname{Im}(\lambda)|$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()

    _save_figure(fig, output_path)


# =============================================================================
# Wrapper
# =============================================================================

def generate_non_wrapping_pulse_model_comparison_figures(
    root_dir: str | Path = ROOT_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    selected_times: tuple[float, ...] = SELECTED_TIMES,
    include_hyperbolicity: bool = INCLUDE_HYPERBOLICITY,
) -> None:
    """
    Generate all model-comparison figures for the non-wrapping pulse test.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root Non_Wrapping_Pulse directory.
    output_dir : str or pathlib.Path
        Output directory for generated comparison figures.
    selected_times : tuple of float
        Times used in spatial/profile comparison figures.
    include_hyperbolicity : bool
        Whether to create the hyperbolicity summary comparison if data exist.

    Returns
    -------
    None
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)

    if not root_dir.exists():
        raise FileNotFoundError(
            f"ROOT_DIR does not exist:\n{root_dir}\n"
            "Edit ROOT_DIR at the top of this script."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[root] {root_dir.resolve()}")
    print(f"[out ] {output_dir.resolve()}")

    #plot_non_wrapping_pulse_bulk_comparison(
    #    root_dir=root_dir,
    #    output_path=output_dir / "Non_Wrapping_Pulse_bulk_comparison.pdf",
    #)
    plot_single_bulk_quantity(
        histories=load_non_wrapping_pulse_histories(root_dir),
        quantity="h_mean",
        ylabel=r"$\overline{h}$",
        output_path=output_dir / "Non_Wrapping_Pulse_h_mean_comparison.pdf",
    )
    plot_single_bulk_quantity(
        histories=load_non_wrapping_pulse_histories(root_dir),
        quantity="h_range",
        ylabel=r"$\max_x h - \min_x h$",
        output_path=output_dir / "Non_Wrapping_Pulse_h_range_comparison.pdf",
    )
    plot_single_bulk_quantity(
        histories=load_non_wrapping_pulse_histories(root_dir),
        quantity="u_mean",
        ylabel=r"$\overline{u_m}$ [m/s]",
        output_path=output_dir / "Non_Wrapping_Pulse_u_mean_comparison.pdf",
    )
    plot_single_bulk_quantity(
        histories=load_non_wrapping_pulse_histories(root_dir),
        quantity="q_mean",
        ylabel=r"$\overline{q}=\overline{h u_m}$",
        output_path=output_dir / "Non_Wrapping_Pulse_q_mean_comparison.pdf",
    )

    plot_non_wrapping_pulse_spatial_profiles(
    root_dir=root_dir,
    selected_times=selected_times,
    output_path_h=output_dir / "Non_Wrapping_Pulse_spatial_height_profiles_comparison.pdf",
    output_path_u=output_dir / "Non_Wrapping_Pulse_spatial_velocity_profiles_comparison.pdf",
)
    
    plot_non_wrapping_pulse_spatial_moment_profiles(
        root_dir=root_dir,
        selected_times=selected_times,
        output_path_a1=output_dir / "Non_Wrapping_Pulse_spatial_a1_profiles_comparison.pdf",
        output_path_a2=output_dir / "Non_Wrapping_Pulse_spatial_a2_profiles_comparison.pdf",
    )

    plot_non_wrapping_pulse_moment_statistics(
        root_dir=root_dir,
        output_dir=output_dir,
    )

    plot_non_wrapping_pulse_vertical_profiles(
        root_dir=root_dir,
        selected_times=selected_times,
        output_path=output_dir / "Non_Wrapping_Pulse_vertical_profiles_comparison.pdf",
        selection="peak_velocity",
    )

    if include_hyperbolicity:
        plot_hyperbolicity_summary_comparison(
            root_dir=root_dir,
            output_path=output_dir / "Non_Wrapping_Pulse_hyperbolicity_summary.pdf",
        )

    print("[done] Non-wrapping pulse comparison figures generated.")


if __name__ == "__main__":
    generate_non_wrapping_pulse_model_comparison_figures()