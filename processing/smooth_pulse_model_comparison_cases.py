"""
Smooth-pulse Horton recharge model-comparison plots for Mild and Aggressive cases.

No argparse version.
Edit ROOT_DIR and CASES below if the folder location or selected cases change.

Run from anywhere with:

    python smooth_pulse_model_comparison_cases.py

Expected folder structure
-------------------------
ROOT_DIR should point to the Smooth_Pulse directory:

    /home/anenin/Documents/Git/thesis/model/processing/Smooth_Pulse

and this folder should contain case-specific folders such as:

    Smooth_Pulse_N0_Mild/
    Smooth_Pulse_N1_Mild/
    Smooth_Pulse_N2_Mild/

    Smooth_Pulse_N0_Aggressive/
    Smooth_Pulse_N1_Aggressive/
    Smooth_Pulse_N2_Aggressive/

Each order folder should contain:

    recharge_swme_N{order}_horton_field_history.csv
    recharge_swme_N{order}_horton_final.csv
    recharge_swme_N{order}_horton_summary.csv

Optional hyperbolicity files:

    recharge_hyperbolicity_summary.csv
    recharge_hyperbolicity_history.csv

Outputs
-------
For each case, figures are saved in:

    ROOT_DIR / f"Smooth_Pulse_Comparison_Figures_{case}"

Generated figures:
    1. Smooth_Pulse_{case}_bulk_comparison.pdf
    2. Smooth_Pulse_{case}_spatial_profiles_comparison.pdf
    3. Smooth_Pulse_{case}_moment_statistics.pdf
    4. Smooth_Pulse_{case}_vertical_profiles_comparison.pdf
    5. Smooth_Pulse_{case}_hyperbolicity_summary.pdf, optional

Notes
-----
This script reads the CSV datasets only. It does not use the already generated
per-order PNG figures.
"""

from __future__ import annotations

from cProfile import label
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import col


# =============================================================================
# Absolute paths and case settings
# =============================================================================

ROOT_DIR = Path("/home/anenin/Documents/Git/thesis/model/processing/Smooth_Pulse/")

# Use ("Mild",) to generate only the mild comparison.
# Use ("Aggressive",) to generate only the aggressive comparison.
# Use ("Mild", "Aggressive") to generate both.
CASES = ("Mild", "Aggressive")

ORDERS = (0, 1, 2)

SELECTED_TIMES = (0.0, 0.9, 1.8)

INCLUDE_HYPERBOLICITY = False

# =============================================================================
# Plot styling
# =============================================================================

plt.style.use("tableau-colorblind10")


def plot_model_curve(ax, x, y, order, label):
    """
    Plot one model-order curve with the thesis styling.

    N=0: blue, solid, thick.
    N=1: orange, dash-dot.
    N=2: black, dotted.
    """
    if order == 0:
        color = "#0072B2"      # blue
        linestyle = "-"
        linewidth = 3.0
        zorder = 1

    elif order == 1:
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


def force_axis_tick_labels(axes) -> None:
    """
    Force tick labels to appear on all subplots, even when sharex/sharey is used.
    """
    for ax in np.asarray(axes).ravel():
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=True,
            labelleft=True,
        )


# =============================================================================
# Shared loading utilities
# =============================================================================

def _case_folder_name(order: int, case: str | None = None) -> str:
    """
    Return the folder name for one moment order and one case.

    Parameters
    ----------
    order : int
        Moment order N.
    case : str or None
        Case suffix. Typical values are 'Mild' and 'Aggressive'.
        If None or empty, the legacy folder name Smooth_Pulse_N{order} is used.

    Returns
    -------
    str
        Folder name containing the datasets for the requested order/case.
    """
    if case is None or str(case).strip() == "":
        return f"Smooth_Pulse_N{order}"

    return f"Smooth_Pulse_N{order}_{case}"


def _comparison_folder_name(case: str | None = None) -> str:
    """
    Return the comparison-output folder name for one case.

    Parameters
    ----------
    case : str or None
        Case suffix. Typical values are 'Mild' and 'Aggressive'.
        If None or empty, the legacy folder Smooth_Pulse_Comparison_Figures is used.

    Returns
    -------
    str
        Output folder name for comparison figures.
    """
    if case is None or str(case).strip() == "":
        return "Smooth_Pulse_Comparison_Figures"

    return f"Smooth_Pulse_Comparison_Figures_{case}"


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


def _get_order_history_path(
    root_dir: str | Path,
    order: int,
    case: str | None = None,
) -> Path:
    """
    Construct the expected field-history path for a given moment order and case.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root Smooth_Pulse directory.
    order : int
        Moment order N.
    case : str or None
        Case suffix, e.g. 'Mild' or 'Aggressive'.

    Returns
    -------
    pathlib.Path
        Path to the corresponding field-history CSV.
    """
    root_dir = Path(root_dir)

    return (
        root_dir
        / _case_folder_name(order, case)
        / f"recharge_swme_N{order}_horton_field_history.csv"
    )


def _get_hyperbolicity_summary_path(
    root_dir: str | Path,
    order: int,
    case: str | None = None,
) -> Path:
    """
    Construct the expected hyperbolicity-summary path.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root Smooth_Pulse directory.
    order : int
        Moment order N.
    case : str or None
        Case suffix, e.g. 'Mild' or 'Aggressive'.

    Returns
    -------
    pathlib.Path
        Path to recharge_hyperbolicity_summary.csv.
    """
    root_dir = Path(root_dir)

    return (
        root_dir
        / _case_folder_name(order, case)
        / "recharge_hyperbolicity_summary.csv"
    )


def _load_order_history(
    root_dir: str | Path,
    order: int,
    case: str | None = None,
) -> pd.DataFrame:
    """
    Load one field-history CSV and attach model-order and case columns.

    Expected columns are at least:
        time, x, h, u_m

    For N >= 1, the file may also contain:
        a1

    For N >= 2, the file may also contain:
        a2

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root Smooth_Pulse result directory.
    order : int
        Moment order N.
    case : str or None
        Case suffix, e.g. 'Mild' or 'Aggressive'.

    Returns
    -------
    pandas.DataFrame
        Loaded and lightly cleaned dataframe.
    """
    path = _get_order_history_path(root_dir, order, case)

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find field-history file:\n{path}\n\n"
            f"Expected folder: {_case_folder_name(order, case)}"
        )

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]
    df["N"] = order
    df["case"] = "" if case is None else str(case)

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


def load_smooth_pulse_histories(
    root_dir: str | Path,
    orders: tuple[int, ...] = ORDERS,
    case: str | None = None,
) -> dict[int, pd.DataFrame]:
    """
    Load all smooth-pulse field-history datasets for one case.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root Smooth_Pulse result directory.
    orders : tuple of int
        Moment orders to load.
    case : str or None
        Case suffix, e.g. 'Mild' or 'Aggressive'.

    Returns
    -------
    dict[int, pandas.DataFrame]
        Dictionary mapping N to its field-history dataframe.
    """
    histories: dict[int, pd.DataFrame] = {}

    for order in orders:
        histories[order] = _load_order_history(root_dir, order, case)

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

def plot_single_bulk_quantity(
    histories: dict[int, pd.DataFrame],
    quantity: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """
    Plot one bulk quantity as a standalone figure.
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

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    _save_figure(fig, output_path)

def plot_smooth_pulse_bulk_comparison(
    root_dir: str | Path,
    output_dir: str | Path,
    case: str | None = None,
    orders: tuple[int, ...] = ORDERS,
) -> None:
    """
    Plot bulk response comparison for N=0, N=1, and N=2.

    Creates four separate figures:
        1. Mean water height.
        2. Height spread.
        3. Mean velocity.
        4. Mean discharge.
    """
    histories = load_smooth_pulse_histories(root_dir, orders, case)
    output_dir = Path(output_dir)

    prefix = "Smooth_Pulse" if case is None else f"Smooth_Pulse_{case}"

    plot_single_bulk_quantity(
        histories=histories,
        quantity="h_mean",
        ylabel=r"$\overline{h}$",
        output_path=output_dir / f"{prefix}_bulk_mean_height.pdf",
    )

    plot_single_bulk_quantity(
        histories=histories,
        quantity="h_range",
        ylabel=r"$\max_x h - \min_x h$",
        output_path=output_dir / f"{prefix}_bulk_height_spread.pdf",
    )

    plot_single_bulk_quantity(
        histories=histories,
        quantity="u_mean",
        ylabel=r"$\overline{u_m}$",
        output_path=output_dir / f"{prefix}_bulk_mean_velocity.pdf",
    )

    plot_single_bulk_quantity(
        histories=histories,
        quantity="q_mean",
        ylabel=r"$\overline{q}=\overline{h u_m}$",
        output_path=output_dir / f"{prefix}_bulk_mean_discharge.pdf",
    )

# =============================================================================
# Figure 2: spatial profiles at selected times
# =============================================================================

def plot_smooth_pulse_spatial_profiles(
    root_dir: str | Path,
    selected_times: tuple[float, ...],
    output_path_h: str | Path,
    output_path_u: str | Path,
    case: str | None = None,
    orders: tuple[int, ...] = ORDERS,
) -> None:
    """
    Plot spatial profiles of h(x) and u_m(x) for N=0, N=1, and N=2.

    Creates two separate figures:
        1. h(x) at selected times.
        2. u_m(x) at selected times.
    """
    histories = load_smooth_pulse_histories(root_dir, orders, case)

    reference_df = histories[orders[0]]
    available_times = [_nearest_time(reference_df, t) for t in selected_times]

    ncols = len(available_times)

    # -------------------------------------------------------------------------
    # Figure 1: h(x)
    # -------------------------------------------------------------------------
    fig_h, axes_h = plt.subplots(
        1,
        ncols,
        figsize=(5.0 * ncols, 3.8),
        sharex=True,
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

        ax.set_title(f"t = {selected_times[col]:.1f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$h$")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    force_axis_tick_labels(axes_h)
    fig_h.tight_layout()
    _save_figure(fig_h, output_path_h)

    # -------------------------------------------------------------------------
    # Figure 2: u_m(x)
    # -------------------------------------------------------------------------
    fig_u, axes_u = plt.subplots(
        1,
        ncols,
        figsize=(5.0 * ncols, 3.8),
        sharex=True,
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

        ax.set_title(f"t = {selected_times[col]:.1f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u_m$")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    force_axis_tick_labels(axes_u)
    fig_u.tight_layout()
    _save_figure(fig_u, output_path_u)

# =============================================================================
# Figure 3: moment dynamics
# =============================================================================

def compute_moment_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute L2 norm and maximum absolute moment amplitudes over time.

    This avoids cancellation problems that occur when plotting only the
    spatial mean of alpha_1 or alpha_2.

    Parameters
    ----------
    df : pandas.DataFrame
        Field-history dataframe.

    Returns
    -------
    pandas.DataFrame
        Time-dependent moment statistics.
    """
    rows: list[dict[str, float]] = []

    for time_value, group in df.groupby("time"):
        row: dict[str, float] = {"time": float(time_value)}

        if "a1" in group.columns:
            a1 = group["a1"].to_numpy(dtype=float)
            row["a1_l2"] = float(np.sqrt(np.sum(a1**2)))
            row["a1_max_abs"] = float(np.max(np.abs(a1)))

        if "a2" in group.columns:
            a2 = group["a2"].to_numpy(dtype=float)
            row["a2_l2"] = float(np.sqrt(np.sum(a2**2)))
            row["a2_max_abs"] = float(np.max(np.abs(a2)))

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

    Relevant model orders:
        alpha_1: N=1 and N=2
        alpha_2: N=2 only
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for order in relevant_orders:
        if order not in histories:
            continue

        df = histories[order]
        stats = compute_moment_statistics(df)

        if quantity not in stats.columns:
            continue

        plot_model_curve(
            ax=ax,
            x=stats["time"],
            y=stats[quantity],
            order=order,
            label=f"N={order}",
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    _save_figure(fig, output_path)

def plot_smooth_pulse_moment_statistics(
    root_dir: str | Path,
    output_dir: str | Path,
    case: str | None = None,
) -> None:
    """
    Plot moment-amplitude statistics as separate figures.

    Creates:
        1. L2 norm alpha_1.
        2. Max absolute alpha_1.
        3. L2 norm alpha_2.
        4. Max absolute alpha_2.
    """
    histories = load_smooth_pulse_histories(root_dir, orders=(1, 2), case=case)
    output_dir = Path(output_dir)

    prefix = "Smooth_Pulse" if case is None else f"Smooth_Pulse_{case}"

    plot_single_moment_quantity(
        histories=histories,
        quantity="a1_l2",
        ylabel=r"$\|\alpha_1\|_2$",
        output_path=output_dir / f"{prefix}_a1_l2_comparison.pdf",
        relevant_orders=(1, 2),
    )

    plot_single_moment_quantity(
        histories=histories,
        quantity="a1_max_abs",
        ylabel=r"$\max_x |\alpha_1|$",
        output_path=output_dir / f"{prefix}_a1_max_abs_comparison.pdf",
        relevant_orders=(1, 2),
    )

    plot_single_moment_quantity(
        histories=histories,
        quantity="a2_l2",
        ylabel=r"$\|\alpha_2\|_2$",
        output_path=output_dir / f"{prefix}_a2_l2_comparison.pdf",
        relevant_orders=(2,),
    )

    plot_single_moment_quantity(
        histories=histories,
        quantity="a2_max_abs",
        ylabel=r"$\max_x |\alpha_2|$",
        output_path=output_dir / f"{prefix}_a2_max_abs_comparison.pdf",
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
    selection: str = "fixed_x",
    fixed_x: float = 0.50,
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


def plot_smooth_pulse_vertical_profiles(
    root_dir: str | Path,
    selected_times: tuple[float, ...],
    output_path: str | Path,
    case: str | None = None,
    orders: tuple[int, ...] = ORDERS,
    selection: str = "fixed_x",
    fixed_x: float = 0.50,
    n_zeta: int = 200,
) -> None:
    """
    Plot reconstructed vertical velocity profiles for N=0, N=1, and N=2.

    Each subplot corresponds to one selected time. In each subplot, the
    reconstructed profile u(zeta) is overlaid for all selected model orders.
    """
    histories = load_smooth_pulse_histories(root_dir, orders, case)
    reference_df = histories[orders[0]]
    requested_times = selected_times

    zeta = np.linspace(0.0, 1.0, n_zeta)

    ncols = len(requested_times)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), sharey=True)

    if ncols == 1:
        axes = np.asarray([axes])

    for col, requested_time in enumerate(requested_times):
        ax = axes[col]

        for order, df in histories.items():
            local_time = _nearest_time(df, requested_time)

            snap = df[np.isclose(df["time"], local_time)].copy()
            snap = snap.sort_values("x")
            row = _select_profile_row(
                df,
                time_value=requested_time,
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

        ax.set_title(f"t = {requested_time:.3f}")
        ax.set_xlabel(r"$u(\zeta)$")
        ax.set_ylabel(r"$\zeta$")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    for ax in np.asarray(axes).ravel():
        ax.set_ylabel(r"$\zeta$")

    force_axis_tick_labels(axes)

    title_case = "" if case is None else f" ({case})"
    fig.tight_layout()

    _save_figure(fig, output_path)


# =============================================================================
# Optional: hyperbolicity summary
# =============================================================================

def plot_hyperbolicity_summary_comparison(
    root_dir: str | Path,
    output_path: str | Path,
    case: str | None = None,
    orders: tuple[int, ...] = ORDERS,
) -> None:
    """
    Plot hyperbolicity summary diagnostics for all available orders.

    This is optional and only useful if store_hyperbolicity=True was active.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    any_loaded = False

    for order in orders:
        path = _get_hyperbolicity_summary_path(root_dir, order, case)

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
        print(f"[skip] No hyperbolicity summary files were found for case={case}.")
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

    title_case = "" if case is None else f" ({case})"
    fig.tight_layout()

    _save_figure(fig, output_path)


# =============================================================================
# Case wrapper
# =============================================================================

def generate_smooth_pulse_model_comparison_figures(
    root_dir: str | Path,
    case: str,
    output_dir: str | Path | None = None,
    selected_times: tuple[float, ...] = SELECTED_TIMES,
    include_hyperbolicity: bool = INCLUDE_HYPERBOLICITY,
) -> None:
    """
    Generate all model-comparison figures for one smooth-pulse case.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root Smooth_Pulse directory.
    case : str
        Case suffix. Typical values are 'Mild' and 'Aggressive'.
    output_dir : str or pathlib.Path, optional
        Output directory. If None, uses
        root_dir / f"Smooth_Pulse_Comparison_Figures_{case}".
    selected_times : tuple of float
        Times used in spatial/profile comparison figures.
    include_hyperbolicity : bool
        Whether to create the hyperbolicity summary comparison if data exist.

    Returns
    -------
    None
    """
    root_dir = Path(root_dir)

    if output_dir is None:
        output_dir = root_dir / _comparison_folder_name(case)
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[case] {case}")
    print(f"[root] {root_dir.resolve()}")
    print(f"[out ] {output_dir.resolve()}")

    plot_smooth_pulse_bulk_comparison(
        root_dir=root_dir,
        output_dir=output_dir,
        case=case,
    )

    plot_smooth_pulse_spatial_profiles(
       root_dir=root_dir,
        selected_times=selected_times,
        output_path_h=output_dir / f"Smooth_Pulse_{case}_spatial_height_profiles_comparison.pdf",
        output_path_u=output_dir / f"Smooth_Pulse_{case}_spatial_velocity_profiles_comparison.pdf",
        case=case,
    )

    plot_smooth_pulse_moment_statistics(
        root_dir=root_dir,
        output_dir=output_dir,
        case=case,
    )

    plot_smooth_pulse_moment_statistics(
        root_dir=root_dir,
        output_dir=output_dir,
        case=case,
    )

    plot_smooth_pulse_vertical_profiles(
        root_dir=root_dir,
        selected_times=selected_times,
        output_path=output_dir / f"Smooth_Pulse_{case}_vertical_profiles_comparison.pdf",
        case=case,
        selection="fixed_x",
        fixed_x=0.50,
    )

    if include_hyperbolicity:
        plot_hyperbolicity_summary_comparison(
            root_dir=root_dir,
            output_path=output_dir / f"Smooth_Pulse_{case}_hyperbolicity_summary.pdf",
            case=case,
        )

def generate_all_cases() -> None:
    """
    Generate comparison figures for every case listed in CASES.
    """
    if not ROOT_DIR.exists():
        raise FileNotFoundError(
            f"ROOT_DIR does not exist:\n{ROOT_DIR}\n"
            "Edit ROOT_DIR at the top of this script."
        )

    for case in CASES:
        generate_smooth_pulse_model_comparison_figures(
            root_dir=ROOT_DIR,
            case=case,
            output_dir=None,
            selected_times=SELECTED_TIMES,
            include_hyperbolicity=INCLUDE_HYPERBOLICITY,
        )

    print("\n[done] Smooth-pulse comparison figures generated.")


if __name__ == "__main__":
    generate_all_cases()
