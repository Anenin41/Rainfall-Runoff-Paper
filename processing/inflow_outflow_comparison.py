#!/usr/bin/env python3
"""
Smooth-pulse Horton inflow/outflow model-comparison plots.

Run with:
    python inflow_outflow_comparison.py

Expected folder structure:

Smooth_Pulse_Inflow_Outflow/
├── Smooth_Pulse_Comparison_Figures_I/
├── Smooth_Pulse_Inflow_Outflow_N0/
│   └── recharge_swme_N0_horton_field_history.csv
├── Smooth_Pulse_Inflow_Outflow_N1/
│   └── recharge_swme_N1_horton_field_history.csv
└── Smooth_Pulse_Inflow_Outflow_N2/
    └── recharge_swme_N2_horton_field_history.csv

Generated figure groups:
    1. Bulk quantities, split into four standalone figures.
    2. Spatial profiles, split into height and velocity figures.
    3. Moment statistics, split into four standalone figures.
    4. Reconstructed vertical velocity profiles.
    5. Outlet hydrographs, split into discharge, height, and velocity figures.
    6. Optional hyperbolicity diagnostics, split into two standalone figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# User settings
# =============================================================================

ROOT_DIR = Path(
    "/home/anenin/Documents/Git/thesis/model/processing/Smooth_Pulse_Inflow_Outflow"
)
OUTPUT_DIR = ROOT_DIR / "Smooth_Pulse_Comparison_Figures_I"

ORDERS = (0, 1, 2)
SOURCE_TAG = "horton"

SELECTED_TIMES = (0.0, 0.2, 0.4, 1.8)

INCLUDE_OUTLET_HYDROGRAPH = True
INCLUDE_HYPERBOLICITY = False

# "shared_peak_N0" is usually best for comparing representative profiles.
# "fixed_x" is stricter if you want all profiles at one prescribed location.
PROFILE_SELECTION = "fixed_x"
FIXED_X = 0.5

N_ZETA = 300


# =============================================================================
# Plot styling
# =============================================================================

plt.style.use("tableau-colorblind10")


def plot_model_curve(ax, x, y, order: int, label: str) -> None:
    """
    Plot one model-order curve using the thesis styling.

    N=0: blue, solid, linewidth 3.0.
    N=1: orange, dash-dot, linewidth 2.0.
    N=2: black, dashed, linewidth 1.5.
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


def as_flat_axes(axes) -> np.ndarray:
    """
    Return a matplotlib axes object/array as a flat numpy array.
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
    Apply a common x-axis range to all axes and show x tick labels everywhere.
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
    Apply a common y-axis range to all axes and show y tick labels everywhere.
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
# Loading utilities
# =============================================================================


def order_folder_name(order: int) -> str:
    return f"Smooth_Pulse_Inflow_Outflow_N{order}"


def history_path(order: int) -> Path:
    return (
        ROOT_DIR
        / order_folder_name(order)
        / f"recharge_swme_N{order}_{SOURCE_TAG}_field_history.csv"
    )


def hyperbolicity_summary_path(order: int) -> Path:
    return ROOT_DIR / order_folder_name(order) / "recharge_hyperbolicity_summary.csv"


def load_histories(orders: tuple[int, ...] = ORDERS) -> dict[int, pd.DataFrame]:
    histories: dict[int, pd.DataFrame] = {}

    for order in orders:
        path = history_path(order)

        if not path.exists():
            raise FileNotFoundError(f"Missing field-history file for N={order}:\n{path}")

        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]

        required = {"time", "x", "h", "u_m"}
        missing = required.difference(df.columns)

        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}\n"
                f"Available columns: {list(df.columns)}"
            )

        for col in ["time", "x", "h", "u_m", "a1", "a2"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["time", "x", "h", "u_m"]).copy()
        df["N"] = order
        histories[order] = df

    return histories


def nearest_time(df: pd.DataFrame, target_time: float) -> float:
    times = np.asarray(sorted(df["time"].unique()), dtype=float)

    if times.size == 0:
        raise ValueError("No stored times found in dataframe.")

    return float(times[np.argmin(np.abs(times - target_time))])


def snapshot_at_time(df: pd.DataFrame, target_time: float) -> tuple[float, pd.DataFrame]:
    t = nearest_time(df, target_time)
    snap = df[np.isclose(df["time"], t)].copy()
    snap = snap.sort_values("x")
    return t, snap


def row_at_x(snap: pd.DataFrame, x_target: float) -> pd.Series:
    distances = np.abs(snap["x"].to_numpy(dtype=float) - float(x_target))
    return snap.iloc[int(np.argmin(distances))]


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")


# =============================================================================
# Bulk statistics
# =============================================================================


def compute_bulk_statistics(df: pd.DataFrame) -> pd.DataFrame:
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
    output_path: Path,
) -> None:
    """
    Plot one bulk quantity as a standalone figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for order, df in histories.items():
        stats = compute_bulk_statistics(df)
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
    save_figure(fig, output_path)


def plot_bulk_comparison(histories: dict[int, pd.DataFrame], output_dir: Path) -> None:
    """
    Plot bulk response comparison as four separate figures.
    """
    plot_single_bulk_quantity(
        histories=histories,
        quantity="h_mean",
        ylabel=r"$\overline{h}$",
        output_path=output_dir / "Inflow_Outflow_bulk_mean_height.pdf",
    )

    plot_single_bulk_quantity(
        histories=histories,
        quantity="h_range",
        ylabel=r"$\max_x h - \min_x h$",
        output_path=output_dir / "Inflow_Outflow_bulk_height_spread.pdf",
    )

    plot_single_bulk_quantity(
        histories=histories,
        quantity="u_mean",
        ylabel=r"$\overline{u_m}$",
        output_path=output_dir / "Inflow_Outflow_bulk_mean_velocity.pdf",
    )

    plot_single_bulk_quantity(
        histories=histories,
        quantity="q_mean",
        ylabel=r"$\overline{q}=\overline{h u_m}$",
        output_path=output_dir / "Inflow_Outflow_bulk_mean_discharge.pdf",
    )


# =============================================================================
# Spatial profiles
# =============================================================================


def plot_spatial_profiles(
    histories: dict[int, pd.DataFrame],
    output_path_h: Path,
    output_path_u: Path,
    selected_times: tuple[float, ...] = SELECTED_TIMES,
) -> None:
    """
    Plot spatial profiles as two separate figures:
        1. h(x) at selected times.
        2. u_m(x) at selected times.

    Both figures use common x- and y-axis ranges across their subplots.
    """
    ncols = len(selected_times)

    # -------------------------------------------------------------------------
    # Figure 1: height profiles h(x)
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

    for col, requested_time in enumerate(selected_times):
        ax = axes_h[col]

        for order, df in histories.items():
            _, snap = snapshot_at_time(df, requested_time)

            plot_model_curve(
                ax=ax,
                x=snap["x"],
                y=snap["h"],
                order=order,
                label=f"N={order}",
            )

        ax.set_title(f"t = {requested_time:.3f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$h$")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    apply_common_xlim(axes_h)
    apply_common_ylim(axes_h)
    fig_h.tight_layout()
    save_figure(fig_h, output_path_h)

    # -------------------------------------------------------------------------
    # Figure 2: mean-velocity profiles u_m(x)
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

    for col, requested_time in enumerate(selected_times):
        ax = axes_u[col]

        for order, df in histories.items():
            _, snap = snapshot_at_time(df, requested_time)

            plot_model_curve(
                ax=ax,
                x=snap["x"],
                y=snap["u_m"],
                order=order,
                label=f"N={order}",
            )

        ax.set_title(f"t = {requested_time:.3f}")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u_m$")
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.95)

    apply_common_xlim(axes_u)
    apply_common_ylim(axes_u)
    fig_u.tight_layout()
    save_figure(fig_u, output_path_u)


# =============================================================================
# Moment statistics
# =============================================================================


def compute_moment_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute L2/Euclidean norm and maximum absolute moment amplitudes over time.

    The L2 norm is the discrete Euclidean norm over the stored spatial cells.
    Since all model-order runs use the same grid, this is suitable for comparing
    N=1 and N=2 in this test.
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


def plot_single_moment_quantity(
    histories: dict[int, pd.DataFrame],
    quantity: str,
    ylabel: str,
    output_path: Path,
    relevant_orders: tuple[int, ...],
) -> None:
    """
    Plot one moment statistic as a standalone figure.

    Relevant model orders:
        alpha_1 quantities: N=1 and N=2.
        alpha_2 quantities: N=2 only.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for order in relevant_orders:
        if order not in histories:
            continue

        stats = compute_moment_statistics(histories[order])
        if quantity not in stats.columns:
            continue

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
    save_figure(fig, output_path)


def plot_moment_statistics(histories: dict[int, pd.DataFrame], output_dir: Path) -> None:
    """
    Plot moment statistics as four separate figures.
    """
    plot_single_moment_quantity(
        histories=histories,
        quantity="a1_l2",
        ylabel=r"$\|\alpha_1\|_2$",
        output_path=output_dir / "Inflow_Outflow_a1_l2_norm_comparison.pdf",
        relevant_orders=(1, 2),
    )

    plot_single_moment_quantity(
        histories=histories,
        quantity="a1_max_abs",
        ylabel=r"$\max_x |\alpha_1|$",
        output_path=output_dir / "Inflow_Outflow_a1_max_abs_comparison.pdf",
        relevant_orders=(1, 2),
    )

    plot_single_moment_quantity(
        histories=histories,
        quantity="a2_l2",
        ylabel=r"$\|\alpha_2\|_2$",
        output_path=output_dir / "Inflow_Outflow_a2_l2_norm_comparison.pdf",
        relevant_orders=(2,),
    )

    plot_single_moment_quantity(
        histories=histories,
        quantity="a2_max_abs",
        ylabel=r"$\max_x |\alpha_2|$",
        output_path=output_dir / "Inflow_Outflow_a2_max_abs_comparison.pdf",
        relevant_orders=(2,),
    )


# =============================================================================
# Vertical profiles
# =============================================================================


def phi_1(zeta: np.ndarray) -> np.ndarray:
    return 1.0 - 2.0 * zeta


def phi_2(zeta: np.ndarray) -> np.ndarray:
    return 1.0 - 6.0 * zeta + 6.0 * zeta**2


def reconstruct_vertical_velocity(row: pd.Series, zeta: np.ndarray) -> np.ndarray:
    profile = np.full_like(zeta, float(row["u_m"]), dtype=float)

    if "a1" in row.index and pd.notna(row["a1"]):
        profile += float(row["a1"]) * phi_1(zeta)

    if "a2" in row.index and pd.notna(row["a2"]):
        profile += float(row["a2"]) * phi_2(zeta)

    return profile


def choose_profile_x(
    histories: dict[int, pd.DataFrame],
    target_time: float,
) -> tuple[float, float]:
    reference_time, snap0 = snapshot_at_time(histories[0], target_time)

    if PROFILE_SELECTION == "shared_peak_N0":
        idx_peak = snap0["u_m"].idxmax()
        return reference_time, float(snap0.loc[idx_peak, "x"])

    if PROFILE_SELECTION == "fixed_x":
        return reference_time, float(FIXED_X)

    raise ValueError("PROFILE_SELECTION must be 'shared_peak_N0' or 'fixed_x'.")


def select_profile_rows(
    histories: dict[int, pd.DataFrame],
    target_time: float,
) -> tuple[float, float, dict[int, pd.Series]]:
    reference_time, x_shared = choose_profile_x(histories, target_time)
    rows: dict[int, pd.Series] = {}

    for order, df in histories.items():
        _, snap = snapshot_at_time(df, target_time)
        rows[order] = row_at_x(snap, x_shared)

    return reference_time, x_shared, rows


def plot_vertical_profiles(
    histories: dict[int, pd.DataFrame],
    output_path: Path,
    selected_times: tuple[float, ...] = SELECTED_TIMES,
) -> None:
    """
    Plot reconstructed vertical velocity profiles for the selected times.

    The y-axis is shared because all subplots use zeta in [0, 1]. The x-axis is
    also shared to make the profile magnitudes directly comparable across time.
    """
    zeta = np.linspace(0.0, 1.0, N_ZETA)

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

    for col, requested_time in enumerate(selected_times):
        ax = axes[col]

        _, _, rows = select_profile_rows(histories, requested_time)

        for order in ORDERS:
            row = rows[order]
            profile = reconstruct_vertical_velocity(row, zeta)

            plot_model_curve(
                ax=ax,
                x=profile,
                y=zeta,
                order=order,
                label=f"N={order}, x={row['x']:.3f}",
            )

        ax.set_title(f"t = {requested_time:.3f}")
        ax.set_xlabel(r"$u(\zeta)$")
        ax.set_ylabel(r"$\zeta$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    apply_common_xlim(axes)
    apply_common_ylim(axes)
    fig.tight_layout()
    save_figure(fig, output_path)


# =============================================================================
# Outlet hydrograph
# =============================================================================


def compute_outlet_series(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for time_value, group in df.groupby("time"):
        outlet = group.sort_values("x").iloc[-1]

        h = float(outlet["h"])
        u = float(outlet["u_m"])

        rows.append(
            {
                "time": float(time_value),
                "x_out": float(outlet["x"]),
                "h_out": h,
                "u_out": u,
                "q_out": h * u,
            }
        )

    return pd.DataFrame(rows).sort_values("time")


def plot_single_outlet_quantity(
    histories: dict[int, pd.DataFrame],
    quantity: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot one outlet quantity as a standalone hydrograph figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for order, df in histories.items():
        outlet = compute_outlet_series(df)
        plot_model_curve(
            ax=ax,
            x=outlet["time"],
            y=outlet[quantity],
            order=order,
            label=f"N={order}",
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_outlet_hydrograph(histories: dict[int, pd.DataFrame], output_dir: Path) -> None:
    """
    Plot outlet hydrographs as three separate figures.
    """
    plot_single_outlet_quantity(
        histories=histories,
        quantity="q_out",
        ylabel=r"$q_{out}=h u_m$",
        output_path=output_dir / "Inflow_Outflow_outlet_discharge.pdf",
    )

    plot_single_outlet_quantity(
        histories=histories,
        quantity="h_out",
        ylabel=r"$h_{out}$",
        output_path=output_dir / "Inflow_Outflow_outlet_height.pdf",
    )

    plot_single_outlet_quantity(
        histories=histories,
        quantity="u_out",
        ylabel=r"$u_{out}$",
        output_path=output_dir / "Inflow_Outflow_outlet_velocity.pdf",
    )


# =============================================================================
# Hyperbolicity diagnostics
# =============================================================================


def load_hyperbolicity_summaries() -> dict[int, pd.DataFrame]:
    loaded: dict[int, pd.DataFrame] = {}

    for order in ORDERS:
        path = hyperbolicity_summary_path(order)

        if not path.exists():
            continue

        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]

        if not df.empty:
            loaded[order] = df

    return loaded


def plot_single_hyperbolicity_quantity(
    summaries: dict[int, pd.DataFrame],
    quantity: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """
    Plot one hyperbolicity diagnostic as a standalone figure.
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    any_plotted = False

    for order, df in summaries.items():
        if quantity not in df.columns:
            continue

        plot_model_curve(
            ax=ax,
            x=df["time"],
            y=df[quantity],
            order=order,
            label=f"N={order}",
        )
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print(f"[skip] hyperbolicity quantity not found: {quantity}")
        return

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)

    fig.tight_layout()
    save_figure(fig, output_path)


def plot_hyperbolicity_summary(output_dir: Path) -> None:
    """
    Plot hyperbolicity diagnostics as separate figures.
    """
    summaries = load_hyperbolicity_summaries()

    if not summaries:
        print("[skip] no hyperbolicity summary files found.")
        return

    plot_single_hyperbolicity_quantity(
        summaries=summaries,
        quantity="num_nonhyperbolic_cells",
        ylabel="Count",
        output_path=output_dir / "Inflow_Outflow_hyperbolicity_nonhyperbolic_cell_count.pdf",
    )

    plot_single_hyperbolicity_quantity(
        summaries=summaries,
        quantity="max_abs_imag_eig",
        ylabel=r"$\max |\operatorname{Im}(\lambda)|$",
        output_path=output_dir / "Inflow_Outflow_hyperbolicity_max_imaginary_eigenvalue.pdf",
    )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    if not ROOT_DIR.exists():
        raise FileNotFoundError(
            f"ROOT_DIR does not exist:\n{ROOT_DIR}\n"
            "Edit ROOT_DIR at the top of this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    histories = load_histories(ORDERS)

    plot_bulk_comparison(
        histories=histories,
        output_dir=OUTPUT_DIR,
    )

    plot_spatial_profiles(
        histories=histories,
        output_path_h=OUTPUT_DIR / "Inflow_Outflow_spatial_height_profiles_comparison.pdf",
        output_path_u=OUTPUT_DIR / "Inflow_Outflow_spatial_velocity_profiles_comparison.pdf",
        selected_times=SELECTED_TIMES,
    )

    plot_moment_statistics(
        histories=histories,
        output_dir=OUTPUT_DIR,
    )

    plot_vertical_profiles(
        histories=histories,
        output_path=OUTPUT_DIR / "Inflow_Outflow_vertical_profiles_comparison.pdf",
        selected_times=SELECTED_TIMES,
    )

    if INCLUDE_OUTLET_HYDROGRAPH:
        plot_outlet_hydrograph(
            histories=histories,
            output_dir=OUTPUT_DIR,
        )

    if INCLUDE_HYPERBOLICITY:
        plot_hyperbolicity_summary(
            output_dir=OUTPUT_DIR,
        )

    print("[done] inflow/outflow comparison figures generated.")


if __name__ == "__main__":
    main()