#!/usr/bin/env python3
"""
Create two zoom/difference plots for the non-wrapping rainfall-exfiltration pulse test.

Generated figures:
1. Vertical profiles with an inset zoom comparing N=0 and N=1.
2. Direct difference profile: Delta u_10(zeta) = u_N1(zeta) - u_N0(zeta).

Expected folder structure:

Non_Wrapping_Pulse/
├── Non_Wrapping_Pulse_N0/
│   └── recharge_swme_N0_constant_field_history.csv
├── Non_Wrapping_Pulse_N1/
│   └── recharge_swme_N1_constant_field_history.csv
└── Non_Wrapping_Pulse_N2/
    └── recharge_swme_N2_constant_field_history.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


# =============================================================================
# User settings
# =============================================================================

ROOT_DIR = Path("/home/anenin/Documents/Git/thesis/model/processing/Non_Wrapping_Pulse")

OUTPUT_DIR = ROOT_DIR / "Non_Wrapping_Pulse_Comparison_Figures"

SELECTED_TIMES = (0.0, 0.5, 1.0)

DIFFERENCE_TIME = 1.0

N_ZETA = 300

# Selection rule for the horizontal point where u(zeta) is reconstructed.
# Recommended:
#   "shared_peak_N0" = choose x from N=0 peak velocity, then use same x for all orders.
# Alternative:
#   "fixed_x" = use FIXED_X for all orders.
SELECTION_RULE = "shared_peak_N0"
FIXED_X = 2.30

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
        color = "#E69F00"      # orange-red
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
    Force tick labels to appear on all subplots, even when sharey=True.
    """
    for ax in np.asarray(axes).ravel():
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=True,
            labelleft=True,
        )

# =============================================================================
# Basis functions and reconstruction
# =============================================================================

def phi_1(zeta: np.ndarray) -> np.ndarray:
    """
    First shifted Legendre-type basis function.
    """
    return 1.0 - 2.0 * zeta


def phi_2(zeta: np.ndarray) -> np.ndarray:
    """
    Second shifted Legendre-type basis function.
    """
    return 1.0 - 6.0 * zeta + 6.0 * zeta**2


def reconstruct_vertical_velocity(row: pd.Series, zeta: np.ndarray) -> np.ndarray:
    """
    Reconstruct u(zeta) from one CSV row.

    N=0:
        u(zeta) = u_m

    N=1:
        u(zeta) = u_m + a1 * phi_1(zeta)

    N=2:
        u(zeta) = u_m + a1 * phi_1(zeta) + a2 * phi_2(zeta)
    """
    profile = np.full_like(zeta, float(row["u_m"]), dtype=float)

    if "a1" in row.index and pd.notna(row["a1"]):
        profile += float(row["a1"]) * phi_1(zeta)

    if "a2" in row.index and pd.notna(row["a2"]):
        profile += float(row["a2"]) * phi_2(zeta)

    return profile


# =============================================================================
# Loading and selection utilities
# =============================================================================

def history_path(order: int) -> Path:
    """
    Return the expected field-history CSV path for moment order N.
    """
    return (
        ROOT_DIR
        / f"Non_Wrapping_Pulse_N{order}"
        / f"recharge_swme_N{order}_constant_field_history.csv"
    )


def load_histories() -> dict[int, pd.DataFrame]:
    """
    Load N=0, N=1, and N=2 field-history CSVs.
    """
    histories = {}

    for order in (0, 1, 2):
        path = history_path(order)

        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for N={order}:\n{path}")

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
        histories[order] = df

    return histories


def nearest_time(df: pd.DataFrame, target_time: float) -> float:
    """
    Return the stored time closest to target_time.
    """
    times = np.asarray(sorted(df["time"].unique()), dtype=float)

    if times.size == 0:
        raise ValueError("No stored times found in dataframe.")

    return float(times[np.argmin(np.abs(times - target_time))])


def snapshot_at_time(df: pd.DataFrame, target_time: float) -> tuple[float, pd.DataFrame]:
    """
    Return the snapshot closest to target_time.
    """
    t = nearest_time(df, target_time)
    snap = df[np.isclose(df["time"], t)].copy()
    snap = snap.sort_values("x")
    return t, snap


def row_at_x(snap: pd.DataFrame, x_target: float) -> pd.Series:
    """
    Return the row closest to x_target.
    """
    x_values = snap["x"].to_numpy(dtype=float)
    idx_pos = int(np.argmin(np.abs(x_values - x_target)))
    return snap.iloc[idx_pos]


def choose_shared_x(histories: dict[int, pd.DataFrame], target_time: float) -> tuple[float, float]:
    """
    Choose a shared x-location for profile reconstruction.

    If SELECTION_RULE == "shared_peak_N0":
        choose the location of maximum u_m in the N=0 solution.

    If SELECTION_RULE == "fixed_x":
        use FIXED_X.
    """
    reference_time, snap0 = snapshot_at_time(histories[0], target_time)

    if SELECTION_RULE == "shared_peak_N0":
        idx_peak = snap0["u_m"].idxmax()
        x_shared = float(snap0.loc[idx_peak, "x"])
        return reference_time, x_shared

    if SELECTION_RULE == "fixed_x":
        return reference_time, float(FIXED_X)

    raise ValueError(
        "Unknown SELECTION_RULE. Use 'shared_peak_N0' or 'fixed_x'."
    )


def select_rows_at_shared_x(
    histories: dict[int, pd.DataFrame],
    target_time: float,
) -> tuple[float, float, dict[int, pd.Series]]:
    """
    Select one shared x-location and return the nearest row for each model order.
    """
    reference_time, x_shared = choose_shared_x(histories, target_time)

    rows = {}

    for order, df in histories.items():
        _, snap = snapshot_at_time(df, target_time)
        rows[order] = row_at_x(snap, x_shared)

    return reference_time, x_shared, rows


# =============================================================================
# Figure 1: vertical profiles with inset zoom
# =============================================================================

def plot_vertical_profiles_standard(
    histories: dict[int, pd.DataFrame],
    output_path: Path,
    selected_times: tuple[float, ...] = SELECTED_TIMES,
) -> None:
    """
    Plot reconstructed vertical velocity profiles without inset zoom.

    This figure remains clean and shows the full N=0, N=1, N=2 hierarchy.
    The N=0/N=1 zoom is generated separately.
    """
    zeta = np.linspace(0.0, 1.0, N_ZETA)

    fig, axes = plt.subplots(
        1,
        len(selected_times),
        figsize=(5.7 * len(selected_times), 5.1),
        sharey=True,
    )

    if len(selected_times) == 1:
        axes = np.asarray([axes])

    for col, target_time in enumerate(selected_times):
        ax = axes[col]

        reference_time, x_shared, rows = select_rows_at_shared_x(
            histories,
            target_time,
        )

        for order in (0, 1, 2):
            row = rows[order]
            profile = reconstruct_vertical_velocity(row, zeta)

            plot_model_curve(
                ax=ax,
                x=profile,
                y=zeta,
                order=order,
                label=f"N={order}, x={row['x']:.3f}",
            )

        ax.set_title(f"t = {reference_time:.3f}")
        ax.set_xlabel(r"$u(\zeta)$")
        ax.set_ylabel(r"$\zeta$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    force_axis_tick_labels(axes)
    fig.subplots_adjust(wspace=0.25, top=0.84)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {output_path}")

def plot_n0_n1_zoomed_profile(
    histories: dict[int, pd.DataFrame],
    output_path: Path,
    target_time: float = DIFFERENCE_TIME,
    ) -> None:
    """
    Generate a separate zoom-only figure comparing N=0 and N=1.

    This avoids cluttering the main reconstructed-profile figure with an inset.
    The comparison uses the same shared x-location as the difference plot.
    """
    zeta = np.linspace(0.0, 1.0, N_ZETA)

    reference_time, x_shared, rows = select_rows_at_shared_x(
        histories,
        target_time,
    )

    u0 = reconstruct_vertical_velocity(rows[0], zeta)
    u1 = reconstruct_vertical_velocity(rows[1], zeta)

    delta = u1 - u0
    max_abs_delta = float(np.max(np.abs(delta)))
    a1_value = float(rows[1]["a1"]) if "a1" in rows[1].index else np.nan

    fig, ax = plt.subplots(figsize=(6.5, 5.2))

    ax.plot(u0, zeta, label=r"$N=0$", linewidth=2.0)
    ax.plot(u1, zeta, label=r"$N=1$", linewidth=2.0)

    u01 = np.concatenate([u0, u1])
    u_min = float(np.min(u01))
    u_max = float(np.max(u01))

    pad = max(1e-5, 0.25 * (u_max - u_min))

    ax.set_xlim(u_min - pad, u_max + pad)
    ax.set_ylim(0.0, 1.0)

    ax.set_title(
        rf"$t={reference_time:.3f}$, $x={x_shared:.3f}$"
    )

    ax.set_xlabel(r"$u(\zeta)$")
    ax.set_ylabel(r"$\zeta$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    annotation = (
        rf"$\max_\zeta |u_1-u_0| = {max_abs_delta:.3e}$" + "\n"
        rf"$\alpha_1(x,t) = {a1_value:.3e}$"
    )

    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {output_path}")

# =============================================================================
# Figure 2: N=1 minus N=0 difference
# =============================================================================

def plot_n1_minus_n0_difference(
    histories: dict[int, pd.DataFrame],
    output_path: Path,
    target_time: float = DIFFERENCE_TIME,
) -> None:
    """
    Plot Delta u_10(zeta) = u_N1(zeta) - u_N0(zeta) at a shared x-location.
    """
    zeta = np.linspace(0.0, 1.0, N_ZETA)

    reference_time, x_shared, rows = select_rows_at_shared_x(
        histories,
        target_time,
    )

    u0 = reconstruct_vertical_velocity(rows[0], zeta)
    u1 = reconstruct_vertical_velocity(rows[1], zeta)

    delta = u1 - u0

    fig, ax = plt.subplots(figsize=(6.8, 5.1))

    ax.plot(
        delta,
        zeta,
        label=r"$u_{N=1}(\zeta)-u_{N=0}(\zeta)$",
    )

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        label="zero difference",
    )

    ax.set_title(
        rf"$t={reference_time:.3f}$, $x={x_shared:.3f}$"
    )
    
    ax.set_xlabel(r"$\Delta u_{10}(\zeta)$")
    ax.set_ylabel(r"$\zeta$")
    ax.grid(True, alpha=0.3)
    ax.legend()

    max_abs_delta = float(np.max(np.abs(delta)))
    a1_value = float(rows[1]["a1"]) if "a1" in rows[1].index else np.nan

    annotation = (
        rf"$\max_\zeta |\Delta u_{{10}}| = {max_abs_delta:.3e}$" + "\n"
        rf"$\alpha_1(x,t) = {a1_value:.3e}$"
    )

    ax.text(
        0.04,
        0.96,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {output_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    histories = load_histories()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_vertical_profiles_standard(
        histories=histories,
        output_path=OUTPUT_DIR / "Non_Wrapping_Pulse_vertical_profiles_clean.pdf",
        selected_times=SELECTED_TIMES,
    )

    plot_n0_n1_zoomed_profile(
        histories=histories,
        output_path=OUTPUT_DIR / "Non_Wrapping_Pulse_N0_N1_zoomed_profile.pdf",
        target_time=DIFFERENCE_TIME,
    )

    plot_n1_minus_n0_difference(
        histories=histories,
        output_path=OUTPUT_DIR / "Non_Wrapping_Pulse_N1_minus_N0_vertical_profile_difference.pdf",
        target_time=DIFFERENCE_TIME,
    )

    print("[done] clean profile, zoom, and difference figures generated.")

if __name__ == "__main__":
    main()
