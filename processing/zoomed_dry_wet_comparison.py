#!/usr/bin/env python3
"""
Report-style zoomed vertical-profile comparisons for the Dry/Wet ablation test.

This script generates one zoomed comparison figure and one direct-difference
figure for each requested pair. The vertical-profile styling is aligned with
`dry_wet_ablation_comparison.py`:

    - Source-free/source-active labels are used instead of Dry/Wet labels.
    - N=1 curves stay in the orange family.
    - N=2 curves stay in the black/grey family.
    - All absolute-profile zoom figures use one common x-axis and y-axis.
    - All difference figures use one common x-axis and y-axis.
    - Tick labels are forced on every generated axis.

Run with:
    python zoomed_dry_wet_comparison_modified.py

Expected folder structure:
    /home/anenin/Documents/Git/thesis/model/processing/Dry_Wet_Test/
    ├── Dry_N1/
    │   └── recharge_swme_N1_constant_field_history.csv
    ├── Dry_N2/
    │   └── recharge_swme_N2_constant_field_history.csv
    ├── Wet_N1/
    │   └── recharge_swme_N1_constant_field_history.csv
    └── Wet_N2/
        └── recharge_swme_N2_constant_field_history.csv

Output folder:
    Dry_Wet_Test/Dry_Wet_Comparison_Figures/Vertical_Profiles_Zoom/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# =============================================================================
# Absolute paths and user settings
# =============================================================================

ROOT_DIR = Path("/home/anenin/Documents/Git/thesis/model/processing/Dry_Wet_Test")

OUTPUT_DIR = (
    ROOT_DIR
    / "Dry_Wet_Comparison_Figures"
    / "Vertical_Profiles_Zoom"
)

SOURCE_TAG = "constant"

# Use the same physical location and time for all pairwise zooms.
TARGET_TIME = 1.0
TARGET_X = 0.501

N_ZETA = 400

# One common axis range is computed from all pairs before plotting.
# This keeps the zoomed figures visually comparable.
PROFILE_X_PAD_FRACTION = 0.05
DIFFERENCE_X_PAD_FRACTION = 0.05
MIN_PROFILE_X_PAD = 1e-5
MIN_DIFFERENCE_X_PAD = 1e-7

# If you only want one figure, keep only the desired pair in this list.
PAIR_SPECS: list[dict[str, Any]] = [
    {
        "name": "Dry_N1_vs_Dry_N2",
        "case_a": ("Dry", 1),
        "case_b": ("Dry", 2),
        "caption_hint": "source-free N=2 is close to, but not identical to, source-free N=1",
    },
    {
        "name": "Wet_N1_vs_Wet_N2",
        "case_a": ("Wet", 1),
        "case_b": ("Wet", 2),
        "caption_hint": "source-active N=2 develops a visibly curved profile relative to source-active N=1",
    },
    {
        "name": "Dry_N1_vs_Wet_N1",
        "case_a": ("Dry", 1),
        "case_b": ("Wet", 1),
        "caption_hint": "the source terms shift the N=1 profile",
    },
    {
        "name": "Dry_N2_vs_Wet_N2",
        "case_a": ("Dry", 2),
        "case_b": ("Wet", 2),
        "caption_hint": "the source terms strongly excite the N=2 curvature mode",
    },
]


# =============================================================================
# Plot styling
# =============================================================================

plt.style.use("tableau-colorblind10")


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


def as_flat_axes(axes) -> np.ndarray:
    """
    Return a matplotlib axes object or axes array as a flat numpy array.
    """
    return np.asarray(axes).ravel()


def force_axis_tick_labels(axes) -> None:
    """
    Force tick labels to appear on every axis.
    """
    for ax in as_flat_axes(axes):
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=True,
            labelleft=True,
        )


def regime_display_name(regime: str) -> str:
    """
    Return the display name for a regime.

    Dry -> Source-free
    Wet -> Source-active
    """
    if regime == "Dry":
        return "Source-free"
    if regime == "Wet":
        return "Source-active"
    return regime


def run_label(regime: str, order: int) -> str:
    """
    Return a compact label for legends.
    """
    return f"{regime_display_name(regime)}, N={order}"


def compact_case_tag(regime: str, order: int) -> str:
    """
    Return a compact tag for annotation-box superscripts.
    """
    if regime == "Dry":
        prefix = "SF"
    elif regime == "Wet":
        prefix = "SA"
    else:
        prefix = regime

    return f"{prefix}{order}"


def plot_vertical_profile_curve(
    ax: plt.Axes,
    x,
    y,
    regime: str,
    order: int,
    label: str,
) -> None:
    """
    Plot vertical-profile curves with the same local palette as the master
    dry/wet comparison file.
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


def plot_difference_curve(
    ax: plt.Axes,
    x,
    y,
    regime: str,
    order: int,
    label: str,
) -> None:
    """
    Plot a difference curve using the style of the second profile in the pair,
    because the plotted quantity is u_b - u_a.
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


def configure_profile_axis(
    ax: plt.Axes,
    xlim: tuple[float, float],
    xlabel: str,
) -> None:
    """
    Apply common axis formatting to an absolute-profile plot.
    """
    ax.set_xlim(*xlim)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\zeta$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    force_axis_tick_labels([ax])


# =============================================================================
# Loading utilities
# =============================================================================

def case_folder(case: str, order: int) -> Path:
    """
    Return the folder for a Dry/Wet and N-order combination.
    """
    return ROOT_DIR / f"{case}_N{order}"


def field_history_path(case: str, order: int) -> Path:
    """
    Return the expected field-history CSV path.
    """
    return case_folder(case, order) / f"recharge_swme_N{order}_{SOURCE_TAG}_field_history.csv"


def load_case_history(case: str, order: int) -> pd.DataFrame:
    """
    Load one field-history CSV.

    Required columns:
        time, x, h, u_m

    Optional columns:
        a1, a2
    """
    path = field_history_path(case, order)

    if not path.exists():
        raise FileNotFoundError(f"Could not find field-history file:\n{path}")

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

    return df.dropna(subset=["time", "x", "h", "u_m"]).copy()


def nearest_time(df: pd.DataFrame, target_time: float) -> float:
    """
    Return stored time closest to target_time.
    """
    times = np.asarray(sorted(df["time"].dropna().unique()), dtype=float)

    if times.size == 0:
        raise ValueError("No time values found in dataframe.")

    return float(times[np.argmin(np.abs(times - target_time))])


def snapshot_at_time(df: pd.DataFrame, target_time: float) -> tuple[float, pd.DataFrame]:
    """
    Return snapshot closest to target_time.
    """
    t = nearest_time(df, target_time)
    snap = df[np.isclose(df["time"], t)].copy().sort_values("x")

    if snap.empty:
        raise ValueError(f"No snapshot found near t={target_time}")

    return t, snap


def select_row_at_x(
    df: pd.DataFrame,
    target_time: float,
    target_x: float,
) -> tuple[float, pd.Series]:
    """
    Select the row closest to target_x at the stored time nearest target_time.
    """
    t, snap = snapshot_at_time(df, target_time)
    distances = np.abs(snap["x"].to_numpy(dtype=float) - float(target_x))
    row = snap.iloc[int(np.argmin(distances))]

    return t, row


# =============================================================================
# Vertical reconstruction
# =============================================================================

def phi_1(zeta: np.ndarray) -> np.ndarray:
    """
    First shifted Legendre basis function.
    """
    return 1.0 - 2.0 * zeta


def phi_2(zeta: np.ndarray) -> np.ndarray:
    """
    Second shifted Legendre basis function.
    """
    return 1.0 - 6.0 * zeta + 6.0 * zeta**2


def reconstruct_vertical_velocity(row: pd.Series, zeta: np.ndarray) -> np.ndarray:
    """
    Reconstruct u(zeta) from one dataframe row.

    N=1:
        u(zeta) = u_m + alpha_1 phi_1(zeta)

    N=2:
        u(zeta) = u_m + alpha_1 phi_1(zeta) + alpha_2 phi_2(zeta)
    """
    profile = np.full_like(zeta, float(row["u_m"]), dtype=float)

    if "a1" in row.index and pd.notna(row["a1"]):
        profile += float(row["a1"]) * phi_1(zeta)

    if "a2" in row.index and pd.notna(row["a2"]):
        profile += float(row["a2"]) * phi_2(zeta)

    return profile


def moment_text(row: pd.Series, label: str) -> list[str]:
    """
    Create compact text lines with the moment values available in a row.
    """
    lines: list[str] = []

    if "a1" in row.index and pd.notna(row["a1"]):
        lines.append(rf"$\alpha_1^{{\mathrm{{{label}}}}}={float(row['a1']):.3e}$")

    if "a2" in row.index and pd.notna(row["a2"]):
        lines.append(rf"$\alpha_2^{{\mathrm{{{label}}}}}={float(row['a2']):.3e}$")

    return lines


# =============================================================================
# Pair-data preparation and common limits
# =============================================================================

def build_pair_data(pair_spec: dict[str, Any]) -> dict[str, Any]:
    """
    Load both cases in a pair and reconstruct the absolute and difference profiles.
    """
    case_a, order_a = pair_spec["case_a"]
    case_b, order_b = pair_spec["case_b"]

    df_a = load_case_history(case_a, order_a)
    df_b = load_case_history(case_b, order_b)

    t_a, row_a = select_row_at_x(df_a, TARGET_TIME, TARGET_X)
    t_b, row_b = select_row_at_x(df_b, TARGET_TIME, TARGET_X)

    t_plot = 0.5 * (t_a + t_b)
    x_plot = 0.5 * (float(row_a["x"]) + float(row_b["x"]))

    zeta = np.linspace(0.0, 1.0, N_ZETA)

    u_a = reconstruct_vertical_velocity(row_a, zeta)
    u_b = reconstruct_vertical_velocity(row_b, zeta)
    delta = u_b - u_a

    out = dict(pair_spec)
    out.update(
        {
            "case_a": case_a,
            "order_a": order_a,
            "case_b": case_b,
            "order_b": order_b,
            "label_a": run_label(case_a, order_a),
            "label_b": run_label(case_b, order_b),
            "row_a": row_a,
            "row_b": row_b,
            "t_plot": t_plot,
            "x_plot": x_plot,
            "zeta": zeta,
            "u_a": u_a,
            "u_b": u_b,
            "delta": delta,
            "max_abs_delta": float(np.max(np.abs(delta))),
        }
    )

    return out


def common_limits(
    arrays: list[np.ndarray],
    pad_fraction: float,
    min_pad: float,
    symmetric: bool = False,
) -> tuple[float, float]:
    """
    Compute one common x-axis limit from a list of arrays.
    """
    values = np.concatenate([np.asarray(arr, dtype=float).ravel() for arr in arrays])
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError("Cannot compute common axis limits from empty data.")

    if symmetric:
        max_abs = float(np.max(np.abs(values)))
        pad = max(min_pad, pad_fraction * max_abs)
        return -(max_abs + pad), max_abs + pad

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    width = vmax - vmin
    pad = max(min_pad, pad_fraction * width)

    return vmin - pad, vmax + pad


def build_common_axis_limits(pair_data: list[dict[str, Any]]) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute common x-axis limits for the absolute-profile and difference figures.
    """
    profile_arrays: list[np.ndarray] = []
    difference_arrays: list[np.ndarray] = []

    for data in pair_data:
        profile_arrays.extend([data["u_a"], data["u_b"]])
        difference_arrays.append(data["delta"])

    profile_xlim = common_limits(
        arrays=profile_arrays,
        pad_fraction=PROFILE_X_PAD_FRACTION,
        min_pad=MIN_PROFILE_X_PAD,
        symmetric=False,
    )

    difference_xlim = common_limits(
        arrays=difference_arrays,
        pad_fraction=DIFFERENCE_X_PAD_FRACTION,
        min_pad=MIN_DIFFERENCE_X_PAD,
        symmetric=True,
    )

    return profile_xlim, difference_xlim


# =============================================================================
# Plotting
# =============================================================================

def plot_pair_zoom(
    pair_data: dict[str, Any],
    profile_xlim: tuple[float, float],
) -> None:
    """
    Generate one report-style zoomed vertical-profile comparison for a pair.
    """
    case_a = pair_data["case_a"]
    order_a = pair_data["order_a"]
    case_b = pair_data["case_b"]
    order_b = pair_data["order_b"]

    fig, ax = plt.subplots(figsize=(6.0, 5.2))

    plot_vertical_profile_curve(
        ax=ax,
        x=pair_data["u_a"],
        y=pair_data["zeta"],
        regime=case_a,
        order=order_a,
        label=pair_data["label_a"],
    )

    plot_vertical_profile_curve(
        ax=ax,
        x=pair_data["u_b"],
        y=pair_data["zeta"],
        regime=case_b,
        order=order_b,
        label=pair_data["label_b"],
    )

    # Use individual limits for N=1 vs N=2 comparisons
    if (case_a == case_b) and (order_a != order_b):
        pair_xlim = common_limits(
            arrays=[pair_data["u_a"], pair_data["u_b"]],
            pad_fraction=PROFILE_X_PAD_FRACTION,
            min_pad=MIN_PROFILE_X_PAD,
            symmetric=False,
        )
    else:
        pair_xlim = profile_xlim

    ax.set_title(rf"$t={pair_data['t_plot']:.3f}$, $x={pair_data['x_plot']:.3f}$")
    configure_profile_axis(ax=ax, xlim=pair_xlim, xlabel=r"$u(\zeta)$")

    annotation_lines = [
        rf"$\max_\zeta |u_b-u_a|={pair_data['max_abs_delta']:.3e}$",
    ]

    short_a = compact_case_tag(case_a, order_a)
    short_b = compact_case_tag(case_b, order_b)

    annotation_lines.extend(moment_text(pair_data["row_a"], short_a))
    annotation_lines.extend(moment_text(pair_data["row_b"], short_b))

    ax.text(
        0.04,
        0.96,
        "\n".join(annotation_lines[:5]),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.85,
        },
    )

    output_name = (
        f"{pair_data['name']}"
        f"_zoom_t{TARGET_TIME:.3f}_x{TARGET_X:.3f}.pdf"
    ).replace(".", "p").replace("ppdf", ".pdf")

    output_path = OUTPUT_DIR / output_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format='pdf')
    plt.close(fig)

    print(f"[saved] {output_path}")


def plot_pair_difference(
    pair_data: dict[str, Any],
    difference_xlim: tuple[float, float],
) -> None:
    """
    Generate a second report-style figure showing u_b - u_a directly.
    """
    case_a = pair_data["case_a"]
    order_a = pair_data["order_a"]
    case_b = pair_data["case_b"]
    order_b = pair_data["order_b"]

    difference_label = f"{pair_data['label_b']} - {pair_data['label_a']}"

    fig, ax = plt.subplots(figsize=(6.0, 5.2))

    plot_difference_curve(
        ax=ax,
        x=pair_data["delta"],
        y=pair_data["zeta"],
        regime=case_b,
        order=order_b,
        label=difference_label,
    )

    ax.axvline(
        0.0,
        color="0.45",
        linestyle=":",
        linewidth=1.1,
        zorder=0,
    )

    # Use individual limits for N=1 vs N=2 comparisons
    if (case_a == case_b) and (order_a != order_b):
        pair_xlim = common_limits(
            arrays=[pair_data["delta"]],
            pad_fraction=DIFFERENCE_X_PAD_FRACTION,
            min_pad=MIN_DIFFERENCE_X_PAD,
            symmetric=True,
        )
    else:
        pair_xlim = difference_xlim

    ax.set_title(rf"$t={pair_data['t_plot']:.3f}$, $x={pair_data['x_plot']:.3f}$")
    configure_profile_axis(ax=ax, xlim=pair_xlim, xlabel=r"$\Delta u(\zeta)$")

    annotation_lines = [
        rf"$\max_\zeta |\Delta u|={pair_data['max_abs_delta']:.3e}$",
    ]

    ax.text(
        0.04,
        0.95,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.85,
        },
    )

    output_name = (
        f"{pair_data['name']}"
        f"_difference_t{TARGET_TIME:.3f}_x{TARGET_X:.3f}.pdf"
    ).replace(".", "p").replace("ppdf", ".pdf")

    output_path = OUTPUT_DIR / output_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", format='pdf')
    plt.close(fig)

    print(f"[saved] {output_path}")


# =============================================================================
# Main generation wrapper
# =============================================================================

def main() -> None:
    """
    Generate report-style zoom and difference figures for all requested pairs.
    """
    if not ROOT_DIR.exists():
        raise FileNotFoundError(
            f"ROOT_DIR does not exist:\n{ROOT_DIR}\n"
            "Edit ROOT_DIR at the top of the script."
        )

    pair_data = [build_pair_data(pair_spec) for pair_spec in PAIR_SPECS]
    profile_xlim, difference_xlim = build_common_axis_limits(pair_data)

    for data in pair_data:
        plot_pair_zoom(data, profile_xlim)
        plot_pair_difference(data, difference_xlim)

    print("[done] report-style dry/wet vertical-profile zoom figures generated.")


if __name__ == "__main__":
    main()