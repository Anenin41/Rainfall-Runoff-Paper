#!/usr/bin/env python3
"""
Standalone plotting script for comparing three recharge/friction runs.

This script extracts the plot_three_recharge_runs_momentum_velocity(...)
function from plotter.py and makes it independent from the main plotter
configuration system.

It creates two figures:
    1. discharge/momentum q = h*u_m over time
    2. velocity u_m over time

It accepts either:
    - summary CSVs with columns: time, mean_h, mean_u_m
    - history CSVs with columns: time, x, h, u_m
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# USER SETTINGS
# =============================================================================

plt.style.use("tableau-colorblind10")


@dataclass
class StandalonePlotConfig:
    """
    Minimal plotting configuration for the standalone comparison script.
    """

    output_dir: Path

    save_png: bool = True
    save_svg: bool = False
    save_pdf: bool = True
    png_dpi: int = 300
    show_figures: bool = False

    use_titles: bool = False
    show_grid: bool = True

    legend_font_size: int = 10
    axis_font_size: int = 11
    tick_font_size: int = 10


# Edit this output directory if needed.
CFG = StandalonePlotConfig(
    output_dir=Path(
        "/home/anenin/Documents/Git/thesis/model/processing/Ersoy/Comparison_Figures"
    )
)


# =============================================================================
# STYLING HELPERS
# =============================================================================

def plot_model_curve(
    ax: plt.Axes,
    x,
    y,
    order: int,
    label: str,
    overlapping: bool = True,
) -> None:
    """
    Plot one model/comparison curve using the preferred thesis styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to plot.
    x, y : array-like
        Data to plot.
    order : int
        Curve index: 0, 1, or 2.
    label : str
        Legend label.
    overlapping : bool
        If True, use the stronger distinguishable styles for overlapping curves.
        If False, use the same colours but simpler solid lines.
    """

    if overlapping:
        # Style for strongly overlapping curves.
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

    else:
        # Style for curves that are clearly separated already.
        if order == 0:
            color = "#0072B2"
            linestyle = "-"
            linewidth = 2.0
            zorder = 1

        elif order == 1:
            color = "#E69F00"
            linestyle = "-"
            linewidth = 2.0
            zorder = 2

        elif order == 2:
            color = "#000000"
            linestyle = "-"
            linewidth = 2.0
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


def apply_axis_style(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
) -> None:
    """
    Apply consistent thesis-style axis formatting.
    """

    ax.set_xlabel(xlabel, fontsize=CFG.axis_font_size)
    ax.set_ylabel(ylabel, fontsize=CFG.axis_font_size)

    if CFG.use_titles and title is not None:
        ax.set_title(title, fontsize=CFG.axis_font_size)

    ax.tick_params(axis="both", labelsize=CFG.tick_font_size)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if CFG.show_grid:
        ax.grid(True, alpha=0.25)

    ax.figure.tight_layout()


def save_figure(fig: plt.Figure, stem: str) -> None:
    """
    Save a figure in the configured output directory.
    """

    CFG.output_dir.mkdir(parents=True, exist_ok=True)

    if CFG.save_png:
        fig.savefig(
            CFG.output_dir / f"{stem}.png",
            dpi=CFG.png_dpi,
            bbox_inches="tight",
        )

    if CFG.save_svg:
        fig.savefig(
            CFG.output_dir / f"{stem}.svg",
            bbox_inches="tight",
            transparent=True,
        )

    if CFG.save_pdf:
        fig.savefig(
            CFG.output_dir / f"{stem}.pdf",
            bbox_inches="tight",
        )


def close_or_show(fig: plt.Figure) -> None:
    """
    Show figure interactively or close it.
    """

    if CFG.show_figures:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_recharge_time_series(
    path: str | Path,
    *,
    use_spatial_mean: bool = True,
    x_target: float | None = None,
) -> pd.DataFrame:
    """
    Load one CSV and convert it to a time series with columns:
        time, h, u_m, q

    Works with:
        1. summary CSVs: time, mean_h, mean_u_m
        2. history CSVs: time, x, h, u_m

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the CSV file.
    use_spatial_mean : bool
        If a history CSV is supplied, average over x at each time.
    x_target : float or None
        If use_spatial_mean=False, select the cell nearest this x location.

    Returns
    -------
    pandas.DataFrame
        Time-sorted dataframe with time, h, u_m, and q=h*u_m.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find CSV file:\n{path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]

    # Case 1: summary CSV
    if {"time", "mean_h", "mean_u_m"}.issubset(df.columns):
        out = df[["time", "mean_h", "mean_u_m"]].copy()
        out = out.rename(columns={"mean_h": "h", "mean_u_m": "u_m"})

    # Case 2: full spatial-history CSV
    elif {"time", "x", "h", "u_m"}.issubset(df.columns):
        if use_spatial_mean:
            out = (
                df.groupby("time", as_index=False)
                  .agg(h=("h", "mean"), u_m=("u_m", "mean"))
            )

        else:
            if x_target is None:
                raise ValueError(
                    "x_target must be provided when use_spatial_mean=False."
                )

            rows = []
            for _, group in df.groupby("time"):
                idx = (group["x"] - x_target).abs().idxmin()
                rows.append(group.loc[idx, ["time", "h", "u_m"]])

            out = pd.DataFrame(rows)

    else:
        raise ValueError(
            f"{path} does not look like a valid summary or history CSV.\n"
            f"Available columns: {list(df.columns)}"
        )

    out = out.sort_values("time").reset_index(drop=True)
    out["q"] = out["h"] * out["u_m"]

    return out


# =============================================================================
# MAIN PLOTTING FUNCTION
# =============================================================================

def plot_three_recharge_runs_momentum_velocity(
    csv_alpha0: str | Path,
    csv_alpha1: str | Path,
    csv_alpha2: str | Path,
    labels: tuple[str, str, str] = (
        r"$\alpha = 0$",
        r"$\alpha = 1$",
        r"$\alpha = 2$",
    ),
    filename: str = "momentum_velocity_three_alpha",
    x_target: float | None = None,
    use_spatial_mean: bool = True,
    overlapping: bool = True,
) -> None:
    """
    Plot discharge q = h*u_m and velocity u_m over time for three runs.

    Parameters
    ----------
    csv_alpha0, csv_alpha1, csv_alpha2 : str or pathlib.Path
        Paths to the three CSV files.
    labels : tuple[str, str, str]
        Legend labels for the three runs.
    filename : str
        Output filename stem.
    x_target : float or None
        Spatial point used if use_spatial_mean=False.
    use_spatial_mean : bool
        If True and history CSVs are supplied, average over x at each time.
    overlapping : bool
        If True, use stronger line styles for overlapping curves.
    """

    paths = [Path(csv) for csv in (csv_alpha0, csv_alpha1, csv_alpha2)]

    data = [
        load_recharge_time_series(
            path,
            use_spatial_mean=use_spatial_mean,
            x_target=x_target,
        )
        for path in paths
    ]

    # -------------------------------------------------------------------------
    # Figure 1: discharge q = h*u_m
    # -------------------------------------------------------------------------
    fig_q, ax_q = plt.subplots(figsize=(6.2, 4.2))

    for order, (df, label) in enumerate(zip(data, labels)):
        plot_model_curve(
            ax=ax_q,
            x=df["time"],
            y=df["q"],
            order=order,
            label=label,
            overlapping=overlapping,
        )

    apply_axis_style(
        ax_q,
        xlabel=r"Time [s]",
        ylabel=r"$q(t, x)$",
        title=None,
    )

    ax_q.legend(fontsize=CFG.legend_font_size, frameon=True)
    fig_q.tight_layout()
    save_figure(fig_q, f"{filename}_momentum")
    close_or_show(fig_q)

    # -------------------------------------------------------------------------
    # Figure 2: velocity u_m
    # -------------------------------------------------------------------------
    fig_u, ax_u = plt.subplots(figsize=(6.2, 4.2))

    for order, (df, label) in enumerate(zip(data, labels)):
        plot_model_curve(
            ax=ax_u,
            x=df["time"],
            y=df["u_m"],
            order=order,
            label=label,
            overlapping=overlapping,
        )

    apply_axis_style(
        ax_u,
        xlabel=r"Time [s]",
        ylabel=r"$u_m(t, x)$",
        title=None,
    )

    ax_u.legend(fontsize=CFG.legend_font_size, frameon=True)
    fig_u.tight_layout()
    save_figure(fig_u, f"{filename}_velocity")
    close_or_show(fig_u)

    # -------------------------------------------------------------------------
    # Figure 3: height h
    # -------------------------------------------------------------------------
    fig_h, ax_h = plt.subplots(figsize=(6.2, 4.2))

    for order, (df, label) in enumerate(zip(data, labels)):
        plot_model_curve(
            ax=ax_h,
            x=df["time"],
            y=df["h"],
            order=order,
            label=label,
            overlapping=True,
        )

    apply_axis_style(
        ax_h,
        xlabel=r"Time [s]",
        ylabel=r"$h(t, x)$",
        title=None,
    )

    ax_h.legend(fontsize=CFG.legend_font_size, frameon=True)
    fig_h.tight_layout()
    save_figure(fig_h, f"{filename}_height")
    close_or_show(fig_h)


# =============================================================================
# RUN SCRIPT
# =============================================================================

if __name__ == "__main__":

    # Edit this folder and filenames to match your actual files.
    COMPARISON_DIR = Path(
        "/home/anenin/Documents/Git/thesis/model/processing/Ersoy"
    ).expanduser().resolve()

    csv_alpha0 = COMPARISON_DIR / "ErsoyData0/recharge_swme_N0_constant_alpha0_field_history.csv"
    csv_alpha1 = COMPARISON_DIR / "ErsoyData1/recharge_swme_N0_constant_alpha1_field_history.csv"
    csv_alpha2 = COMPARISON_DIR / "ErsoyData2/recharge_swme_N0_constant_alpha2_field_history.csv"

    plot_three_recharge_runs_momentum_velocity(
        csv_alpha0=csv_alpha0,
        csv_alpha1=csv_alpha1,
        csv_alpha2=csv_alpha2,
        labels=(
            r"$\alpha = 0$",
            r"$\alpha = 1$",
            r"$\alpha = 2$",
        ),
        filename="alpha_comparison_momentum_velocity",
        use_spatial_mean=True,
        x_target=0.5,
        overlapping=False
    )

    print(f"Finished. Figures saved to:\n{CFG.output_dir}")