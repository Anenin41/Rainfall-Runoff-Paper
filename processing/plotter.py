# Packages
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import configparser
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ==============================================================================
# CONFIG
# ==============================================================================
CONFIG_FILE = Path("config.ini")


@dataclass
class PlotterConfig:
    """
    Central configuration container for the plotter, populated from config.ini.

    Attributes
    ----------
    project_root : Path | None
        Absolute path to the project root directory. When None the root is
        inferred as two levels above this script.
    results_subdir : Path
        Path to the results directory, relative to the project root.
    figures_subdir : Path
        Path to the output figures directory, relative to the project root.
        Created automatically if it does not exist.

    summary_filename : str
        Filename of the per-timestep summary CSV (spatial aggregates).
    history_filename : str
        Filename of the full spatial-history CSV (one row per cell per step).
    hyper_summary_filename : str
        Filename of the hyperbolicity summary CSV.
    hyper_history_filename : str
        Filename of the per-cell hyperbolicity history CSV.
    case_tag : str
        Optional prefix prepended to every saved figure filename, useful for
        distinguishing outputs from different simulation runs.

    save_png : bool
        Whether to save figures as PNG files.
    save_svg : bool
        Whether to save figures as SVG files.
    save_pdf : bool
        Whether to save figures as PDF files.
    png_dpi : int
        Resolution (dots per inch) used when saving PNG files.
    show_figures : bool
        If True, display each figure interactively instead of closing it after
        saving.

    use_titles : bool
        Whether to add a title to each axes.
    show_grid : bool
        Whether to draw a light background grid on each axes.
    line_width : float
        Default line width for all plot lines.
    legend_font_size : int
        Font size for legend text.
    axis_font_size : int
        Font size for axis labels and titles.
    tick_font_size : int
        Font size for tick labels.

    snapshot_times : list[float] | None
        Explicit list of simulation times at which to take spatial snapshots.
        When None, times are derived from ``snapshot_time_fractions`` instead.
    snapshot_time_fractions : list[float]
        Fractions of the total simulation duration used to pick snapshot times
        when ``snapshot_times`` is None (e.g. [0.0, 0.5, 1.0]).

    hyper_enabled : bool
        Master switch for all hyperbolicity diagnostics and plots.
    hyper_tol : float
        Tolerance threshold for classifying a cell as non-hyperbolic; cells
        whose maximum imaginary eigenvalue magnitude exceeds this value are
        flagged.
    heatmap_eps : float
        Small positive floor applied to the heatmap data before taking a
        logarithm, preventing log(0) when ``use_log_heatmap`` is True.
    use_log_heatmap : bool
        If True, render the hyperbolicity heatmap on a logarithmic colour scale.
    overlay_variable : str | None
        Name of a state variable (column in the history CSV) to overlay on the
        hyperbolicity snapshot plot. None disables the overlay.

    plot_mean_height : bool
        Enable the mean-height-vs-time time-series plot.
    plot_mean_velocity : bool
        Enable the mean-velocity-vs-time time-series plot.
    plot_height_envelope : bool
        Enable the height-envelope plot (mean ± min/max band).
    plot_height_snapshots : bool
        Enable spatial snapshots of the height field.
    plot_velocity_snapshots : bool
        Enable spatial snapshots of the velocity field.
    plot_moments : bool
        Enable time-series and snapshot plots for all detected moment columns.

    plot_hyper_max_imag_vs_time : bool
        Enable the global maximum imaginary eigenvalue vs. time plot.
    plot_hyper_nonhyperbolic_cells_vs_time : bool
        Enable the non-hyperbolic cell count / fraction vs. time plot.
    plot_hyper_worst_x_vs_time : bool
        Enable the x-location of the worst hyperbolicity violation vs. time.
    plot_hyper_heatmap : bool
        Enable the space-time heatmap of the imaginary eigenvalue magnitude.
    plot_hyper_binary_heatmap : bool
        Enable the binary (hyperbolic / non-hyperbolic) space-time heatmap.
    plot_hyper_snapshots : bool
        Enable spatial snapshots of the imaginary eigenvalue profile.
    plot_hyper_overlay : bool
        Enable the dual-axis overlay of hyperbolicity loss and a state variable.
    """

    project_root: Path | None
    results_subdir: Path
    figures_subdir: Path

    summary_filename: str
    history_filename: str
    hyper_summary_filename: str
    hyper_history_filename: str
    case_tag: str

    save_png: bool
    save_svg: bool
    save_pdf: bool
    png_dpi: int
    show_figures: bool

    use_titles: bool
    show_grid: bool
    line_width: float
    legend_font_size: int
    axis_font_size: int
    tick_font_size: int

    snapshot_times: list[float] | None
    snapshot_time_fractions: list[float]

    hyper_enabled: bool
    hyper_tol: float
    heatmap_eps: float
    use_log_heatmap: bool
    overlay_variable: str | None

    plot_mean_height: bool
    plot_mean_velocity: bool
    plot_height_envelope: bool
    plot_height_snapshots: bool
    plot_velocity_snapshots: bool
    plot_moments: bool

    plot_hyper_max_imag_vs_time: bool
    plot_hyper_nonhyperbolic_cells_vs_time: bool
    plot_hyper_worst_x_vs_time: bool
    plot_hyper_heatmap: bool
    plot_hyper_binary_heatmap: bool
    plot_hyper_snapshots: bool
    plot_hyper_overlay: bool


def _parse_float_list(raw: str) -> list[float]:
    """
    Parse a comma-separated string of floats into a Python list.

    Empty strings and strings containing only whitespace return an empty list.
    Individual tokens are stripped of surrounding whitespace before conversion.

    Parameters
    ----------
    raw : str
        Raw config value, e.g. ``"0.0, 0.25, 0.5, 1.0"``.

    Returns
    -------
    list[float]
        Parsed floating-point values in the order they appear in ``raw``.
    """
    raw = raw.strip()
    if not raw:
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_optional_string(raw: str) -> str | None:
    """
    Convert a raw config string to either a non-empty string or None.

    Useful for config values that are semantically "optional": a blank or
    whitespace-only entry is treated as absent (None), while any non-empty
    value is returned as-is after stripping.

    Parameters
    ----------
    raw : str
        Raw config value.

    Returns
    -------
    str | None
        The stripped string, or None if it was empty.
    """
    raw = raw.strip()
    return raw if raw else None


def load_plotter_config(config_file: Path = CONFIG_FILE) -> PlotterConfig:
    """
    Read ``config.ini`` and return a fully populated :class:`PlotterConfig`.

    The INI file must contain the following sections: ``[paths]``, ``[files]``,
    ``[output]``, ``[style]``, ``[snapshots]``, ``[hyperbolicity]``, and
    ``[plots]``. Missing sections or keys will raise a ``KeyError`` from the
    underlying :mod:`configparser`.

    Parameters
    ----------
    config_file : Path
        Path to the INI configuration file. Defaults to ``config.ini`` in the
        current working directory.

    Returns
    -------
    PlotterConfig
        Dataclass instance containing all settings parsed from the file.

    Raises
    ------
    FileNotFoundError
        If ``config_file`` does not exist or cannot be read.
    """
    parser = configparser.ConfigParser()
    if not parser.read(config_file):
        raise FileNotFoundError(f"Could not read plotter config:\n{config_file.resolve()}")

    paths = parser["paths"]
    files = parser["files"]
    output = parser["output"]
    style = parser["style"]
    snapshots = parser["snapshots"]
    hyper = parser["hyperbolicity"]
    plots = parser["plots"]

    project_root_raw = paths.get("project_root", "").strip()
    project_root = Path(project_root_raw).expanduser().resolve() if project_root_raw else None

    snapshot_times = _parse_float_list(snapshots.get("snapshot_times", ""))
    if not snapshot_times:
        snapshot_times = None

    overlay_variable = _parse_optional_string(hyper.get("overlay_variable", ""))

    return PlotterConfig(
        project_root=project_root,
        results_subdir=Path(paths.get("results_subdir")),
        figures_subdir=Path(paths.get("figures_subdir")),

        summary_filename=files.get("summary_filename"),
        history_filename=files.get("history_filename"),
        hyper_summary_filename=files.get("hyper_summary_filename"),
        hyper_history_filename=files.get("hyper_history_filename"),
        case_tag=files.get("case_tag", "").strip(),

        save_png=output.getboolean("save_png"),
        save_svg=output.getboolean("save_svg"),
        save_pdf=output.getboolean("save_pdf"),
        png_dpi=output.getint("png_dpi"),
        show_figures=output.getboolean("show_figures"),

        use_titles=style.getboolean("use_titles"),
        show_grid=style.getboolean("show_grid"),
        line_width=style.getfloat("line_width"),
        legend_font_size=style.getint("legend_font_size"),
        axis_font_size=style.getint("axis_font_size"),
        tick_font_size=style.getint("tick_font_size"),

        snapshot_times=snapshot_times,
        snapshot_time_fractions=_parse_float_list(
            snapshots.get("snapshot_time_fractions", "0.0,0.25,0.5,0.75,1.0")
        ),

        hyper_enabled=hyper.getboolean("enabled"),
        hyper_tol=hyper.getfloat("hyper_tol"),
        heatmap_eps=hyper.getfloat("heatmap_eps"),
        use_log_heatmap=hyper.getboolean("use_log_heatmap"),
        overlay_variable=overlay_variable,

        plot_mean_height=plots.getboolean("plot_mean_height"),
        plot_mean_velocity=plots.getboolean("plot_mean_velocity"),
        plot_height_envelope=plots.getboolean("plot_height_envelope"),
        plot_height_snapshots=plots.getboolean("plot_height_snapshots"),
        plot_velocity_snapshots=plots.getboolean("plot_velocity_snapshots"),
        plot_moments=plots.getboolean("plot_moments"),

        plot_hyper_max_imag_vs_time=plots.getboolean("plot_hyper_max_imag_vs_time"),
        plot_hyper_nonhyperbolic_cells_vs_time=plots.getboolean("plot_hyper_nonhyperbolic_cells_vs_time"),
        plot_hyper_worst_x_vs_time=plots.getboolean("plot_hyper_worst_x_vs_time"),
        plot_hyper_heatmap=plots.getboolean("plot_hyper_heatmap"),
        plot_hyper_binary_heatmap=plots.getboolean("plot_hyper_binary_heatmap"),
        plot_hyper_snapshots=plots.getboolean("plot_hyper_snapshots"),
        plot_hyper_overlay=plots.getboolean("plot_hyper_overlay"),
    )


CFG = load_plotter_config()


# =============================================================================
# PATHS
# =============================================================================

def get_project_root() -> Path:
    """
    Resolve the absolute path to the project root directory.

    Uses ``CFG.project_root`` when explicitly set in the config; otherwise
    falls back to two directory levels above this script file, which assumes
    the conventional layout ``<project_root>/src/plotter.py`` (or similar).

    Returns
    -------
    Path
        Absolute path to the project root.
    """
    if CFG.project_root is not None:
        return CFG.project_root
    return Path(__file__).resolve().parent.parent


ROOT = get_project_root()
RESULTS_DIR = ROOT / CFG.results_subdir
FIGURES_DIR = ROOT / CFG.figures_subdir
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = RESULTS_DIR / CFG.summary_filename
HISTORY_FILE = RESULTS_DIR / CFG.history_filename
HYPER_SUMMARY_FILE = RESULTS_DIR / CFG.hyper_summary_filename
HYPER_HISTORY_FILE = RESULTS_DIR / CFG.hyper_history_filename


# =============================================================================
# HELPERS
# =============================================================================

def require_file(path: Path) -> None:
    """
    Assert that a required input file exists, raising a clear error if not.

    Parameters
    ----------
    path : Path
        Path to the file that must exist.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not point to an existing file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")


def require_columns(df: pd.DataFrame,
                    required: Iterable[str],
                    label: str) -> None:
    """
    Assert that a DataFrame contains a set of required column names.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    required : Iterable[str]
        Column names that must all be present in ``df``.
    label : str
        Human-readable name for the DataFrame (used in the error message).

    Raises
    ------
    ValueError
        If one or more required columns are absent, listing both the missing
        columns and the columns that are actually available.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def case_stem(name: str) -> str:
    """
    Prepend the configured case tag to a figure filename stem.

    If ``CFG.case_tag`` is non-empty the returned stem is
    ``"<tag>_<name>"``, otherwise ``name`` is returned unchanged. This allows
    multiple simulation runs to write figures into the same directory without
    overwriting each other.

    Parameters
    ----------
    name : str
        Base filename stem (no extension).

    Returns
    -------
    str
        Possibly-prefixed filename stem.
    """
    return f"{CFG.case_tag}_{name}" if CFG.case_tag else name


def save_figure(fig: plt.Figure, stem: str) -> None:
    """
    Save a matplotlib figure to disk in the configured format(s).

    The output path is ``FIGURES_DIR / <case_stem(stem)>.<ext>``. PNG and PDF
    outputs are written independently according to ``CFG.save_png`` and
    ``CFG.save_pdf``. Both formats use ``bbox_inches="tight"`` to avoid
    clipping.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save.
    stem : str
        Base filename stem (without extension). The case tag is applied via
        :func:`case_stem` before writing.
    """
    stem = case_stem(stem)
    if CFG.save_png:
        fig.savefig(FIGURES_DIR / f"{stem}.png",
                    dpi=CFG.png_dpi,
                    bbox_inches="tight")
    if CFG.save_svg:
        fig.savefig(FIGURES_DIR / f"{stem}.svg",
                    bbox_inches="tight", transparent=True)
    if CFG.save_pdf:
        fig.savefig(FIGURES_DIR / f"{stem}.pdf",
                    bbox_inches="tight")


def apply_axis_style(ax: plt.Axes,
                     xlabel: str,
                     ylabel: str,
                     title: str | None = None) -> None:
    """
    Apply the global style settings from ``CFG`` to a matplotlib axes object.

    This function centralises all cosmetic decisions so that every plot in the
    module has a consistent appearance. It sets axis labels, tick sizes, removes
    the top and right spines, optionally adds a title and grid, and calls
    ``tight_layout`` on the parent figure.

    Parameters
    ----------
    ax : plt.Axes
        The axes to style.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    title : str | None
        Axes title. Only rendered when ``CFG.use_titles`` is True; ignored
        entirely if None.
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


def close_or_show(fig: plt.Figure) -> None:
    """
    Either display or close a figure depending on the config.

    When ``CFG.show_figures`` is True the figure is shown interactively via
    ``plt.show()``, which blocks until the window is closed. Otherwise the
    figure is closed immediately to free memory.

    Parameters
    ----------
    fig : plt.Figure
        The figure to show or close.
    """
    if CFG.show_figures:
        plt.show()
    else:
        plt.close(fig)


def nearest_times(df: pd.DataFrame,
                  target_times: Iterable[float]) -> list[float]:
    """
    Map a collection of target times to the nearest times available in a DataFrame.

    For each value in ``target_times`` the function finds the closest entry in
    the unique ``"time"`` column of ``df`` (by absolute difference). Duplicates
    in the result are removed and the output is returned in ascending order.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``"time"`` column.
    target_times : Iterable[float]
        Desired simulation times.

    Returns
    -------
    list[float]
        Sorted, deduplicated list of times from ``df["time"]`` that are closest
        to each requested target.
    """
    available = np.array(sorted(df["time"].unique()), dtype=float)
    selected = []

    for t in target_times:
        idx = np.argmin(np.abs(available - t))
        selected.append(float(available[idx]))

    return sorted(set(selected))


def choose_snapshot_times(df: pd.DataFrame) -> list[float]:
    """
    Determine which simulation times to use for spatial snapshot plots.

    If ``CFG.snapshot_times`` is set (i.e. the user provided an explicit list
    in the config), those times are snapped to the nearest available entries in
    ``df`` via :func:`nearest_times`. Otherwise, times are derived by
    multiplying the maximum time in ``df`` by each fraction in
    ``CFG.snapshot_time_fractions`` and again snapping to the nearest available
    entries.

    Parameters
    ----------
    df : pd.DataFrame
        Spatial-history DataFrame with a ``"time"`` column.

    Returns
    -------
    list[float]
        Sorted list of times at which snapshots should be plotted.
    """
    if CFG.snapshot_times is not None:
        return nearest_times(df, CFG.snapshot_times)

    t_end = float(df["time"].max())
    target_times = [frac * t_end for frac in CFG.snapshot_time_fractions]
    return nearest_times(df, target_times)


def detect_moment_columns(summary_df: pd.DataFrame,
                          history_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Detect and return paired moment columns from the summary and history DataFrames.

    Moment columns follow the naming convention ``mean_a<N>`` in the summary
    CSV and ``a<N>`` in the history CSV, where ``N`` is a non-negative integer.
    Both lists are sorted numerically by ``N``.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Per-timestep summary DataFrame (e.g. spatially averaged quantities).
    history_df : pd.DataFrame
        Full spatial-history DataFrame.

    Returns
    -------
    tuple[list[str], list[str]]
        A pair ``(summary_moments, history_moments)`` where each list contains
        the detected column names in ascending moment order.

    Raises
    ------
    ValueError
        If the number of detected moment columns differs between the two
        DataFrames, indicating an inconsistency in the data files.
    """
    summary_moments = sorted(
        [col for col in summary_df.columns if re.fullmatch(r"mean_a\d+", col)],
        key=lambda s: int(s.replace("mean_a", ""))
    )
    history_moments = sorted(
        [col for col in history_df.columns if re.fullmatch(r"a\d+", col)],
        key=lambda s: int(s.replace("a", ""))
    )

    if len(summary_moments) != len(history_moments):
        raise ValueError(
            "Mismatch between summary and history moment columns.\n"
            f"Summary: {summary_moments}\n"
            f"History: {history_moments}"
        )

    return summary_moments, history_moments


def first_nonhyperbolic_time(hyper_summary_df: pd.DataFrame) -> float | None:
    """
    Return the earliest time at which any non-hyperbolic cell was detected.

    Scans ``hyper_summary_df`` for rows where ``num_nonhyperbolic_cells > 0``
    and returns the time of the first such row (in the order the DataFrame is
    sorted, which should be ascending by time).

    Parameters
    ----------
    hyper_summary_df : pd.DataFrame
        Hyperbolicity summary DataFrame. Must contain columns ``"time"`` and
        ``"num_nonhyperbolic_cells"``.

    Returns
    -------
    float | None
        The first simulation time with at least one non-hyperbolic cell, or
        None if the simulation remained fully hyperbolic throughout.
    """
    bad = hyper_summary_df[hyper_summary_df["num_nonhyperbolic_cells"] > 0]
    if bad.empty:
        return None
    return float(bad.iloc[0]["time"])


def worst_time(hyper_summary_df: pd.DataFrame) -> float:
    """
    Return the simulation time at which the hyperbolicity violation was worst.

    "Worst" is defined as the global maximum of ``max_abs_imag_eig`` across all
    timesteps, i.e. the moment when the largest imaginary eigenvalue magnitude
    was recorded anywhere in the domain.

    Parameters
    ----------
    hyper_summary_df : pd.DataFrame
        Hyperbolicity summary DataFrame. Must contain columns ``"time"`` and
        ``"max_abs_imag_eig"``.

    Returns
    -------
    float
        Simulation time corresponding to the peak hyperbolicity violation.
    """
    idx = hyper_summary_df["max_abs_imag_eig"].idxmax()
    return float(hyper_summary_df.loc[idx, "time"])

def load_comparison_config(config_file: Path = CONFIG_FILE):
    parser = configparser.ConfigParser()
    parser.read(config_file)

    if "comparison" not in parser:
        return None

    comp = parser["comparison"]

    return {
        "enabled": comp.getboolean("plot_alpha_comparison", fallback=False),
        "alpha0_file": comp.get("alpha0_file"),
        "alpha1_file": comp.get("alpha1_file"),
        "alpha2_file": comp.get("alpha2_file"),
        "labels": (
            comp.get("alpha0_label", fallback=r"$\alpha = 0$"),
            comp.get("alpha1_label", fallback=r"$\alpha = 1$"),
            comp.get("alpha2_label", fallback=r"$\alpha = 2$"),
        ),
        "filename": comp.get("filename", fallback="alpha_comparison_momentum_velocity"),
        "use_spatial_mean": comp.getboolean("use_spatial_mean", fallback=True),
        "x_target": comp.getfloat("x_target", fallback=0.5),
    }

# =============================================================================
# INPUT
# =============================================================================

def read_solution_data():
    """
    Load, validate, and sort the solution summary and history CSV files.

    Reads the files pointed to by ``SUMMARY_FILE`` and ``HISTORY_FILE``,
    checks that all required columns are present, sorts both DataFrames into a
    canonical order, and detects any moment columns (``a<N>`` / ``mean_a<N>``).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]
        A four-element tuple ``(summary_df, history_df, summary_moments,
        history_moments)`` where:

        - ``summary_df`` is sorted by ``(time, step)``
        - ``history_df`` is sorted by ``(time, x)``
        - ``summary_moments`` lists ``mean_a<N>`` columns in ascending order
        - ``history_moments`` lists the corresponding ``a<N>`` columns

    Raises
    ------
    FileNotFoundError
        If either CSV file is missing.
    ValueError
        If required columns are absent or the moment columns are mismatched.
    """
    require_file(SUMMARY_FILE)
    require_file(HISTORY_FILE)

    summary_df = pd.read_csv(SUMMARY_FILE)
    history_df = pd.read_csv(HISTORY_FILE)

    require_columns(
        summary_df,
        ["step", "time", "mean_h", "mean_u_m", "min_h", "max_h"],
        "Summary CSV",
    )
    require_columns(
        history_df,
        ["step", "time", "x", "h", "u_m"],
        "History CSV",
    )

    summary_df = summary_df.sort_values(["time", "step"]).reset_index(drop=True)
    history_df = history_df.sort_values(["time", "x"]).reset_index(drop=True)

    summary_moments, history_moments = detect_moment_columns(summary_df, history_df)

    return summary_df, history_df, summary_moments, history_moments


def read_hyperbolicity_data():
    """
    Load, validate, and sort the hyperbolicity summary and history CSV files.

    Returns immediately with ``(None, None)`` if ``CFG.hyper_enabled`` is
    False. Otherwise reads the files pointed to by ``HYPER_SUMMARY_FILE`` and
    ``HYPER_HISTORY_FILE``, validates their columns, and sorts both DataFrames
    into a canonical order.

    Returns
    -------
    tuple[pd.DataFrame | None, pd.DataFrame | None]
        ``(hyper_summary_df, hyper_history_df)`` sorted by ``(time, step)``
        and ``(time, x)`` respectively, or ``(None, None)`` if hyperbolicity
        analysis is disabled.

    Raises
    ------
    FileNotFoundError
        If either hyperbolicity CSV file is missing.
    ValueError
        If required columns are absent from either file.
    """
    if not CFG.hyper_enabled:
        return None, None

    require_file(HYPER_SUMMARY_FILE)
    require_file(HYPER_HISTORY_FILE)

    hyper_summary_df = pd.read_csv(HYPER_SUMMARY_FILE)
    hyper_history_df = pd.read_csv(HYPER_HISTORY_FILE)

    require_columns(
        hyper_summary_df,
        [
            "step",
            "time",
            "num_nonhyperbolic_cells",
            "fraction_nonhyperbolic_cells",
            "max_abs_imag_eig",
            "worst_cell_index",
            "worst_x",
        ],
        "Hyper summary CSV",
    )
    require_columns(
        hyper_history_df,
        [
            "step",
            "time",
            "cell_index",
            "x",
            "max_abs_imag_eig",
            "is_hyperbolic",
        ],
        "Hyper history CSV",
    )

    hyper_summary_df = hyper_summary_df.sort_values(["time", "step"]).reset_index(drop=True)
    hyper_history_df = hyper_history_df.sort_values(["time", "x"]).reset_index(drop=True)

    return hyper_summary_df, hyper_history_df


# =============================================================================
# STANDARD PLOTS
# =============================================================================

def plot_time_series(summary_df: pd.DataFrame,
                     column: str,
                     ylabel: str,
                     filename: str,
                     title: str | None = None) -> None:
    """
    Plot a single quantity from the summary DataFrame as a function of time.

    Creates a simple line plot of ``summary_df[column]`` vs. ``summary_df["time"]``,
    applies the global axis style, saves the figure, then closes or shows it.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Per-timestep summary DataFrame. Must contain ``"time"`` and ``column``.
    column : str
        Name of the column to plot on the y-axis.
    ylabel : str
        Y-axis label (may include LaTeX math).
    filename : str
        Output filename stem (no extension). The case tag and file extension
        are appended automatically by :func:`save_figure`.
    title : str | None
        Axes title. Only shown when ``CFG.use_titles`` is True.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(summary_df["time"], summary_df[column], linewidth=CFG.line_width)
    apply_axis_style(ax, xlabel="Time", ylabel=ylabel, title=title)
    save_figure(fig, filename)
    close_or_show(fig)


def plot_height_envelope(summary_df: pd.DataFrame) -> None:
    """
    Plot the mean water height over time with a shaded min/max envelope.

    Draws the mean height as a solid line and fills the region between the
    minimum and maximum height with a semi-transparent band, giving a quick
    visual summary of the spatial spread of the water surface at each timestep.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Per-timestep summary DataFrame. Must contain ``"time"``, ``"mean_h"``,
        ``"min_h"``, and ``"max_h"``.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(summary_df["time"], summary_df["mean_h"],
            linewidth=CFG.line_width, label="Mean")
    ax.fill_between(
        summary_df["time"],
        summary_df["min_h"],
        summary_df["max_h"],
        alpha=0.20,
        label="Min/Max range",
    )
    ax.legend(fontsize=CFG.legend_font_size, frameon=False)
    apply_axis_style(ax, xlabel="Time", ylabel="h", title="Height statistics")
    save_figure(fig, "height_envelope")
    close_or_show(fig)


def plot_snapshots(history_df: pd.DataFrame,
                   value_column: str,
                   ylabel: str,
                   filename: str,
                   title: str | None = None) -> None:
    """
    Plot spatial profiles of a field variable at several selected times.

    Snapshot times are determined by :func:`choose_snapshot_times`. Each
    snapshot is plotted as a separate labelled line; rows are matched to a
    given time using ``np.isclose`` to handle floating-point imprecision.

    Parameters
    ----------
    history_df : pd.DataFrame
        Full spatial-history DataFrame. Must contain ``"time"``, ``"x"``, and
        ``value_column``.
    value_column : str
        Name of the field variable column to plot on the y-axis.
    ylabel : str
        Y-axis label (may include LaTeX math).
    filename : str
        Output filename stem (no extension).
    title : str | None
        Axes title. Only shown when ``CFG.use_titles`` is True.
    """
    chosen_times = choose_snapshot_times(history_df)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for t in chosen_times:
        snap = history_df[np.isclose(history_df["time"], t)].sort_values("x")
        ax.plot(
            snap["x"],
            snap[value_column],
            linewidth=CFG.line_width,
            label=fr"$t={t:.3f}$"
        )

    ax.legend(fontsize=CFG.legend_font_size, frameon=False)
    apply_axis_style(ax, xlabel="x", ylabel=ylabel, title=title)
    save_figure(fig, filename)
    close_or_show(fig)

def reconstruct_velocity_profile(
    row : pd.Series,
    zeta_points : np.ndarray,
    ) -> np.ndarray:
    """
    Reconstruct the full vertical velocity profile u(zeta) from one history row.

    The history CSV already stores primitive variables:
        x, h, u_m, a1, a2, ...
    so the profile can be rebuilt directly from the SWME ansatz
        u(zeta) = u_m + sum_j a_j * phi_j(zeta),
    where phi_j are the scaled Legendre basis functions used by the solver.

    Parameters
    ----------
    row : pd.Series
        One row from the history DataFrame, corresponding to a fixed (time, x).
        Must contain at least ``u_m`` and may optionally contain ``a_1``,
        ``a_2``, ``a_3``.
    zeta_points : np.ndarray
        Points in the projected vertical coordinate in [0, 1].

    Returns
    -------
    np.ndarray
        Reconstructed velocity profile evaluated at ``zeta_points``.
    """
    zeta = np.asarray(zeta_points, dtype = float)
    u = np.full_like(zeta, float(row["u_m"]), dtype = float)

    # Check how many moment coefficients in dataset
    if "a1" in row.index:
        u += float(row["a1"]) * (1.0 - 2.0 * zeta)
    if "a2" in row.index:
        u += float(row["a2"]) * (1.0 - 6.0 * zeta + 6.0 * zeta**2)
    if "a3" in row.index:
        u += float(row["a3"]) * (1.0 - 12.0 * zeta + 30.0 * zeta**2 - 20.0 * zeta**3)
    
    return u

def plot_velocity_profile_snapshots(
    history_df : pd.DataFrame,
    x_target : float, 
    zeta_points : np.ndarray | None = None,
    filename : str = "velocity_profile_snapshots",
    title : str | None = None,    
    ) -> None:
    """
    Plot the reconstructed vertical velocity profiles u(zeta) at selected times.

    For each selected time, the function finds the history row whose x-position
    is closest to ``x_target`` and reconstructs the full vertical velocity
    profile from ``u_m``, ``a1``, ...

    Parameters
    ----------
    history_df : pd.Dataframe
        Full spatial-history DataFrame. Must contain ``time``, ``x``, ``u_m``
        and optionally moment columns ``a1``, ...
    x_target : float
        Spatial position at which the vertical profile should be plotted.
    zeta_points : np.ndarray | None
        Points in the projected vertical coordinate zeta in [0, 1]. If None, a 
        default uniform grid of 200 points is used.
    filename : str
        Output filename stem (no extension).
    title : str | None
        Axes title. Only shown when ``CFG.use_titles`` is True.
    """
    require_columns(history_df, ["time", "x", "u_m"], "History CSV")

    if zeta_points is None:
        zeta_points = np.linspace(0.0, 1.0, 200)

    chosen_times = choose_snapshot_times(history_df)

    available_x = np.array(sorted(history_df["x"].unique()), dtype = float)
    x_plot = float(available_x[np.argmin(np.abs(available_x - x_target))])

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    for t in chosen_times:
        snap = history_df[np.isclose(history_df["time"], t)].sort_values("x")

        if snap.empty:
            continue

        x_values = snap["x"].to_numpy(dtype = float)
        row_idx = int(np.argmin(np.abs(x_values - x_plot)))
        row = snap.iloc[row_idx]

        u_profile = reconstruct_velocity_profile(row, zeta_points)

        ax.plot(
            u_profile,
            zeta_points,
            linewidth = CFG.line_width,
            label = fr"$t={t:.3f}$",
        )

    ax.legend(fontsize = CFG.legend_font_size, frameon = False)
    apply_axis_style(
        ax,
        xlabel=r"$u(\zeta)$",
        ylabel=r"$\zeta$",
        title = title or rf"Velocity profiles at $x \approx {x_plot:.3f}$",
    )
    save_figure(fig, filename)
    close_or_show(fig)


# =============================================================================
# HYPERBOLICITY PLOTS
# =============================================================================

def plot_max_imag_vs_time(hyper_summary_df: pd.DataFrame) -> None:
    """
    Plot the global maximum imaginary eigenvalue magnitude over time.

    Shows ``max_abs_imag_eig`` (the worst hyperbolicity violation in the entire
    domain at each step) as a line, with a dashed horizontal line at
    ``CFG.hyper_tol`` to indicate the threshold below which the system is
    considered hyperbolic.

    Parameters
    ----------
    hyper_summary_df : pd.DataFrame
        Hyperbolicity summary DataFrame. Must contain ``"time"`` and
        ``"max_abs_imag_eig"``.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(hyper_summary_df["time"],
            hyper_summary_df["max_abs_imag_eig"],
            linewidth=CFG.line_width)
    ax.axhline(CFG.hyper_tol, linestyle="--", linewidth=1.0, label="Tolerance")
    ax.legend(fontsize=CFG.legend_font_size, frameon=False)
    apply_axis_style(
        ax,
        xlabel="Time",
        ylabel=r"$\max_x \max_j |\Im(\lambda_j)|$",
        title="Maximum imaginary part of eigenvalues",
    )
    save_figure(fig, "hyper_max_imag_vs_time")
    close_or_show(fig)


def plot_nonhyperbolic_cells_vs_time(hyper_summary_df: pd.DataFrame) -> None:
    """
    Plot the count and fraction of non-hyperbolic cells over time.

    Draws two lines on the same axes: the raw number of cells whose
    imaginary eigenvalue magnitude exceeds ``CFG.hyper_tol``, and the
    corresponding fraction of the total cell count.

    Parameters
    ----------
    hyper_summary_df : pd.DataFrame
        Hyperbolicity summary DataFrame. Must contain ``"time"``,
        ``"num_nonhyperbolic_cells"``, and ``"fraction_nonhyperbolic_cells"``.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(hyper_summary_df["time"],
            hyper_summary_df["num_nonhyperbolic_cells"],
            linewidth=CFG.line_width,
            label="Count")
    ax.plot(hyper_summary_df["time"],
            hyper_summary_df["fraction_nonhyperbolic_cells"],
            linewidth=CFG.line_width,
            label="Fraction")
    ax.legend(fontsize=CFG.legend_font_size, frameon=False)
    apply_axis_style(
        ax,
        xlabel="Time",
        ylabel="Non-hyperbolic cells",
        title="Extent of hyperbolicity loss",
    )
    save_figure(fig, "hyper_nonhyperbolic_cells_vs_time")
    close_or_show(fig)


def plot_worst_location_vs_time(hyper_summary_df: pd.DataFrame) -> None:
    """
    Plot the x-location of the most severe hyperbolicity violation over time.

    Tracks how the spatial position of the worst cell (largest imaginary
    eigenvalue) evolves through the simulation.

    Parameters
    ----------
    hyper_summary_df : pd.DataFrame
        Hyperbolicity summary DataFrame. Must contain ``"time"`` and
        ``"worst_x"``.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(hyper_summary_df["time"],
            hyper_summary_df["worst_x"],
            linewidth=CFG.line_width)
    apply_axis_style(
        ax,
        xlabel="Time",
        ylabel="x-location",
        title="Location of strongest violation",
    )
    save_figure(fig, "hyper_worst_x_vs_time")
    close_or_show(fig)


def plot_hyperbolicity_heatmap(hyper_history_df: pd.DataFrame) -> None:
    """
    Render a space-time heatmap of the imaginary eigenvalue magnitude.

    Pivots the history DataFrame into a 2-D array (rows = time, columns = x)
    of ``max_abs_imag_eig`` values and displays it as a colour mesh. When
    ``CFG.use_log_heatmap`` is True the colour scale is logarithmic and values
    are floored at ``CFG.heatmap_eps`` before taking the log.

    Parameters
    ----------
    hyper_history_df : pd.DataFrame
        Per-cell hyperbolicity history. Must contain ``"time"``, ``"x"``, and
        ``"max_abs_imag_eig"``.
    """
    pivot = hyper_history_df.pivot(index="time", columns="x", values="max_abs_imag_eig")
    times = pivot.index.to_numpy(dtype=float)
    xs = pivot.columns.to_numpy(dtype=float)
    Z = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    if CFG.use_log_heatmap:
        Zp = np.maximum(Z, CFG.heatmap_eps)
        pcm = ax.pcolormesh(
            xs,
            times,
            Zp,
            shading="auto",
            norm=LogNorm(vmin=max(np.nanmin(Zp), CFG.heatmap_eps),
                         vmax=np.nanmax(Zp)),
        )
        cbar_label = r"$\max_j |\Im(\lambda_j)|$ (log scale)"
    else:
        pcm = ax.pcolormesh(xs, times, Z, shading="auto")
        cbar_label = r"$\max_j |\Im(\lambda_j)|$"

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(cbar_label, fontsize=CFG.axis_font_size)

    apply_axis_style(
        ax,
        xlabel="x",
        ylabel="Time",
        title="Space-time map of hyperbolicity loss",
    )
    save_figure(fig, "hyper_heatmap")
    close_or_show(fig)


def plot_binary_hyperbolicity_heatmap(hyper_history_df: pd.DataFrame) -> None:
    """
    Render a binary space-time heatmap marking non-hyperbolic cells.

    Pivots the ``is_hyperbolic`` flag and inverts it so that 1 indicates a
    non-hyperbolic cell and 0 indicates a hyperbolic cell. The resulting colour
    mesh provides an unambiguous view of where and when hyperbolicity is lost.

    Parameters
    ----------
    hyper_history_df : pd.DataFrame
        Per-cell hyperbolicity history. Must contain ``"time"``, ``"x"``, and
        ``"is_hyperbolic"``.
    """
    pivot = hyper_history_df.pivot(index="time", columns="x", values="is_hyperbolic")
    times = pivot.index.to_numpy(dtype=float)
    xs = pivot.columns.to_numpy(dtype=float)
    Z = 1.0 - pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    pcm = ax.pcolormesh(xs, times, Z, shading="auto", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("1 = non-hyperbolic", fontsize=CFG.axis_font_size)

    apply_axis_style(
        ax,
        xlabel="x",
        ylabel="Time",
        title="Binary hyperbolicity map",
    )
    save_figure(fig, "hyper_binary_heatmap")
    close_or_show(fig)


def plot_hyperbolicity_snapshots(hyper_summary_df: pd.DataFrame,
                                 hyper_history_df: pd.DataFrame) -> None:
    """
    Plot spatial profiles of the imaginary eigenvalue magnitude at key times.

    Always includes the initial time (t = 0) and the time of the worst global
    violation (:func:`worst_time`). If any non-hyperbolic cells were detected,
    the first such time (:func:`first_nonhyperbolic_time`) is also included. A
    dashed horizontal line at ``CFG.hyper_tol`` marks the hyperbolicity
    threshold.

    Parameters
    ----------
    hyper_summary_df : pd.DataFrame
        Hyperbolicity summary DataFrame (used to identify key times).
    hyper_history_df : pd.DataFrame
        Per-cell hyperbolicity history. Must contain ``"time"``, ``"x"``, and
        ``"max_abs_imag_eig"``.
    """
    chosen = [0.0, worst_time(hyper_summary_df)]
    t_bad = first_nonhyperbolic_time(hyper_summary_df)
    if t_bad is not None:
        chosen.append(t_bad)

    chosen_times = nearest_times(hyper_history_df, chosen)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for t in chosen_times:
        snap = hyper_history_df[np.isclose(hyper_history_df["time"], t)].sort_values("x")
        ax.plot(
            snap["x"],
            snap["max_abs_imag_eig"],
            linewidth=CFG.line_width,
            label=fr"$t={t:.3f}$"
        )

    ax.axhline(CFG.hyper_tol, linestyle="--", linewidth=1.0, label="Tolerance")
    ax.legend(fontsize=CFG.legend_font_size, frameon=False)
    apply_axis_style(
        ax,
        xlabel="x",
        ylabel=r"$\max_j |\Im(\lambda_j)|$",
        title="Spatial snapshots of hyperbolicity loss",
    )
    save_figure(fig, "hyper_snapshots")
    close_or_show(fig)


def plot_overlay_at_worst_time(summary_df: pd.DataFrame,
                               history_df: pd.DataFrame,
                               hyper_summary_df: pd.DataFrame,
                               hyper_history_df: pd.DataFrame) -> None:
    """
    Overlay a state variable and the imaginary eigenvalue profile at the worst time.

    Produces a dual-axis line plot at the simulation time where the global
    hyperbolicity violation was greatest (see :func:`worst_time`). The left
    y-axis shows ``max_abs_imag_eig`` and the right y-axis shows the state
    variable named by ``CFG.overlay_variable``. Returns immediately without
    plotting if ``CFG.overlay_variable`` is None or if the column is not found
    in ``history_df``.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Per-timestep summary DataFrame (not directly used but kept for API
        consistency).
    history_df : pd.DataFrame
        Full spatial-history DataFrame. Must contain ``"time"``, ``"x"``, and
        the column named by ``CFG.overlay_variable``.
    hyper_summary_df : pd.DataFrame
        Hyperbolicity summary DataFrame (used to find the worst time).
    hyper_history_df : pd.DataFrame
        Per-cell hyperbolicity history. Must contain ``"time"``, ``"x"``, and
        ``"max_abs_imag_eig"``.
    """
    overlay_variable = CFG.overlay_variable
    if overlay_variable is None:
        return
    if overlay_variable not in history_df.columns:
        print(f"[warning] Overlay variable '{overlay_variable}' not found. Skipping.")
        return

    t_worst = worst_time(hyper_summary_df)
    hyper_snap = hyper_history_df[np.isclose(hyper_history_df["time"], t_worst)].sort_values("x")
    state_snap = history_df[np.isclose(history_df["time"], t_worst)].sort_values("x")

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    ax1.plot(hyper_snap["x"],
             hyper_snap["max_abs_imag_eig"],
             linewidth=CFG.line_width,
             label=r"$\max_j |\Im(\lambda_j)|$")
    ax1.set_xlabel("x", fontsize=CFG.axis_font_size)
    ax1.set_ylabel(r"$\max_j |\Im(\lambda_j)|$", fontsize=CFG.axis_font_size)
    ax1.tick_params(axis="both", labelsize=CFG.tick_font_size)

    ax2 = ax1.twinx()
    ax2.plot(state_snap["x"],
             state_snap[overlay_variable],
             linewidth=CFG.line_width,
             linestyle="--",
             label=overlay_variable)
    ax2.set_ylabel(overlay_variable, fontsize=CFG.axis_font_size)
    ax2.tick_params(axis="both", labelsize=CFG.tick_font_size)

    if CFG.use_titles:
        ax1.set_title(
            f"Hyperbolicity loss vs {overlay_variable} at t={t_worst:.3f}",
            fontsize=CFG.axis_font_size,
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=CFG.legend_font_size,
               frameon=False,
               loc="best")

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()
    save_figure(fig, f"hyper_overlay_{overlay_variable}")
    close_or_show(fig)

def plot_three_recharge_runs_momentum_velocity(
        csv_alpha0 : str | Path, 
        csv_alpha1 : str | Path, 
        csv_alpha2 : str | Path, 
        labels : tuple[str, str, str] = (
            r"$\alpha = 0$", r"$\alpha = 1$", r"$\alpha = 2$",
            ),
        filename : str = "momentum_velocity_three_alpha",
        x_target : float | None = None,
        use_spatial_mean : bool = True,
        ) -> None:
    """
    Plot discharge q = h * u_m and velocity u_m over time for three different
    CSV files. Designed specifically to recreate the Section 3.6 discussion from
    the Ersoy et al. paper.

    Works with:
        - summary CSVs containing: time, mean_h, mean_u_m,
        - history CSVs containing: time, x, h, u_m

    If history CSVs are given:
    - use_spatial_mean = True averages over x at each time.
    - use_spatial_mean = False selects the cell nearest to x_target.
    """

    paths = [Path(csv) for csv in (csv_alpha0, csv_alpha1, csv_alpha2)]

    def _load_time_series(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)

        # Case 1: summary CSV
        if {"time", "mean_h", "mean_u_m"}.issubset(df.columns):
            out = df[["time", "mean_h", "mean_u_m"]].copy()
            out = out.rename(columns={"mean_h": "h", "mean_u_m": "u_m"})

        # Case 2: full history CSV
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
                for time_value, group in df.groupby("time"):
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

    data = [_load_time_series(path) for path in paths]

    # -------------------------
    # Figure 1: discharge q=h*u_m
    # -------------------------
    fig_q, ax_q = plt.subplots(figsize=(6.2, 4.2))

    for df, label in zip(data, labels):
        ax_q.plot(df["time"], df["q"], linewidth=CFG.line_width, label=label)

    apply_axis_style(
        ax_q,
        xlabel=r"$t$",
        ylabel=r"$q(t,x)$",
        title=None,
    )

    ax_q.legend(fontsize=CFG.legend_font_size)
    fig_q.tight_layout()
    save_figure(fig_q, f"{filename}_momentum")
    close_or_show(fig_q)

    # -------------------------
    # Figure 2: velocity u_m
    # -------------------------
    fig_u, ax_u = plt.subplots(figsize=(6.2, 4.2))

    for df, label in zip(data, labels):
        ax_u.plot(df["time"], df["u_m"], linewidth=CFG.line_width, label=label)

    apply_axis_style(
        ax_u,
        xlabel=r"$t$",
        ylabel=r"$u(t,x)$",
        title=None,
    )

    ax_u.legend(fontsize=CFG.legend_font_size)
    fig_u.tight_layout()
    save_figure(fig_u, f"{filename}_velocity")
    close_or_show(fig_u)

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Entry point: load data from CSV files and generate all enabled plots.

    Prints a summary of resolved paths to stdout, then calls each plotting
    function that is enabled in the config. Hyperbolicity plots are only
    generated when ``CFG.hyper_enabled`` is True. Finishes by printing the
    output directory to stdout.
    """
    print("Plotter config :", CONFIG_FILE.resolve())
    print("Project root   :", ROOT)
    print("Results dir    :", RESULTS_DIR)
    print("Figures dir    :", FIGURES_DIR)
    print()

    summary_df, history_df, summary_moments, history_moments = read_solution_data()
    hyper_summary_df, hyper_history_df = read_hyperbolicity_data()

    print("Summary file   :", SUMMARY_FILE)
    print("History file   :", HISTORY_FILE)
    if CFG.hyper_enabled:
        print("Hyper summary  :", HYPER_SUMMARY_FILE)
        print("Hyper history  :", HYPER_HISTORY_FILE)
    print()

    if CFG.plot_mean_height:
        plot_time_series(summary_df, "mean_h", "Mean height",
                         "mean_height_vs_time", "Mean water height over time")

    if CFG.plot_mean_velocity:
        plot_time_series(summary_df, "mean_u_m", r"Mean $u_m$",
                         "mean_velocity_vs_time", "Mean velocity over time")

    if CFG.plot_height_envelope:
        plot_height_envelope(summary_df)

    if CFG.plot_height_snapshots:
        plot_snapshots(history_df, "h", "Height",
                       "height_snapshots", "Height snapshots at selected times")

    if CFG.plot_velocity_snapshots:
        plot_snapshots(history_df, "u_m", r"$u_m$",
                       "velocity_snapshots", r"Velocity snapshots at selected times")
        plot_velocity_profile_snapshots(history_df, x_target=0.5,\
                                        filename="vertical_velocity_profiles_xmid",
                                        title="Vertical velocity profiles at selected times")

    if CFG.plot_moments:
        for summary_col, history_col in zip(summary_moments, history_moments):
            idx = int(history_col.replace("a", ""))
            plot_time_series(
                summary_df,
                summary_col,
                rf"Mean $\alpha_{idx}$",
                f"mean_a{idx}_vs_time",
                rf"Mean $\alpha_{idx}$ over time",
            )
            plot_snapshots(
                history_df,
                history_col,
                rf"$\alpha_{idx}$",
                f"a{idx}_snapshots",
                rf"$\alpha_{idx}$ snapshots at selected times",
            )

    if CFG.hyper_enabled:
        if CFG.plot_hyper_max_imag_vs_time:
            plot_max_imag_vs_time(hyper_summary_df)

        if CFG.plot_hyper_nonhyperbolic_cells_vs_time:
            plot_nonhyperbolic_cells_vs_time(hyper_summary_df)

        if CFG.plot_hyper_worst_x_vs_time:
            plot_worst_location_vs_time(hyper_summary_df)

        if CFG.plot_hyper_heatmap:
            plot_hyperbolicity_heatmap(hyper_history_df)

        if CFG.plot_hyper_binary_heatmap:
            plot_binary_hyperbolicity_heatmap(hyper_history_df)

        if CFG.plot_hyper_snapshots:
            plot_hyperbolicity_snapshots(hyper_summary_df, hyper_history_df)

        if CFG.plot_hyper_overlay:
            plot_overlay_at_worst_time(
                summary_df, history_df, hyper_summary_df, hyper_history_df
            )
        
    comp_cfg = load_comparison_config()
        
    if comp_cfg is not None and comp_cfg["enabled"]:
        COMPARISON_DIR = Path(
            "/home/anenin/Documents/Git/thesis/model/processing/Ersoy"
        ).expanduser().resolve()
        
        plot_three_recharge_runs_momentum_velocity(
            COMPARISON_DIR / comp_cfg["alpha0_file"],
            COMPARISON_DIR / comp_cfg["alpha1_file"],
            COMPARISON_DIR / comp_cfg["alpha2_file"],
            labels=comp_cfg["labels"],
            filename=comp_cfg["filename"],
            use_spatial_mean=comp_cfg["use_spatial_mean"],
            x_target=comp_cfg["x_target"],
        )

    print(f"Finished. Plots saved to:\n{FIGURES_DIR}")


if __name__ == "__main__":
    main()
