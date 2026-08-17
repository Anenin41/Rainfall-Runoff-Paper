"""Page construction and visual style, as a value rather than a global.

Two deliberate departures from `processing/plotter.py`:

* **Style is an argument.** `plotter.py` reads a module-level `CFG` in every
  plotting function, so nothing can be rendered twice with different settings
  and importing the module requires a config file on disk. Here a
  `ReportStyle` is passed down explicitly and nothing global is mutated -
  in particular no `rcParams` are touched, so importing this module cannot
  change how unrelated code plots.
* **No pyplot.** Figures are built as bare `matplotlib.figure.Figure` objects.
  `plt.figure()` would register each one in pyplot's global list, which for a
  multi-page report is a memory leak waiting to happen and forces a matching
  `plt.close` on every path including the failing ones. A `Figure` that nobody
  holds is simply garbage-collected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure


@dataclass(frozen=True)
class ReportStyle:
    """Everything the pages need to know about how to look."""

    page_size: tuple[float, float] = (8.27, 11.69)     # A4 portrait, inches
    dpi: int = 150
    title_size: float = 15.0
    subtitle_size: float = 9.5
    label_size: float = 9.0
    tick_size: float = 8.0
    legend_size: float = 8.0
    caption_size: float = 7.5
    body_size: float = 8.5
    line_width: float = 1.4
    grid: bool = True
    colormap: str = "viridis"
    heatmap_floor: float = 1e-16
    max_profile_curves: int = 6
    max_snapshot_curves: int = 5

    def new_page(self, nrows: int = 1, ncols: int = 1, *,
                 height_ratios=None, hspace: float = 0.45,
                 wspace: float = 0.30):
        """A blank page and its axes grid.

        Returns `(figure, axes)` with `axes` always a 2-D object array, so page
        code can index `axes[row, col]` without special-casing a 1x1 grid.
        """
        figure = Figure(figsize=self.page_size, dpi=self.dpi)
        axes = figure.subplots(nrows, ncols, squeeze=False,
                               height_ratios=height_ratios)
        figure.subplots_adjust(left=0.11, right=0.95, top=0.90, bottom=0.08,
                               hspace=hspace, wspace=wspace)
        return figure, axes

    def page_header(self, figure: Figure, title: str, subtitle: str | None = None) -> None:
        figure.suptitle(title, fontsize=self.title_size, x=0.11, ha="left", y=0.975)
        if subtitle:
            figure.text(0.11, 0.945, subtitle, fontsize=self.subtitle_size,
                        ha="left", va="top", color="0.35")

    def axis(self, ax, xlabel: str | None = None, ylabel: str | None = None,
             title: str | None = None, legend: bool = False) -> None:
        """Apply the house style to one axes.

        Note there is no `tight_layout()` here. `plotter.py:492` calls it inside
        its per-axes helper, so a multi-panel figure re-lays itself out once per
        panel; page layout is the page's business, set once in `new_page`.
        """
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=self.label_size)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=self.label_size)
        if title:
            ax.set_title(title, fontsize=self.label_size + 0.5)
        ax.tick_params(labelsize=self.tick_size)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if self.grid:
            ax.grid(alpha=0.25, linewidth=0.6)
        if legend:
            ax.legend(fontsize=self.legend_size, frameon=False)

    def caption(self, ax, text: str) -> None:
        """A note under an axes, for the things a curve cannot say itself."""
        ax.text(0.0, -0.30, text, transform=ax.transAxes,
                fontsize=self.caption_size, color="0.35", va="top", wrap=True)

    def text_block(self, ax, lines, *, size: float | None = None,
                   family: str = "monospace") -> None:
        """Fill an axes with text instead of a plot."""
        ax.axis("off")
        ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
                fontsize=size or self.body_size, family=family,
                va="top", ha="left", linespacing=1.5)

    def placeholder(self, ax, title: str, message: str) -> None:
        """An explicit statement that data is missing.

        Never a blank panel: a reader cannot tell an empty axes from a clean
        result, and for hyperbolicity in particular the difference between "no
        loss detected" and "nothing was measured" is the whole point.
        """
        ax.axis("off")
        ax.text(0.5, 0.62, title, transform=ax.transAxes, ha="center",
                va="center", fontsize=self.label_size + 1, color="0.30")
        ax.text(0.5, 0.42, message, transform=ax.transAxes, ha="center",
                va="center", fontsize=self.caption_size + 0.5, color="0.45",
                wrap=True)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.80")
            spine.set_linestyle((0, (4, 4)))

    def heatmap_norm(self, values, *, log: bool = True) -> Normalize | None:
        """Colour normalisation for a strictly non-negative field.

        Floors at `heatmap_floor` before taking logs, so exact zeros - which
        are the common case for `|Im(lambda)|` - do not collapse the scale.
        Falls back to linear when the data spans too little to be worth a log.
        """
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return None
        top = float(np.max(finite))
        if not log or top <= self.heatmap_floor:
            return None
        floored = np.maximum(finite, self.heatmap_floor)
        bottom = float(np.min(floored))
        if top / max(bottom, self.heatmap_floor) < 100.0:
            return None
        return LogNorm(vmin=max(bottom, self.heatmap_floor), vmax=top)

    def curve_times(self, times) -> np.ndarray:
        """Indices of an evenly spaced handful of times, for overlay plots."""
        times = np.asarray(times)
        count = min(self.max_snapshot_curves, len(times))
        if count == 0:
            return np.zeros(0, dtype=int)
        return np.unique(np.linspace(0, len(times) - 1, count).round().astype(int))


DEFAULT_STYLE = ReportStyle()
