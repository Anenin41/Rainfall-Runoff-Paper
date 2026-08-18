"""Page construction and visual style, as a value rather than a global.

Three deliberate departures from `processing/plotter.py`:

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
* **Text is laid out against a measured box, never against a guess.** Every
  string this module draws outside an axes - captions, monospace blocks,
  placeholders - is wrapped to the width of the box it is drawn in and, where
  it still does not fit, shrunk until it does. Matplotlib's own `wrap=True`
  wraps to the *figure* width rather than the artist's box, which is why the
  earlier report had captions running off the page and two-column text blocks
  overprinting each other. Nothing here relies on it.

The second half of that last point is what makes the pages safe to compose:
`caption` takes the room it needs out of *its own* axes rather than drawing
into the gap below, so a caption can never land on the next panel's title no
matter how long the text or how tight the grid.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure


class Prose(str):
    """A paragraph in a text block that the renderer may re-wrap.

    A plain `str` in a text block is verbatim - it is a key/value line whose
    column alignment carries meaning, so breaking it would be worse than
    shrinking the font. A `Prose` is flowing text with no internal alignment
    to protect, so it is re-wrapped to whatever width the block turns out to
    have. Subclassing `str` keeps both kinds in one list and keeps
    `" ".join(lines)` working for callers that only want the words.
    """

    def __new__(cls, text: str, indent: str = "  "):
        item = super().__new__(cls, text)
        item.indent = indent
        return item

    @property
    def continuation(self) -> str:
        """Indent for the second and later lines, hanging under a bullet."""
        return self.indent + ("  " if self.startswith(("- ", "* ")) else "")


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
    min_size: float = 5.5
    line_width: float = 1.4
    grid: bool = True
    colormap: str = "viridis"
    heatmap_floor: float = 1e-16
    max_profile_curves: int = 6
    max_snapshot_curves: int = 5

    # Font metrics, as a fraction of the point size. DejaVu Sans Mono - the
    # matplotlib default monospace - has a single advance width of 0.602 em, so
    # the monospace figure is near-exact; the proportional one is a deliberate
    # over-estimate of the average advance, which makes every fit conservative.
    mono_em: float = 0.610
    text_em: float = 0.600
    line_spacing: float = 1.5
    caption_spacing: float = 1.35

    # ------------------------------------------------------------------
    # pages
    # ------------------------------------------------------------------

    def new_page(self, nrows: int = 1, ncols: int = 1, *,
                 height_ratios=None, hspace: float = 0.38,
                 wspace: float = 0.32, left: float = 0.11,
                 right: float = 0.95, top: float = 0.895,
                 bottom: float = 0.075):
        """A blank page and its axes grid.

        Returns `(figure, axes)` with `axes` always a 2-D object array, so page
        code can index `axes[row, col]` without special-casing a 1x1 grid.

        The margins are arguments because the pages differ in what has to fit
        outside the axes: a page of monospace blocks wants the full width, a
        page of colourbars wants a wider gutter between the columns.
        """
        figure = Figure(figsize=self.page_size, dpi=self.dpi)
        axes = figure.subplots(nrows, ncols, squeeze=False,
                               height_ratios=height_ratios)
        figure.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                               hspace=hspace, wspace=wspace)
        return figure, axes

    def page_header(self, figure: Figure, title: str,
                    subtitle: str | None = None, *, left: float = 0.11) -> None:
        figure.suptitle(title, fontsize=self.title_size, x=left, ha="left",
                        y=0.972)
        if subtitle:
            figure.text(left, 0.947, subtitle, fontsize=self.subtitle_size,
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
            ax.set_title(title, fontsize=self.label_size + 0.5, pad=6.0)
        ax.tick_params(labelsize=self.tick_size)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if self.grid:
            ax.grid(alpha=0.25, linewidth=0.6)
        if legend:
            ax.legend(fontsize=self.legend_size, frameon=False,
                      borderaxespad=0.4, handlelength=1.6, labelspacing=0.35)

    # ------------------------------------------------------------------
    # text metrics
    # ------------------------------------------------------------------

    def char_width(self, size: float, family: str = "monospace") -> float:
        """Width of one character, in inches, at `size` points."""
        em = self.mono_em if family == "monospace" else self.text_em
        return em * size / 72.0

    def columns(self, width_inches: float, size: float,
                family: str = "monospace") -> int:
        """How many characters fit across `width_inches`."""
        return max(12, int(width_inches / self.char_width(size, family)))

    @staticmethod
    def axes_size(ax) -> tuple[float, float]:
        """The axes box in inches, which is what text has to fit inside."""
        box = ax.get_position()
        page_width, page_height = ax.figure.get_size_inches()
        return box.width * page_width, box.height * page_height

    def wrap(self, lines, columns: int) -> list[str]:
        """Re-wrap the `Prose` entries of a text block, verbatim ones intact."""
        wrapped: list[str] = []
        for line in lines:
            if not isinstance(line, Prose):
                wrapped.append(str(line))
                continue
            pieces = textwrap.wrap(
                str(line), max(columns, 20), initial_indent=line.indent,
                subsequent_indent=line.continuation, break_long_words=False,
                break_on_hyphens=False)
            wrapped.extend(pieces or [""])
        return wrapped

    def fit(self, lines, width_inches: float, height_inches: float, *,
            size: float, family: str) -> tuple[list[str], float]:
        """Wrap `lines` and pick the largest point size that still fits.

        Shrinking changes the wrap, which changes the height, so this iterates
        rather than solving once. It converges in two or three passes and is
        capped at `min_size` - a block that cannot fit even then is a page
        design error, not something to render illegibly small.
        """
        rendered = self.wrap(lines, self.columns(width_inches, size, family))
        for _ in range(8):
            longest = max((len(line) for line in rendered), default=0)
            needed_width = longest * self.char_width(size, family)
            needed_height = len(rendered) * self.line_spacing * size / 72.0
            overflow = max(needed_width / width_inches,
                           needed_height / max(height_inches, 1e-6))
            if overflow <= 1.0 or size <= self.min_size:
                break
            size = max(self.min_size, size / overflow)
            rendered = self.wrap(lines, self.columns(width_inches, size, family))
        return rendered, size

    def column_width(self, *, ncols: int = 1, left: float = 0.11,
                     right: float = 0.95, wspace: float = 0.32) -> float:
        """Width in inches of one column of the grid `new_page` would build.

        Lets a page measure its text before it has any axes to measure it
        against, which is what pages with data-dependent text length need in
        order to size their rows to it.
        """
        span = (right - left) * self.page_size[0]
        return span / (ncols + (ncols - 1) * wspace)

    def measure(self, lines, width_inches: float, *, size: float | None = None,
                family: str = "monospace") -> tuple[list[str], float, float]:
        """Wrap `lines` to a column and report how tall they will be.

        Returns `(lines, size, height)`, the height in figure fractions and
        ready to be handed to `new_page` as a row ratio.
        """
        rendered, size = self.fit(lines, width_inches, float("inf"),
                                  size=float(size or self.body_size),
                                  family=family)
        height = len(rendered) * self.line_spacing * size / 72.0 / self.page_size[1]
        return rendered, size, height

    # ------------------------------------------------------------------
    # text on a page
    # ------------------------------------------------------------------

    def text_block(self, ax, lines, *, size: float | None = None,
                   family: str = "monospace") -> float:
        """Fill an axes with text instead of a plot.

        Returns the point size actually used. The block is wrapped to the axes
        width and shrunk to the axes height, so a long warning or a narrow
        column can no longer print over its neighbour or off the page edge.
        """
        ax.axis("off")
        width, height = self.axes_size(ax)
        rendered, size = self.fit(lines, width, height,
                                  size=float(size or self.body_size),
                                  family=family)
        ax.text(0.0, 1.0, "\n".join(rendered), transform=ax.transAxes,
                fontsize=size, family=family, va="top", ha="left",
                linespacing=self.line_spacing)
        return size

    def caption(self, ax, text: str, *, gap: float | None = None) -> None:
        """A note under an axes, for the things a curve cannot say itself.

        The note is drawn in a band taken out of the *bottom of its own axes*,
        never out of the gap to the panel below. That is the whole reason this
        cannot collide with anything: the space it uses is space no other
        artist was ever going to be given.
        """
        figure = ax.figure
        _, page_height = figure.get_size_inches()
        width, _ = self.axes_size(ax)
        lines: list[str] = []
        for paragraph in str(text).split("\n"):
            lines.extend(textwrap.wrap(
                paragraph, self.columns(width, self.caption_size, "sans-serif"))
                or [""])

        line_height = self.caption_spacing * self.caption_size / 72.0 / page_height
        if gap is None:
            # Room for the tick labels and the x-label, which live in the same
            # band: about one tick label, one axis label and the pad of each.
            ticks_and_label = 3.6 if ax.get_xlabel() else 1.8
            gap = ticks_and_label * self.label_size / 72.0 / page_height
        band = len(lines) * line_height + gap

        box = ax.get_position()
        band = min(band, box.height * 0.55)
        self._shrink_from_bottom(ax, band)
        figure.text(box.x0, box.y0 + band - gap, "\n".join(lines),
                    fontsize=self.caption_size, color="0.35", va="top",
                    ha="left", linespacing=self.caption_spacing)

    def placeholder(self, ax, title: str, message: str) -> None:
        """An explicit statement that data is missing.

        Never a blank panel: a reader cannot tell an empty axes from a clean
        result, and for hyperbolicity in particular the difference between "no
        loss detected" and "nothing was measured" is the whole point.
        """
        # Not `ax.axis("off")`: that clears the frame too, and the dashed
        # border is the part that tells a reader at a glance that this panel
        # is a statement about missing data rather than a plot of it.
        ax.set_xticks([])
        ax.set_yticks([])
        width, _ = self.axes_size(ax)
        size = self.caption_size + 0.5
        body = "\n".join(textwrap.wrap(
            message, self.columns(width * 0.88, size, "sans-serif")))
        ax.text(0.5, 0.56, title, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=self.label_size + 1, color="0.30")
        ax.text(0.5, 0.48, body, transform=ax.transAxes, ha="center",
                va="top", fontsize=size, color="0.45", linespacing=1.4)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("0.80")
            spine.set_linestyle((0, (4, 4)))

    # ------------------------------------------------------------------
    # colour
    # ------------------------------------------------------------------

    def colorbar(self, ax, mappable, label: str | None = None, *,
                 width: float = 0.016, pad: float = 0.012, ticks=None):
        """A colourbar in a slot cut from the right of `ax`.

        `figure.colorbar(mappable, ax=ax)` steals 15% of the axes width and
        then places the bar, its tick labels and its label to the right of
        that, which on a two-column page pushes the label into the next
        column's y-label. Here the bar is a fixed narrow strip inside the
        panel's own cell, so the gutter stays a gutter.
        """
        figure = ax.figure
        box = ax.get_position()
        ax.set_position([box.x0, box.y0,
                         max(box.width - width - pad, box.width * 0.55),
                         box.height])
        inner = ax.get_position()
        cax = figure.add_axes([inner.x1 + pad, inner.y0, width, inner.height])
        bar = figure.colorbar(mappable, cax=cax, ticks=ticks)
        if label:
            bar.set_label(label, fontsize=self.label_size)
        cax.tick_params(labelsize=self.tick_size)
        bar.outline.set_linewidth(0.6)
        # Remembered so a later `caption` shrinks the bar with its axes.
        ax._report_colorbar_axes = cax
        return bar

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

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    @staticmethod
    def _shrink_from_bottom(ax, band: float) -> None:
        """Give up `band` (in figure fractions) from the bottom of an axes."""
        for target in (ax, getattr(ax, "_report_colorbar_axes", None)):
            if target is None:
                continue
            box = target.get_position()
            target.set_position([box.x0, box.y0 + band, box.width,
                                 max(box.height - band, 0.02)])


DEFAULT_STYLE = ReportStyle()
