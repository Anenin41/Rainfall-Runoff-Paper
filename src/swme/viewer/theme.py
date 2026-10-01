"""Colours and the plotly template, for light and dark.

Colour is assigned by the job it does, never cycled:

- **identity** (which run): the categorical slots in fixed order. A run's slot
  is its moment order N, so N = 1 is the same colour in every view and every
  comparison - the reader learns it once. Runs that share an order are told
  apart by line dash, a second channel that does not rely on colour at all.
- **magnitude** (depth, |Im lambda|): one hue, light to dark.
- **polarity** (velocity, moments, differences): blue <-> red through a
  neutral grey that reads as "nothing".
- **state** (hyperbolic or not): the reserved status colours, always with a
  text label beside them.

The categorical steps are the documented, validated reference palette (the
dataviz skill's `references/palette.md`), used unchanged and in its order;
dark mode is its own set of steps, not an inversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    page: str
    ink: str
    ink_secondary: str
    muted: str
    grid: str
    axis: str
    categorical: tuple[str, ...]
    sequential: tuple[str, ...]
    sequential_alt: tuple[str, ...]
    diverging: tuple[str, ...]
    ordinal3: tuple[str, str, str]
    good: str = "#0ca30c"
    critical: str = "#d03b3b"

    def series(self, slot: int) -> str:
        return self.categorical[int(slot) % len(self.categorical)]

    @property
    def band(self) -> str:
        """Fill for a min-max envelope: the series hue, mostly transparent."""
        return _alpha(self.categorical[0], 0.18)

    def scale(self, colors) -> list[list]:
        """Evenly spaced plotly colourscale from a list of steps."""
        last = len(colors) - 1
        return [[index / last, color] for index, color in enumerate(colors)]

    @property
    def sequential_scale(self):
        return self.scale(self.sequential)

    @property
    def sequential_alt_scale(self):
        return self.scale(self.sequential_alt)

    @property
    def diverging_scale(self):
        return self.scale(self.diverging)

    @property
    def template(self) -> go.layout.Template:
        """The plotly template, built once per theme.

        Building one validates a few hundred properties; every figure
        uses it, and during playback a figure is built every frame.
        """
        cached = _TEMPLATES.get(self.name)
        if cached is None:
            cached = _TEMPLATES[self.name] = self._build_template()
        return cached

    def _build_template(self) -> go.layout.Template:
        axis = dict(gridcolor=self.grid, linecolor=self.axis, zeroline=False,
                    showline=True, ticks="outside", tickcolor=self.axis,
                    tickfont=dict(color=self.muted, size=11),
                    title=dict(font=dict(color=self.ink_secondary, size=12)))
        return go.layout.Template(layout=dict(
            font=dict(family=FONT, color=self.ink, size=12),
            paper_bgcolor=self.surface,
            plot_bgcolor=self.surface,
            colorway=list(self.categorical),
            xaxis=axis, yaxis=axis,
            hoverlabel=dict(bgcolor=self.surface, bordercolor=self.axis,
                            font=dict(family=FONT, color=self.ink, size=12)),
            legend=dict(font=dict(color=self.ink_secondary, size=11),
                        bgcolor="rgba(0,0,0,0)"),
            title=dict(font=dict(color=self.ink, size=14), x=0.0, xanchor="left"),
            coloraxis=dict(colorbar=dict(outlinewidth=0,
                                         tickfont=dict(color=self.muted))),
        ))


_TEMPLATES: dict[str, go.layout.Template] = {}


def _alpha(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


LIGHT = Theme(
    name="light",
    surface="#fcfcfb", page="#f9f9f7",
    ink="#0b0b0b", ink_secondary="#52514e", muted="#898781",
    grid="#e1e0d9", axis="#c3c2b7",
    categorical=("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    # Blue 100 -> 700: near zero recedes toward the light surface.
    sequential=("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                "#256abf", "#184f95", "#0d366b"),
    # The second simultaneous magnitude context takes the next slot's hue.
    sequential_alt=("#fde4d8", "#f8b99c", "#f28d63", "#eb6834",
                    "#c4501f", "#933a14", "#62260b"),
    diverging=("#184f95", "#3987e5", "#9ec5f4", "#f0efec",
               "#f4a9a8", "#e34948", "#a32a29"),
    ordinal3=("#86b6ef", "#2a78d6", "#104281"),
)

DARK = Theme(
    name="dark",
    surface="#1a1a19", page="#0d0d0d",
    ink="#ffffff", ink_secondary="#c3c2b7", muted="#898781",
    grid="#2c2c2a", axis="#383835",
    categorical=("#3987e5", "#d95926", "#199e70", "#c98500",
                 "#d55181", "#008300", "#9085e9", "#e66767"),
    # Reversed: near zero recedes toward the *dark* surface.
    sequential=("#0d366b", "#184f95", "#256abf", "#3987e5",
                "#6da7ec", "#9ec5f4", "#cde2fb"),
    sequential_alt=("#4a1d08", "#7a3312", "#b04a1d", "#d95926",
                    "#ec835a", "#f4b293", "#fbdccd"),
    diverging=("#3987e5", "#256abf", "#184f95", "#383835",
               "#a32a29", "#d14140", "#e66767"),
    ordinal3=("#184f95", "#3987e5", "#9ec5f4"),
)

THEMES = {"light": LIGHT, "dark": DARK}


def get(name: str | None) -> Theme:
    return THEMES.get(name or "light", LIGHT)


@dataclass(frozen=True)
class SeriesStyle:
    color: str
    dash: str


def run_styles(orders, theme: Theme) -> list[SeriesStyle]:
    """Colour by moment order, dash by repeat within an order.

    `orders` is the moment order of each run, in the order the runs are shown.
    Two N = 1 runs (source-free and source-active, say) share N = 1's colour
    and differ by dash; all-distinct orders are all solid. The style depends
    only on a run's order and its position among same-order runs, so removing
    an unrelated run from a comparison never repaints the others.
    """
    seen: dict[int, int] = {}
    styles = []
    for order in orders:
        repeat = seen.get(int(order), 0)
        seen[int(order)] = repeat + 1
        styles.append(SeriesStyle(theme.series(int(order)), DASHES[repeat % len(DASHES)]))
    return styles
