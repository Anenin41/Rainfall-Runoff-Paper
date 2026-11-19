"""An animated GIF of a run's stored steps, drawn with matplotlib.

The panels are those of the report's final-state page, `h`, `u_m` and every
moment, at every stored step instead of only the last. A run with topography
gets one more panel first, the free surface `h + Z` over the bed, as on the
report's topography page. Each axis is fixed to the range its quantity covers
over the whole stored history, as in the viewer, so only the curves move.

Only the curves and the time are redrawn per frame (blitting): the axes, ticks
and labels are drawn once. Measured on the 200-cell lake run: the viewer's 400
stored steps take 11 s and give 2.3 MB; 240 steps took 6 s with blitting and
about 50 s redrawing every frame in full. The viewer's GIF button calls
`gif_bytes`.
"""

from __future__ import annotations

import io

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .data import RunData
from .style import ReportStyle

DEFAULT_FRAME_MS = 50          # 20 frames a second, the viewer's 1x
GIF_COLOURS = 64               # plenty for line plots; keeps the file small


def _panels(run: RunData):
    """`(label, values over the stored steps, (steps, cells))` per panel."""
    values = run.snapshots.values
    panels = []
    if run.has_topography:
        panels.append(("free surface $h + Z$", values[:, :, 1] + run.bed_elevation))
    panels.append(("water depth $h$", values[:, :, 1]))
    panels.append(("mean velocity $u_m$", values[:, :, 2]))
    panels += [(rf"moment $\alpha_{{{index}}}$", values[:, :, 2 + index])
               for index in range(1, run.order + 1)]
    return panels


def _fixed_range(data) -> tuple[float, float]:
    """The finite range of `data`, widened by 5 %, never of zero width."""
    finite = np.asarray(data, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1.0, 1.0
    low, high = float(finite.min()), float(finite.max())
    pad = 0.05 * (high - low) if high > low else 0.05 * max(abs(high), 1.0)
    return low - pad, high + pad


def build_animation(run: RunData, style: ReportStyle | None = None, *, dpi: int = 90):
    """`(figure, update)` for the stored steps of `run`.

    `update(k)` draws stored step `k` (curves and time only) onto the canvas
    and returns the figure's curves, so tests can read what was drawn.
    """
    if run.snapshots is None or len(run.snapshots.times) == 0:
        raise ValueError("this run has no field history to animate; "
                         "set postprocessing.store_history: true and re-run")
    style = style or ReportStyle()
    panels = _panels(run)
    rows = int(np.ceil(len(panels) / 2))
    figure = Figure(figsize=(8.0, 2.4 * rows + 0.7), dpi=dpi)
    canvas = FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(rows, 2, squeeze=False).ravel())
    figure.subplots_adjust(left=0.09, right=0.97, top=1 - 0.55 / figure.get_figheight(),
                           bottom=0.5 / figure.get_figheight(), hspace=0.45, wspace=0.28)
    x = run.x
    curves = []
    for ax, (label, data) in zip(axes, panels):
        if run.has_topography and label.startswith("free surface"):
            ax.plot(x, run.bed_elevation, color="0.45", linewidth=1.0, label="bed $Z$")
        curve, = ax.plot(x, data[0], linewidth=style.line_width, animated=True)
        curves.append(curve)
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_ylim(*_fixed_range(data))
        style.axis(ax, "$x$", label)
    for ax in axes[len(panels):]:
        ax.axis("off")
    figure.text(0.02, 1 - 0.25 / figure.get_figheight(), run.title, fontsize=style.label_size,
                va="center", ha="left")
    clock = figure.text(0.98, 1 - 0.25 / figure.get_figheight(), "", ha="right",
                        va="center", fontsize=style.label_size + 1, animated=True)
    canvas.draw()
    background = canvas.copy_from_bbox(figure.bbox)
    times = run.snapshots.times

    def update(k: int):
        canvas.restore_region(background)
        for curve, (_, data) in zip(curves, panels):
            curve.set_ydata(data[k])
            figure.draw_artist(curve)
        clock.set_text(f"t = {float(times[k]):.5g}")
        figure.draw_artist(clock)
        return curves

    return figure, update


def gif_bytes(run: RunData, *, frame_ms: int = DEFAULT_FRAME_MS,
              style: ReportStyle | None = None, dpi: int = 90) -> bytes:
    """The stored steps of `run` as an animated GIF that loops.

    A GIF stores frame delays in hundredths of a second, and browsers slow
    delays under 20 ms down to 100 ms, so `frame_ms` is rounded to 10 ms and
    held at 20 ms or more.
    """
    from PIL import Image

    figure, update = build_animation(run, style, dpi=dpi)
    canvas = figure.canvas
    size = canvas.get_width_height()
    frames = []
    for k in range(len(run.snapshots.times)):
        update(k)
        image = Image.frombuffer("RGBA", size, canvas.buffer_rgba(), "raw", "RGBA", 0, 1)
        frames.append(image.convert("RGB").quantize(colors=GIF_COLOURS,
                                                    method=Image.Quantize.FASTOCTREE))
    duration = max(20, int(round(frame_ms / 10.0)) * 10)
    buffer = io.BytesIO()
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
    return buffer.getvalue()
