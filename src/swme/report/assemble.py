"""Page selection and PDF assembly.

Page *order* is the static order of `DEFAULT_PAGES` and never depends on the
data, so two reports of the same case have the same page sequence and a
difference in `ReportResult.page_keys` means something. Page *selection* is the
`available` predicate, evaluated once up front so the result can say what was
skipped as well as what was drawn.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from . import hyperbolicity as hyperbolicity_pages
from . import pages as page_module
from .data import RunData
from .style import DEFAULT_STYLE, ReportStyle


@dataclass(frozen=True)
class PageSpec:
    key: str
    title: str
    render: Callable
    available: Callable[[RunData], bool] = lambda run: True


@dataclass(frozen=True)
class ReportResult:
    path: Path
    page_keys: tuple[str, ...]
    skipped_keys: tuple[str, ...]
    warnings: tuple[str, ...]


DEFAULT_PAGES: tuple[PageSpec, ...] = (
    PageSpec("cover", "Run summary", page_module.page_cover),
    PageSpec("final_state", "Final state", page_module.page_final_state),
    PageSpec("time_histories", "Time histories", page_module.page_time_histories),
    PageSpec("space_time", "Space-time evolution", page_module.page_space_time,
             lambda run: run.snapshots is not None),
    PageSpec("velocity_profiles", "Vertical velocity profiles",
             page_module.page_velocity_profiles),
    PageSpec("hyperbolicity", "Hyperbolicity",
             hyperbolicity_pages.page_hyperbolicity),
    PageSpec("hyperbolicity_maps", "Hyperbolicity in space and time",
             hyperbolicity_pages.page_hyperbolicity_maps,
             hyperbolicity_pages.has_cell_spectra),
    PageSpec("wet_dry", "Wet-dry behaviour", page_module.page_wet_dry,
             lambda run: run.went_dry),
    PageSpec("topography", "Topography", page_module.page_topography,
             lambda run: run.has_topography),
)


def selected_pages(run: RunData, pages=None) -> list[PageSpec]:
    return [spec for spec in (pages or DEFAULT_PAGES) if spec.available(run)]


def iter_figures(run: RunData, *, style: ReportStyle | None = None,
                 pages=None) -> Iterator[tuple[PageSpec, "object"]]:
    """Render each applicable page, yielding `(spec, figure)`.

    Exists so tests can inspect figures without producing a PDF, and so a
    notebook user can pull out a single page.
    """
    style = style or DEFAULT_STYLE
    for spec in selected_pages(run, pages):
        yield spec, spec.render(run, style)


def build_report(run: RunData, output_path, *, style: ReportStyle | None = None,
                 pages=None, title: str | None = None) -> ReportResult:
    """Render the report to a single multi-page PDF.

    Written to a sibling temporary file and moved into place, so an
    interrupted or failing render never leaves a truncated `report.pdf` sitting
    next to the CSVs looking like a finished artifact.
    """
    from matplotlib.backends.backend_pdf import PdfPages

    style = style or DEFAULT_STYLE
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")

    chosen = selected_pages(run, pages)
    all_keys = [spec.key for spec in (pages or DEFAULT_PAGES)]
    drawn = [spec.key for spec in chosen]

    try:
        with PdfPages(temporary) as pdf:
            for spec in chosen:
                figure = spec.render(run, style)
                pdf.savefig(figure)
                # Figures are bare `Figure` objects, never registered with
                # pyplot, so dropping the reference is all the cleanup needed.
                figure.clear()
            info = pdf.infodict()
            info["Title"] = title or run.title
            info["Subject"] = "Shallow Water Moment Equations - single run report"
            info["Creator"] = "swme.report"
            info["CreationDate"] = datetime.now(timezone.utc)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    os.replace(temporary, output_path)
    return ReportResult(
        path=output_path,
        page_keys=tuple(drawn),
        skipped_keys=tuple(key for key in all_keys if key not in set(drawn)),
        warnings=run.warnings,
    )
