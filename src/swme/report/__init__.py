"""Multi-page PDF reports about a single run (RESTRUCTURE_PLAN.md Step 8.5).

    from swme import report
    run = report.from_directory("results/wetdry_dam_break")
    report.build_report(run, "results/wetdry_dam_break/report.pdf")

or, straight off a finished simulation, `report.from_simulation(sim, final)`.

`import swme.report` deliberately does **not** import matplotlib. The data
layer (`data`, `history`) is pure pandas/numpy, and the rendering modules are
resolved lazily through the module `__getattr__` below, so a scripted solver run
that never asks for a report pays nothing - the same reason Step 8 moved
`swme.plotting` inside the `--plot` branch of the CLI.
"""

from __future__ import annotations

from .data import (
    HyperbolicityData,
    RunData,
    RunMetadata,
    SchemeCounters,
    from_directory,
    from_simulation,
)
from .history import FieldSnapshots

__all__ = [
    "RunData", "RunMetadata", "SchemeCounters", "HyperbolicityData",
    "FieldSnapshots", "from_directory", "from_simulation",
    "ReportStyle", "DEFAULT_STYLE", "build_report", "iter_figures",
    "selected_pages", "DEFAULT_PAGES", "PageSpec", "ReportResult",
]

_LAZY = {
    "ReportStyle": ("style", "ReportStyle"),
    "DEFAULT_STYLE": ("style", "DEFAULT_STYLE"),
    "build_report": ("assemble", "build_report"),
    "iter_figures": ("assemble", "iter_figures"),
    "selected_pages": ("assemble", "selected_pages"),
    "DEFAULT_PAGES": ("assemble", "DEFAULT_PAGES"),
    "PageSpec": ("assemble", "PageSpec"),
    "ReportResult": ("assemble", "ReportResult"),
}


def __getattr__(name: str):
    """Resolve the rendering names on first use (PEP 562).

    This is what keeps matplotlib off the import path of a plain solver run.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(f".{target[0]}", __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
