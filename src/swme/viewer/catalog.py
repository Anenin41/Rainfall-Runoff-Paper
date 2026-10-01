"""Which runs exist under a results tree, without reading any of their data.

The sidebar lists every run at start-up, and `results/` can hold gigabytes of
field history, so discovery reads only what is cheap: the `*_run.json` sidecar,
the header line of `*_final.csv`, and file sizes. Payloads are loaded later, one
run at a time, by `store.RunStore`.

Discovery follows `moment-sw-report`: a directory holding `*_final.csv` is a
run, otherwise `purge.find_runs` walks the tree one family deep. Unlike the
report, a directory holding several runs is not an error here - each prefix is
listed as its own entry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..purge import find_runs
from ..report.data import _EMPTY_CSV_BYTES, _LEGACY_HYPERBOLICITY, moment_columns_of


@dataclass(frozen=True)
class RunEntry:
    """One run on disk, as the run browser lists it."""

    key: str
    directory: Path
    prefix: str
    family: str
    name: str
    order: int
    config_name: str | None = None
    model: str | None = None
    hyperbolic: bool | None = None
    scheme: str | None = None
    t_end: float | None = None
    resolution: int | None = None
    infiltration_type: str | None = None
    has_sidecar: bool = False
    has_history: bool = False
    has_hyperbolicity: bool = False
    size_bytes: int = 0

    @property
    def closure(self) -> str:
        if self.hyperbolic is None:
            return "?"
        return "HSWME" if self.hyperbolic else "SWME"

    @property
    def label(self) -> str:
        """Short name for legends: the leaf directory, plus the prefix if shared."""
        return self.name


def _has_data(path: Path) -> bool:
    return path.exists() and path.stat().st_size > _EMPTY_CSV_BYTES


def _read_sidecar(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _order_from_header(final: Path) -> int:
    """N from the moment columns of the final-state header, without the rows."""
    return len(moment_columns_of(pd.read_csv(final, nrows=0)))


def _entries_in(directory: Path, root: Path) -> list[RunEntry]:
    finals = sorted(directory.glob("*_final.csv"))
    shared = len(finals) > 1
    relative = directory.relative_to(root) if directory != root else Path(".")
    family = "" if relative.parent == Path(".") else relative.parent.as_posix()
    leaf = directory.name if directory != root else root.resolve().name

    entries = []
    for final in finals:
        prefix = final.name[: -len("_final.csv")]
        meta = _read_sidecar(directory / f"{prefix}_run.json")
        try:
            order = int(meta["order"]) if "order" in meta else _order_from_header(final)
        except (ValueError, KeyError, pd.errors.ParserError):
            continue
        hyper_files = [directory / f"{prefix}_hyperbolicity_history.csv",
                       directory / _LEGACY_HYPERBOLICITY["history"]]
        size = sum(path.stat().st_size for path in directory.glob(f"{prefix}_*")
                   if path.is_file())
        entries.append(RunEntry(
            key=f"{relative.as_posix()}::{prefix}",
            directory=directory,
            prefix=prefix,
            family=family,
            name=f"{leaf} / {prefix}" if shared else leaf,
            order=order,
            config_name=meta.get("config_name"),
            model=meta.get("model"),
            hyperbolic=meta.get("hyperbolic"),
            scheme=meta.get("scheme"),
            t_end=meta.get("t_end"),
            resolution=meta.get("resolution"),
            infiltration_type=meta.get("infiltration_type"),
            has_sidecar=bool(meta),
            has_history=_has_data(directory / f"{prefix}_field_history.csv"),
            has_hyperbolicity=any(_has_data(path) for path in hyper_files),
            size_bytes=size,
        ))
    return entries


def discover(root) -> list[RunEntry]:
    """Every run under `root`, sorted by family and then name.

    `root` may itself be a run directory, in which case it is the only entry.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"No such directory: {root}")
    if any(root.glob("*_final.csv")):
        directories = [root]
    else:
        directories = [run for run in find_runs(root) if any(run.glob("*_final.csv"))]
    entries = [entry for directory in directories
               for entry in _entries_in(directory, root)]
    return sorted(entries, key=lambda entry: (entry.family, entry.name))


def families(entries) -> dict[str, list[RunEntry]]:
    """Entries grouped by family, preserving order. Top-level runs share `""`."""
    grouped: dict[str, list[RunEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.family, []).append(entry)
    return grouped
