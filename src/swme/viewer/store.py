"""Loaded runs, cached, so moving a slider never re-reads a CSV.

A run is loaded through `report.data.from_directory`, the same loader the PDF
uses, but **without** its per-cell hyperbolicity: those are the largest files a
run writes (26 MB for a 160-cell smoke test, far more for a thesis run) and most
views never look at them. They are read on first request by
`RunStore.hyperbolicity` and cached separately, with a smaller budget.

Entries are keyed by the newest modification time of the run's files, so a run
that is re-computed while the viewer is open is reloaded rather than served
stale.

Dash serves callbacks from several threads and one selection fires several
callbacks at once, all wanting the same run. A per-key lock makes the first
caller load it and the rest wait for that result rather than loading it again.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..report.data import RunData, from_directory, load_hyperbolicity
from ..report.history import DEFAULT_CHUNK_ROWS
from .catalog import RunEntry, discover

DEFAULT_MAX_SNAPSHOTS = 400
DEFAULT_CAPACITY = 6
DEFAULT_HYPERBOLICITY_CAPACITY = 2


@dataclass(frozen=True)
class HyperbolicityView:
    """A run's per-cell spectra plus the numbers derived from them once.

    `run` is the cached `RunData` with its `hyperbolicity` filled in, so the
    report's own text helpers (`scheme_counter_lines`, `model_level_lines`)
    can be called on it unchanged. `derived` is `recompute_summary` of the
    per-cell frame, or the recorded summary adapted to the same columns when
    the per-cell frame is absent (`corrected` is then False).
    """

    run: RunData
    derived: pd.DataFrame | None
    corrected: bool
    warnings: tuple[str, ...] = ()


class _LRU:
    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._items: OrderedDict = OrderedDict()
        self._guard = threading.Lock()
        self._locks: dict = {}

    def lock_for(self, key) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(key, threading.Lock())

    def get(self, key):
        with self._guard:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
        return None

    def put(self, key, value) -> None:
        with self._guard:
            self._items[key] = value
            self._items.move_to_end(key)
            while len(self._items) > self.capacity:
                old, _ = self._items.popitem(last=False)
                self._locks.pop(old, None)

    def __len__(self) -> int:
        return len(self._items)


def fingerprint(entry: RunEntry) -> int:
    """Newest mtime of the run's files: changes whenever the run is re-written."""
    stamps = [path.stat().st_mtime_ns for path in entry.directory.glob(f"{entry.prefix}_*")
              if path.is_file()]
    return max(stamps, default=0)


class RunStore:
    """The run catalogue for one results root, and an LRU of loaded runs."""

    def __init__(self, root, *, max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
                 chunk_rows: int = DEFAULT_CHUNK_ROWS,
                 capacity: int = DEFAULT_CAPACITY,
                 hyperbolicity_capacity: int = DEFAULT_HYPERBOLICITY_CAPACITY):
        self.root = Path(root)
        self.max_snapshots = int(max_snapshots)
        self.chunk_rows = int(chunk_rows)
        self._runs = _LRU(capacity)
        self._spectra = _LRU(hyperbolicity_capacity)
        self._entries: dict[str, RunEntry] = {}
        self.refresh()

    # -- catalogue -------------------------------------------------------

    def refresh(self) -> list[RunEntry]:
        """Re-scan the root. Loaded runs stay cached until their files change."""
        self._entries = {entry.key: entry for entry in discover(self.root)}
        return self.entries()

    def entries(self) -> list[RunEntry]:
        return list(self._entries.values())

    def entry(self, key: str) -> RunEntry:
        try:
            return self._entries[key]
        except KeyError:
            raise KeyError(f"Unknown run {key!r}; reload the run list.") from None

    # -- payloads --------------------------------------------------------

    def run(self, key: str) -> RunData:
        """The run's `RunData`, without per-cell hyperbolicity."""
        entry = self.entry(key)
        cache_key = (key, fingerprint(entry))
        cached = self._runs.get(cache_key)
        if cached is not None:
            return cached
        with self._runs.lock_for(cache_key):
            cached = self._runs.get(cache_key)
            if cached is None:
                cached = from_directory(entry.directory, prefix=entry.prefix,
                                        max_snapshots=self.max_snapshots,
                                        chunk_rows=self.chunk_rows,
                                        hyperbolicity=False)
                self._runs.put(cache_key, cached)
        return cached

    def hyperbolicity(self, key: str) -> HyperbolicityView:
        """The run with its spectra loaded, and the derived per-step summary."""
        entry = self.entry(key)
        cache_key = (key, fingerprint(entry))
        cached = self._spectra.get(cache_key)
        if cached is not None:
            return cached
        with self._spectra.lock_for(cache_key):
            cached = self._spectra.get(cache_key)
            if cached is None:
                cached = self._load_spectra(key)
                self._spectra.put(cache_key, cached)
        return cached

    def _load_spectra(self, key: str) -> HyperbolicityView:
        # Imported here: the hyperbolicity module draws pages and so imports
        # matplotlib, which only this view needs.
        from ..report.hyperbolicity import _adapt_recorded_summary, recompute_summary

        base = self.run(key)
        warnings: list[str] = []
        spectra = load_hyperbolicity(base.directory, base.prefix, base.meta, warnings)
        run = dataclasses.replace(base, hyperbolicity=spectra)
        if spectra is None:
            return HyperbolicityView(run, None, False, tuple(warnings))
        if spectra.has_cells:
            return HyperbolicityView(
                run, recompute_summary(spectra.cells, spectra.tolerance), True,
                tuple(warnings))
        return HyperbolicityView(run, _adapt_recorded_summary(spectra.summary),
                                 False, tuple(warnings))
