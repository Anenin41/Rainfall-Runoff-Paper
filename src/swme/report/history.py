"""Bounded-memory access to a run's field history.

The field history is the one output that scales with run length rather than
with mesh size: the largest on disk is 185 MB / 1.54M rows, and `results/`
totals 1.4 GB. `processing/plotter.py` reads it with a single `pd.read_csv` and
then pivots the whole frame, which is exactly what a report cannot afford —
`chunksize` appears nowhere in the repo before this module.

So the history is decimated to a bounded, evenly spaced set of snapshots, and
**peak memory does not depend on the number of steps**:

    chunk_rows*(5+N)*8   the parser's working chunk
  + n_snap*n_cells*(3+N)*8   the snapshot cube

At the defaults with N=2 and 1000 cells that is ~21 MB, whatever the run length.

Two passes, where the first is normally free. The available step list comes
from `summary_history.csv`, written in the same block as the field history and
~250x smaller; only when the summary is missing does pass one touch the big
file, and then it reads just the `step` column. Pass two streams the payload,
keeping only the chosen steps.

Not reservoir sampling: a random subset cannot guarantee t=0 and t_end are
present, and the space-time maps need *even* coverage in time or the
pcolormesh rows are unevenly spaced and visually lie about wave speeds. Even
spacing needs the step count, which is what pass one provides. Determinism also
makes the two loaders comparable - `snapshots_from_history_list` runs the same
`select_steps`, so an in-process report and one rebuilt from CSVs pick the same
steps by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_MAX_SNAPSHOTS = 240
DEFAULT_CHUNK_ROWS = 200_000


@dataclass(frozen=True)
class FieldSnapshots:
    """An evenly spaced, bounded subset of a run's stored states.

    Attributes
    ----------
    steps, times : np.ndarray
        Shape (n_snap,). The steps actually kept, in increasing order.
    values : np.ndarray
        Shape (n_snap, n_cells, 3 + N), each slice laid out as
        `[x, h, u_m, a_1..a_N]` - the same column order
        `SWME1D.compute_vertical_velocity_profile` expects, so pages never
        reshape.
    total_available : int
        Unique steps in the source *before* decimation, so a page can say
        "24 of 4812 steps shown" rather than implying it drew everything.
    columns : tuple[str, ...]
    """

    steps: np.ndarray
    times: np.ndarray
    values: np.ndarray
    total_available: int
    columns: tuple[str, ...]

    @property
    def n_cells(self) -> int:
        return self.values.shape[1]

    @property
    def x(self) -> np.ndarray:
        return self.values[0, :, 0]

    def nearest_index(self, t: float) -> int:
        """Index of the stored snapshot closest in time to `t`."""
        return int(np.argmin(np.abs(self.times - float(t))))

    def field(self, name: str) -> np.ndarray:
        """A (n_snap, n_cells) space-time array for one column."""
        return self.values[:, :, self.columns.index(name)]


def select_steps(steps, max_snapshots: int = DEFAULT_MAX_SNAPSHOTS) -> np.ndarray:
    """Evenly spaced subset of `steps`, always including the first and last.

    Deterministic, so the CSV path and the in-process path agree.
    """
    steps = np.asarray(sorted(set(int(s) for s in steps)), dtype=np.int64)
    if max_snapshots < 1:
        raise ValueError(f"max_snapshots must be >= 1, got {max_snapshots}.")
    if len(steps) <= max_snapshots:
        return steps
    picks = np.linspace(0, len(steps) - 1, max_snapshots).round().astype(int)
    return steps[np.unique(picks)]


def steps_in_field_history(path, chunk_rows: int = DEFAULT_CHUNK_ROWS) -> np.ndarray:
    """The unique steps in a field-history CSV, reading only the step column.

    The fallback for when `summary_history.csv` is absent. Memory is O(n_steps),
    and no payload column is parsed.
    """
    found: set[int] = set()
    for chunk in pd.read_csv(path, usecols=["step"], chunksize=chunk_rows):
        found.update(chunk["step"].to_numpy(dtype=np.int64).tolist())
    return np.asarray(sorted(found), dtype=np.int64)


def stream_field_history(path,
                         *,
                         steps_wanted,
                         columns,
                         chunk_rows: int = DEFAULT_CHUNK_ROWS,
                         total_available: int | None = None) -> FieldSnapshots:
    """Read only `steps_wanted` out of a field-history CSV.

    Parameters
    ----------
    path : str | os.PathLike
        The `{prefix}_field_history.csv` to read.
    steps_wanted : array-like of int
        Typically the output of `select_steps`.
    columns : sequence of str
        The primitive columns to keep, in order: `[x, h, u_m, a_1..a_N]`.
    chunk_rows : int
        Rows per `read_csv` chunk. Bounds peak memory; does not change results.
    total_available : int, optional
        Unique steps in the source, for reporting. Defaults to len(steps_wanted).
    """
    wanted = np.asarray(sorted(set(int(s) for s in steps_wanted)), dtype=np.int64)
    slot_of = {int(step): index for index, step in enumerate(wanted)}
    columns = tuple(columns)
    fragments: list[list[np.ndarray]] = [[] for _ in wanted]
    times = np.full(len(wanted), np.nan)

    usecols = ["step", "time", *columns]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_rows):
        hit = chunk[chunk["step"].isin(wanted)]
        if hit.empty:
            continue
        for step, rows in hit.groupby("step", sort=False):
            slot = slot_of[int(step)]
            fragments[slot].append(rows[list(columns)].to_numpy(dtype=np.float64))
            times[slot] = float(rows["time"].iloc[0])

    missing = [int(step) for step, frags in zip(wanted, fragments) if not frags]
    if missing:
        raise ValueError(
            f"Field history {path} contains no rows for step(s) {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''}."
        )

    # One step's rows can straddle a chunk boundary, so each slot is stacked
    # and then sorted by x. That makes the result independent of where the
    # boundaries happened to fall.
    stacked = []
    for frags in fragments:
        block = frags[0] if len(frags) == 1 else np.vstack(frags)
        stacked.append(block[np.argsort(block[:, 0], kind="stable")])

    widths = {block.shape[0] for block in stacked}
    if len(widths) != 1:
        raise ValueError(
            f"Field history {path} has a varying number of cells per step: "
            f"{sorted(widths)}."
        )

    return FieldSnapshots(
        steps=wanted,
        times=times,
        values=np.stack(stacked),
        total_available=int(total_available if total_available is not None else len(wanted)),
        columns=columns,
    )


def snapshots_from_history_list(history,
                                *,
                                columns,
                                max_snapshots: int = DEFAULT_MAX_SNAPSHOTS) -> FieldSnapshots:
    """The in-process counterpart: decimate `ClassicalSimulation1D.history`.

    Runs the same `select_steps` as the CSV path, so both pick the same steps.
    """
    by_step = {int(entry["step"]): entry for entry in history}
    chosen = select_steps(by_step.keys(), max_snapshots)
    columns = tuple(columns)
    values = np.stack([
        np.asarray(by_step[int(step)]["data"], dtype=np.float64)[:, :len(columns)]
        for step in chosen
    ])
    return FieldSnapshots(
        steps=chosen,
        times=np.asarray([float(by_step[int(s)]["time"]) for s in chosen]),
        values=values,
        total_available=len(by_step),
        columns=columns,
    )
