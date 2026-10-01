"""`RunData` - one description of a run, however it was obtained.

Pages take a `RunData` and never learn where the numbers came from:
`from_simulation` reads a finished run in-process, `from_directory` rebuilds one
from CSVs. That indirection is what lets the same page code serve a run that
just finished and a thesis run from a year ago, and what lets the report degrade
instead of crashing when a run captured less than the full set of outputs.

Everything except the final state is optional, and every absence is recorded in
`RunData.warnings` rather than raised. The concrete case that forces this: all
40 hyperbolicity CSVs currently in `results/` are 1 byte - stale artifacts of a
defect fixed in Step 7 - so a loader that assumed a readable file would fail on
almost every run on disk.

No module-level configuration, no import-time side effects, no `cwd`
dependence, and nothing here imports matplotlib. That is the direct lesson of
`processing/plotter.py`, whose `CFG = load_plotter_config()` runs at import,
reads a path relative to the working directory, and creates a directory as a
side effect.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import coefficients  # noqa: F401  (documents the profile provenance)
from ..wetdry import DEFAULT_THRESHOLDS, WetDryThresholds
from .history import (
    DEFAULT_CHUNK_ROWS,
    DEFAULT_MAX_SNAPSHOTS,
    FieldSnapshots,
    select_steps,
    snapshots_from_history_list,
    steps_in_field_history,
    stream_field_history,
)

# A file at or below this size holds at most a header or a bare newline. The
# real stale files on disk are exactly 1 byte.
_EMPTY_CSV_BYTES = 4

_MOMENT_RE = re.compile(r"^a(\d+)$")

_LEGACY_HYPERBOLICITY = {
    "history": "recharge_hyperbolicity_history.csv",
    "summary": "recharge_hyperbolicity_summary.csv",
}


# ---------------------------------------------------------------------------
# vertical velocity profile
# ---------------------------------------------------------------------------

_profile_pde = None


def vertical_velocity_profile(order: int, values, z_points) -> np.ndarray:
    """`u(z)` per cell, via the package's own generic implementation.

    Deliberately delegates to `SWME1D.compute_vertical_velocity_profile` rather
    than re-deriving the basis. `processing/` currently carries six hand-rolled
    copies of this, every one of them capped at `a_2` or `a_3`, so an N=6 run
    renders as though it were N=3 with no warning - the same silent truncation
    Step 4.5 removed from inside the package. The package version is
    `eval_phi`-backed and N-agnostic.

    `values` must be `[x, h, u_m, a_1..a_N]`, with `u_m` at column 2.
    """
    global _profile_pde
    if _profile_pde is None:
        from .. import pde as pde_module
        # Carries no state used by the profile computation; built once.
        _profile_pde = pde_module.SWME1D('lakeAtRest', 0.0, 1.0, False, False)
    return _profile_pde.compute_vertical_velocity_profile(order, values, z_points)


# ---------------------------------------------------------------------------
# value objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchemeCounters:
    """The always-on scheme-level hyperbolicity counters.

    These measure the **path-averaged interface matrix**, not `A(U)` at any
    state, and they mean different things per scheme - which is why the scheme
    name travels with them. `is_vacuous` is the important one: LF and PRICE
    never eigendecompose anything, so their zero carries no information and must
    never be rendered as reassurance.
    """

    scheme: str | None = None
    spectra_examined: int = 0
    nonhyperbolic_count: int = 0
    max_abs_imaginary_eigenvalue: float = 0.0
    tolerance: float = 0.0

    @property
    def eigendecomposes(self) -> bool:
        return self.scheme in {"Roe", "Osher"}

    @property
    def records_per_interface(self) -> int:
        """Roe records the path average once; Osher each weight-scaled node."""
        return {"Roe": 1, "Osher": 5}.get(self.scheme or "", 0)

    @property
    def is_vacuous(self) -> bool:
        return not self.eigendecomposes or self.spectra_examined == 0


@dataclass(frozen=True)
class HyperbolicityData:
    """The model-level, cell-by-cell spectrum of `A(U)`, when it was captured."""

    summary: pd.DataFrame | None = None
    cells: pd.DataFrame | None = None
    tolerance: float = 1e-10
    stride: int | None = None

    @property
    def has_cells(self) -> bool:
        return self.cells is not None and len(self.cells) > 0


@dataclass(frozen=True)
class RunMetadata:
    """Whatever the `{prefix}_run.json` sidecar recorded. All optional."""

    schema: int | None = None
    config_name: str | None = None
    model: str | None = None
    hyperbolic: bool | None = None
    order: int | None = None
    resolution: int | None = None
    domain: tuple[float, float] | None = None
    initial_condition: str | None = None
    boundary_condition: str | None = None
    scheme: str | None = None
    time_integrator: str | None = None
    scheme_well_balanced: bool | None = None
    viscosity: float | None = None
    slip_length: float | None = None
    t_end: float | None = None
    elapsed_seconds: float | None = None
    store_history: bool | None = None
    store_hyperbolicity: bool | None = None
    hyperbolicity_stride: int | None = None
    hyperbolicity_tol: float | None = None
    bed_profile: str | None = None
    bed_params: dict = field(default_factory=dict)
    reference_water_level: float | None = None
    has_topography: bool | None = None
    infiltration_type: str | None = None
    rainfall_rate: float | None = None
    mass_created_by_clamping: float | None = None
    generated_at: str | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict) -> "RunMetadata":
        known = {f for f in cls.__dataclass_fields__ if f != "raw"}
        kwargs = {key: value for key, value in payload.items() if key in known}
        domain = kwargs.get("domain")
        if domain is not None:
            kwargs["domain"] = (float(domain[0]), float(domain[1]))
        return cls(**kwargs, raw=payload)

    @property
    def is_empty(self) -> bool:
        return not self.raw


# ---------------------------------------------------------------------------
# RunData
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunData:
    """One run, as the report sees it."""

    final: pd.DataFrame
    order: int
    moment_columns: tuple[str, ...]
    meta: RunMetadata
    source: str

    summary_history: pd.DataFrame | None = None
    snapshots: FieldSnapshots | None = None
    hyperbolicity: HyperbolicityData | None = None
    scheme_counters: SchemeCounters | None = None
    bed_elevation: np.ndarray | None = None
    thresholds: WetDryThresholds | None = None
    directory: Path | None = None
    prefix: str | None = None
    warnings: tuple[str, ...] = ()

    # -- derived ---------------------------------------------------------

    @property
    def x(self) -> np.ndarray:
        return self.final["x"].to_numpy()

    @property
    def n_cells(self) -> int:
        return len(self.final)

    @property
    def title(self) -> str:
        name = self.meta.config_name or (self.prefix and Path(self.prefix).name) or "run"
        model = self.meta.model or "SWME1D"
        tag = "HSWME" if self.meta.hyperbolic else "SWME"
        if model == "RechargeSWME1D":
            tag = f"Recharge {tag}"
        return f"{name} - {tag}, N = {self.order}"

    @property
    def effective_thresholds(self) -> WetDryThresholds:
        return self.thresholds or DEFAULT_THRESHOLDS

    @property
    def has_topography(self) -> bool:
        return (self.bed_elevation is not None
                and bool(np.any(np.abs(self.bed_elevation) > 0.0)))

    @property
    def min_depth(self) -> float:
        depths = [float(np.nanmin(self.final["h"].to_numpy()))]
        if self.summary_history is not None and "min_h" in self.summary_history:
            depths.append(float(np.nanmin(self.summary_history["min_h"].to_numpy())))
        elif self.snapshots is not None:
            depths.append(float(np.nanmin(self.snapshots.field("h"))))
        return min(depths)

    @property
    def went_dry(self) -> bool:
        return self.min_depth < self.effective_thresholds.h_wet

    def final_array(self) -> np.ndarray:
        """`[x, h, u_m, a_1..a_N]`, ready for the profile computation."""
        return self.final[list(primitive_columns(self.order))].to_numpy(dtype=np.float64)

    def velocity_profile(self, values, z_points) -> np.ndarray:
        return vertical_velocity_profile(self.order, values, z_points)

    def inventory(self) -> list[tuple[str, bool, str]]:
        """(label, present, note) per optional data source, for the cover page."""
        hyper = self.hyperbolicity
        return [
            ("Final state", True, f"{self.n_cells} cells"),
            ("Time histories", self.summary_history is not None,
             f"{len(self.summary_history)} steps" if self.summary_history is not None
             else "store_history was off"),
            ("Field snapshots", self.snapshots is not None,
             f"{len(self.snapshots.steps)} of {self.snapshots.total_available} steps"
             if self.snapshots is not None else "no field history"),
            ("Model hyperbolicity", bool(hyper and hyper.has_cells),
             f"{len(hyper.cells)} cell records" if hyper and hyper.has_cells
             else "store_hyperbolicity was off"),
            ("Scheme counters", self.scheme_counters is not None,
             f"{self.scheme_counters.scheme}" if self.scheme_counters is not None
             else "scheme not recorded"),
            ("Bed elevation", self.has_topography,
             self.meta.bed_profile or "flat bed"),
            ("Run metadata", not self.meta.is_empty,
             "sidecar found" if not self.meta.is_empty else "no run.json sidecar"),
        ]


def primitive_columns(order: int) -> tuple[str, ...]:
    return ("x", "h", "u_m", *(f"a{i}" for i in range(1, order + 1)))


def moment_columns_of(frame: pd.DataFrame, prefix: str = "") -> tuple[str, ...]:
    """Moment columns in numeric order, N-agnostic and with no upper cap.

    Sorted numerically rather than lexically, so `a10` follows `a9`.
    """
    found = []
    for column in frame.columns:
        if prefix and not column.startswith(prefix):
            continue
        match = _MOMENT_RE.match(column[len(prefix):])
        if match:
            found.append((int(match.group(1)), column))
    found.sort()
    indices = [index for index, _ in found]
    if indices and indices != list(range(1, len(indices) + 1)):
        raise ValueError(
            f"Moment columns are not contiguous from a1: found {indices}. "
            "The report cannot infer N from a gapped set."
        )
    return tuple(column for _, column in found)


# ---------------------------------------------------------------------------
# in-process loader
# ---------------------------------------------------------------------------

def from_simulation(sim,
                    final_state,
                    *,
                    metadata: dict | RunMetadata | None = None,
                    max_snapshots: int = DEFAULT_MAX_SNAPSHOTS) -> RunData:
    """Build a `RunData` from a finished simulation, without touching disk.

    Every attribute is read defensively, so a partially-populated stand-in
    works as well as a real `ClassicalSimulation1D`.
    """
    order = int(sim.order)
    columns = primitive_columns(order)
    final = pd.DataFrame(np.asarray(final_state, dtype=np.float64)[:, :len(columns)],
                         columns=list(columns))

    if isinstance(metadata, RunMetadata):
        meta = metadata
    elif metadata:
        meta = RunMetadata.from_dict(metadata)
    else:
        meta = RunMetadata()

    warnings: list[str] = []

    snapshots = None
    summary = None
    history = getattr(sim, "history", None)
    if history:
        snapshots = snapshots_from_history_list(
            history, columns=columns, max_snapshots=max_snapshots)
        summary = summary_from_history(history, order)
    else:
        warnings.append("No field history was stored (store_history was off).")

    hyperbolicity = None
    cells = getattr(sim, "hyperbolicity_history", None)
    summary_rows = getattr(sim, "hyperbolicity_summary", None)
    if cells or summary_rows:
        hyperbolicity = HyperbolicityData(
            summary=pd.DataFrame(summary_rows) if summary_rows else None,
            cells=_drop_unparseable(pd.DataFrame(cells)) if cells else None,
            tolerance=float(getattr(sim, "hyperbolicity_tol", 1e-10)),
            stride=int(getattr(sim, "hyperbolicity_stride", 1)),
        )
    else:
        warnings.append(
            "Model-level hyperbolicity was not captured "
            "(store_hyperbolicity was off).")

    scheme = getattr(sim, "spatial_discretization", None)
    counters = None
    if scheme is not None:
        counters = SchemeCounters(
            scheme=type(scheme).__name__,
            spectra_examined=int(getattr(scheme, "spectra_examined", 0)),
            nonhyperbolic_count=int(getattr(scheme, "nonhyperbolic_count", 0)),
            max_abs_imaginary_eigenvalue=float(
                getattr(scheme, "max_abs_imaginary_eigenvalue", 0.0)),
            tolerance=float(getattr(scheme, "hyperbolicity_tolerance", 1e-10)),
        )

    bed = None
    mesh = getattr(sim, "mesh", None)
    if mesh is not None and getattr(mesh, "has_topography", False):
        # Physical cells only: pages must never see ghost cells.
        bed = np.asarray(mesh.bed_elevation, dtype=np.float64)[1:-1]

    return RunData(
        final=final,
        order=order,
        moment_columns=columns[3:],
        meta=meta,
        source="simulation",
        summary_history=summary,
        snapshots=snapshots,
        hyperbolicity=hyperbolicity,
        scheme_counters=counters,
        bed_elevation=bed,
        thresholds=getattr(getattr(sim, "pde_type", None), "wet_dry", None),
        warnings=tuple(warnings),
    )


def summary_from_history(history, order: int) -> pd.DataFrame:
    """Per-step spatial statistics, matching `cli.write_outputs` column for column.

    Duplicated rather than imported so that `swme.report` never imports the CLI;
    a test asserts the two loaders agree, which catches any drift.
    """
    columns = primitive_columns(order)
    rows = []
    for entry in history:
        frame = pd.DataFrame(np.asarray(entry["data"], dtype=np.float64)[:, :len(columns)],
                             columns=list(columns))
        row = {
            "step": entry["step"],
            "time": entry["time"],
            "mean_h": frame["h"].mean(),
            "mean_u_m": frame["u_m"].mean(),
            "min_h": frame["h"].min(),
            "max_h": frame["h"].max(),
        }
        for i in range(1, order + 1):
            name = f"a{i}"
            row[f"mean_{name}"] = frame[name].mean()
            row[f"min_{name}"] = frame[name].min()
            row[f"max_{name}"] = frame[name].max()
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# directory loader
# ---------------------------------------------------------------------------

def _read_optional_csv(path: Path, label: str, warnings: list[str]) -> pd.DataFrame | None:
    """Read a CSV that may legitimately be missing, empty, or a stale stub.

    The single choke point for every optional input, rather than a `try` at
    each call site - because the failure it guards is the common case, not the
    exotic one: every `*_hyperbolicity_*.csv` in `results/` is 1 byte.
    """
    if not path.exists():
        warnings.append(f"{label}: not found ({path.name}).")
        return None
    if path.stat().st_size <= _EMPTY_CSV_BYTES:
        warnings.append(f"{label}: file is empty ({path.name}).")
        return None
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        warnings.append(f"{label}: file has no data ({path.name}).")
        return None
    if frame.empty:
        warnings.append(f"{label}: file has no rows ({path.name}).")
        return None
    return frame


def _drop_unparseable(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove the `eigvals` column: a numpy repr, not data.

    `simulation.py` stores the raw complex array, which `to_csv` renders as
    `"[ 1.+0.j -1.+0.j]"`. It is unparseable and fully redundant with the
    `eigvals_real`/`eigvals_imag` string columns beside it.
    """
    return frame.drop(columns=["eigvals"], errors="ignore")


def parse_eigenvalue_column(values) -> np.ndarray:
    """Parse `;`-joined `%.16e` strings into a (n_rows, n_eigenvalues) array.

    An empty string means "no spectrum recorded" and becomes a NaN row rather
    than a parse error - `simulation.py` writes it that way when a step had no
    finite spectrum at all.
    """
    parsed = [
        np.fromstring(text, sep=";") if isinstance(text, str) and text.strip() else None
        for text in values
    ]
    widths = {len(row) for row in parsed if row is not None}
    if not widths:
        return np.zeros((len(parsed), 0))
    width = max(widths)
    out = np.full((len(parsed), width), np.nan)
    for index, row in enumerate(parsed):
        if row is not None and len(row) == width:
            out[index] = row
    return out


def find_prefix(directory: Path, prefix: str | None = None) -> str:
    """The run's filename stem inside `directory`.

    Non-recursive on purpose: `results/` nests one run per leaf directory, so
    recursing would merge two runs into one report.
    """
    if prefix is not None:
        return prefix
    candidates = sorted(directory.glob("*_final.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No run output in {directory}: expected a file matching "
            "'*_final.csv'."
        )
    if len(candidates) > 1:
        stems = ", ".join(p.name[: -len("_final.csv")] for p in candidates)
        raise ValueError(
            f"{directory} holds more than one run ({stems}). "
            "Pass prefix= to choose one."
        )
    return candidates[0].name[: -len("_final.csv")]


def from_directory(directory,
                   *,
                   prefix: str | None = None,
                   max_snapshots: int = DEFAULT_MAX_SNAPSHOTS,
                   chunk_rows: int = DEFAULT_CHUNK_ROWS,
                   metadata: dict | RunMetadata | None = None,
                   hyperbolicity: bool = True) -> RunData:
    """Rebuild a `RunData` from the CSVs a run left behind.

    Only `{prefix}_final.csv` is required; everything else contributes if it is
    there and is noted in `warnings` if it is not.

    `hyperbolicity=False` skips the per-cell spectra, which are the largest
    files a run writes. The interactive viewer uses it to defer them until
    they are asked for, then calls `load_hyperbolicity` itself.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"No such directory: {directory}")
    prefix = find_prefix(directory, prefix)
    warnings: list[str] = []

    final = pd.read_csv(directory / f"{prefix}_final.csv")
    moments = moment_columns_of(final)
    order = len(moments)
    columns = primitive_columns(order)

    # metadata sidecar
    if isinstance(metadata, RunMetadata):
        meta = metadata
    elif metadata:
        meta = RunMetadata.from_dict(metadata)
    else:
        sidecar = directory / f"{prefix}_run.json"
        if sidecar.exists():
            meta = RunMetadata.from_dict(json.loads(sidecar.read_text()))
        else:
            meta = RunMetadata()
            warnings.append(
                f"No {prefix}_run.json sidecar: the flux scheme, bed profile and "
                "wet-dry thresholds of this run are unknown. Re-run to record them."
            )
    if meta.order is not None and meta.order != order:
        warnings.append(
            f"Sidecar says N = {meta.order} but the CSV has {order} moment "
            "columns; using the CSV."
        )

    summary = _read_optional_csv(
        directory / f"{prefix}_summary_history.csv", "Time histories", warnings)
    if summary is not None:
        summary = summary.sort_values(["time", "step"]).reset_index(drop=True)

    snapshots = None
    history_path = directory / f"{prefix}_field_history.csv"
    if history_path.exists() and history_path.stat().st_size > _EMPTY_CSV_BYTES:
        available = (summary["step"].to_numpy() if summary is not None
                     else steps_in_field_history(history_path, chunk_rows))
        snapshots = stream_field_history(
            history_path,
            steps_wanted=select_steps(available, max_snapshots),
            columns=columns,
            chunk_rows=chunk_rows,
            total_available=len(set(int(s) for s in available)),
        )
    else:
        warnings.append("Field snapshots: no field history on disk.")

    hyperbolicity_data = (load_hyperbolicity(directory, prefix, meta, warnings)
                          if hyperbolicity else None)

    counters = None
    if meta.scheme is not None:
        raw = meta.raw.get("scheme_counters", {})
        counters = SchemeCounters(
            scheme=meta.scheme,
            spectra_examined=int(raw.get("spectra_examined", 0)),
            nonhyperbolic_count=int(raw.get("nonhyperbolic_count", 0)),
            max_abs_imaginary_eigenvalue=float(
                raw.get("max_abs_imaginary_eigenvalue", 0.0)),
            tolerance=float(raw.get("tolerance", 1e-10)),
        )

    bed = _rebuild_bed(final, meta, warnings)
    thresholds = None
    if meta.raw.get("wet_dry"):
        thresholds = WetDryThresholds(**meta.raw["wet_dry"])

    return RunData(
        final=final,
        order=order,
        moment_columns=moments,
        meta=meta,
        source="directory",
        summary_history=summary,
        snapshots=snapshots,
        hyperbolicity=hyperbolicity_data,
        scheme_counters=counters,
        bed_elevation=bed,
        thresholds=thresholds,
        directory=directory,
        prefix=prefix,
        warnings=tuple(warnings),
    )


def load_hyperbolicity(directory: Path, prefix: str, meta: RunMetadata,
                       warnings: list[str]) -> HyperbolicityData | None:
    """Read the two hyperbolicity CSVs, under either the current or legacy name.

    Runs written before Step 8.5 used a hardcoded `recharge_` stem regardless of
    model (defect D6), so both spellings have to be accepted forever.
    """
    def pick(kind: str) -> Path:
        current = directory / f"{prefix}_hyperbolicity_{kind}.csv"
        legacy = directory / _LEGACY_HYPERBOLICITY[kind]
        return current if current.exists() else legacy

    summary = _read_optional_csv(
        pick("summary"), "Model hyperbolicity (summary)", warnings)
    cells = _read_optional_csv(
        pick("history"), "Model hyperbolicity (per cell)", warnings)
    if summary is None and cells is None:
        return None
    return HyperbolicityData(
        summary=summary.sort_values(["time", "step"]).reset_index(drop=True)
        if summary is not None else None,
        cells=_drop_unparseable(cells.sort_values(["time", "x"]).reset_index(drop=True))
        if cells is not None else None,
        tolerance=float(meta.hyperbolicity_tol or 1e-10),
        stride=meta.hyperbolicity_stride,
    )


def _rebuild_bed(final: pd.DataFrame, meta: RunMetadata,
                 warnings: list[str]) -> np.ndarray | None:
    """Re-derive the bed from the sidecar's profile name and parameters.

    The CSVs carry no `Z` column, and widening them would break byte-comparison
    against every result already in `results/`. Evaluating the named profile at
    the CSV's own `x` reproduces `mesh.set_bed_elevation` to round-off, because
    both sample the cell centres.
    """
    if not meta.bed_profile or meta.bed_profile == "flat":
        return None
    from .. import topography as topography_module
    try:
        profile = topography_module.get_bed_profile(meta.bed_profile, **meta.bed_params)
    except (KeyError, TypeError, ValueError) as error:
        warnings.append(
            f"Bed profile '{meta.bed_profile}' could not be rebuilt ({error}); "
            "the topography page is omitted."
        )
        return None
    return np.asarray(profile(final["x"].to_numpy()), dtype=np.float64)
