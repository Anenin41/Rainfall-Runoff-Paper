"""Command-line entry point: build a simulation from a YAML config and run it.

Replaces the former `swme/main.py`, a ~530-line script whose config parsing and
if/elif dispatch tables had accumulated branches for models deleted in Step 4
(RESTRUCTURE_PLAN.md Step 7). What changed beyond the rewrite itself:

* **Config format is YAML**, per decision 6. `configparser` turned everything
  into strings and needed a `getfloat`/`getboolean`/`getint` call at every use
  site; a typo'd key silently fell back to a default, and a mistyped section
  name simply produced a `KeyError` from deep inside the run.
* **Unknown keys are rejected.** With INI, `viscosty = 1e-3` was silently
  ignored and the run proceeded with the default. Every section here validates
  its key set, so a typo names itself.
* **Unsupported values raise instead of printing.** `main.py` printed
  "PDE_type is not implemented yet" and carried on to a `NameError` on the
  undefined `_pde`; the pvm and time-integrator dispatches had the same shape.
* **CSV output is model-agnostic.** It used to live entirely inside the
  `RechargeSWME1D` branch, so a plain SWME1D or HSWME1D run - including the
  shipped topography and wet-dry cases - produced no numerical artifact at all.
  Recharge filenames are unchanged; see `_output_prefix`.
* **Plotting is opt-in** (`--plot`). It used to be unconditional, so every run
  ended by blocking on an interactive window - wrong for a solver driven from
  scripts, and the reason `scripts/run_thesis_configs.sh` has to set
  `MPLBACKEND=Agg`.

The numerics are untouched: the same objects are built with the same values, so
a config carried over from the INI era reproduces its results bit for bit.
"""

from __future__ import annotations

import argparse
import json
import os
import timeit
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from . import mesh
from . import pde
from . import simulation
from . import spatialDiscretization
from . import timeIntegration
from . import topography as topography_module
from .wetdry import WetDryThresholds

_PACKAGE_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _PACKAGE_DIR / 'config'
DEFAULT_CONFIG = _CONFIG_DIR / 'config.yaml'
_CONFIG_SUFFIXES = ('.yaml', '.yml')

try:
    from recharge.initial_conditions import RechargeSWME1D_CustomIC as RechargeSWME1D
    from recharge.laws import (
        AdmissibleMixingFriction, ConstantInfiltration, HortonInfiltration,
    )
    HAS_RECHARGE = True
except ImportError:                                    # pragma: no cover
    HAS_RECHARGE = False


# --------------------------------------------------------------------------
# config loading and validation
# --------------------------------------------------------------------------

_SECTION_KEYS = {
    'pde': {
        'type', 'initial_condition', 'viscosity', 'slip_length',
        'linear_source', 'hyperbolic',
        # recharge-only below
        'rainfall_rate', 'infiltration_type',
        'horton_f0', 'horton_fc', 'horton_k',
        'constant_infiltration_rate', 'constant_infiltration_eps',
        'constant_limit_by_rainfall', 'constant_limit_by_available_water',
        'mixing_friction_model', 'alpha_R', 'alpha_I',
    },
    'grid': {'x1', 'x2', 'resolution_x'},
    'numerics': {
        'order', 't_end', 'method', 'fvm_type', 'pvm', 'time_integrator',
        'boundary_condition',
    },
    'wet_dry': {'eps_div', 'h_dry', 'h_wet'},
    'postprocessing': {
        'output_dir', 'store_history', 'store_hyperbolicity',
        'history_stride', 'hyperbolicity_stride',
    },
    # `topography` is validated separately: everything except the reserved keys
    # below is a parameter of the chosen bed profile, so the key set is open.
}

_REQUIRED_SECTIONS = ('pde', 'grid', 'numerics')

_TOPOGRAPHY_RESERVED = frozenset({
    'bed_profile', 'reference_water_level',
    'perturbation_amplitude', 'perturbation_center', 'perturbation_width',
})


def _fail(message: str) -> None:
    raise ValueError(message)


def _section(config: dict, name: str, *, required: bool = False) -> dict:
    """Fetch a section, checking it is a mapping and has no unknown keys."""
    if name not in config or config[name] is None:
        if required:
            _fail(f"Config is missing the required '{name}:' section.")
        return {}

    section = config[name]
    if not isinstance(section, dict):
        _fail(f"Config section '{name}:' must be a mapping, got {type(section).__name__}.")

    known = _SECTION_KEYS.get(name)
    if known is not None:
        unknown = set(section) - known
        if unknown:
            _fail(
                f"Unknown key(s) in '{name}:': {', '.join(sorted(unknown))}. "
                f"Valid keys are: {', '.join(sorted(known))}."
            )
    return section


def _number(section: dict, name: str, key: str, default=None) -> float:
    """Read a float, tolerating YAML 1.1's signed-exponent rule.

    PyYAML follows YAML 1.1, whose float pattern requires a *signed* exponent:
    `1.0e-3` resolves to a float but `1.0e3` resolves to a plain string. Rather
    than let that surface as a confusing type error deep in the solver, coerce
    numeric-looking strings here and say so if the coercion fails.
    """
    if key not in section or section[key] is None:
        if default is None:
            _fail(f"Config section '{name}:' is missing required key '{key}'.")
        return default
    value = section[key]
    if isinstance(value, bool):
        _fail(f"'{name}.{key}' must be a number, got the boolean {value}.")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        _fail(
            f"'{name}.{key}' must be a number, got {value!r}. (Note YAML needs "
            "a signed exponent: write 1.0e-3 or 1.0e+3, not 1.0e3.)"
        )


def _integer(section: dict, name: str, key: str, default=None) -> int:
    if key not in section or section[key] is None:
        if default is None:
            _fail(f"Config section '{name}:' is missing required key '{key}'.")
        return default
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            coerced = int(str(value))
        except ValueError:
            _fail(f"'{name}.{key}' must be an integer, got {value!r}.")
        return coerced
    return value


def _boolean(section: dict, name: str, key: str, default=None) -> bool:
    if key not in section or section[key] is None:
        if default is None:
            _fail(f"Config section '{name}:' is missing required key '{key}'.")
        return default
    value = section[key]
    if not isinstance(value, bool):
        _fail(
            f"'{name}.{key}' must be true or false, got {value!r}. "
            "(Quoted strings like \"true\" are not booleans in YAML.)"
        )
    return value


def _text(section: dict, name: str, key: str, default=None) -> str:
    if key not in section or section[key] is None:
        if default is None:
            _fail(f"Config section '{name}:' is missing required key '{key}'.")
        return default
    return str(section[key])


def _choice(section: dict, name: str, key: str, allowed, default=None) -> str:
    value = _text(section, name, key, default)
    if value not in allowed:
        _fail(
            f"'{name}.{key}' = {value!r} is not supported. "
            f"Choose one of: {', '.join(sorted(allowed))}."
        )
    return value


def load_config(path) -> dict:
    """Parse a YAML config and check its top-level shape."""
    path = Path(path)
    with open(path, 'r') as handle:
        config = yaml.safe_load(handle)

    if config is None:
        _fail(f"Config '{path}' is empty.")
    if not isinstance(config, dict):
        _fail(f"Config '{path}' must be a mapping of sections, got {type(config).__name__}.")

    known_sections = set(_SECTION_KEYS) | {'topography'}
    unknown = set(config) - known_sections
    if unknown:
        _fail(
            f"Unknown section(s) in '{path}': {', '.join(sorted(unknown))}. "
            f"Valid sections are: {', '.join(sorted(known_sections))}."
        )
    for name in _REQUIRED_SECTIONS:
        _section(config, name, required=True)
    return config


def resolve_config(name_or_path) -> Path:
    """Resolve a --config value to a readable file.

    Accepts a path, or the bare name of a config shipped in `swme/config/`
    (with or without a .yaml suffix), so cases can be run as
    `--config thesis_5p3_pulse_N1` from any working directory.
    """
    if name_or_path is None:
        return DEFAULT_CONFIG

    candidates = [Path(name_or_path)]
    stem = Path(name_or_path).name
    candidates.append(_CONFIG_DIR / stem)
    candidates += [_CONFIG_DIR / f'{stem}{suffix}' for suffix in _CONFIG_SUFFIXES]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Config '{name_or_path}' not found. Tried: "
        + ", ".join(str(c) for c in candidates)
        + ".\nAvailable shipped configs: "
        + ", ".join(shipped_config_names())
    )


def shipped_config_names() -> list[str]:
    names = set()
    for suffix in _CONFIG_SUFFIXES:
        names.update(path.stem for path in _CONFIG_DIR.glob(f'*{suffix}'))
    return sorted(names)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------

def build_topography(config: dict) -> topography_module.TopographySettings:
    """Bed elevation from the optional `topography:` section.

    Absent means a flat bed at zero, which leaves `mesh.has_topography` False
    and the solver on its original non-augmented path (Step 5).
    """
    section = config.get('topography')
    if not section:
        return topography_module.TopographySettings()
    if not isinstance(section, dict):
        _fail("Config section 'topography:' must be a mapping.")

    profile_name = section.get('bed_profile', 'flat')
    params = {
        key: _number(section, 'topography', key)
        for key in section if key not in _TOPOGRAPHY_RESERVED
    }
    return topography_module.TopographySettings(
        bed_elevation=topography_module.get_bed_profile(profile_name, **params),
        reference_water_level=_number(section, 'topography', 'reference_water_level', 1.0),
        perturbation_amplitude=_number(section, 'topography', 'perturbation_amplitude', 0.0),
        perturbation_center=_number(section, 'topography', 'perturbation_center', 0.0),
        perturbation_width=_number(section, 'topography', 'perturbation_width', 1.0),
    )


def build_wet_dry(config: dict) -> WetDryThresholds:
    """Wet-dry thresholds from the optional `wet_dry:` section.

    The defaults never activate above h ~ 1e-3, so a config that omits this
    behaves exactly as it did before Step 6.
    """
    section = _section(config, 'wet_dry')
    defaults = WetDryThresholds()
    return WetDryThresholds(
        eps_div=_number(section, 'wet_dry', 'eps_div', defaults.eps_div),
        h_dry=_number(section, 'wet_dry', 'h_dry', defaults.h_dry),
        h_wet=_number(section, 'wet_dry', 'h_wet', defaults.h_wet),
    )


def _build_infiltration_model(section: dict):
    kind = _choice(section, 'pde', 'infiltration_type', {'horton', 'constant'})
    if kind == 'horton':
        return HortonInfiltration(
            _number(section, 'pde', 'horton_f0'),
            _number(section, 'pde', 'horton_fc'),
            _number(section, 'pde', 'horton_k'),
        )
    return ConstantInfiltration(
        I0=_number(section, 'pde', 'constant_infiltration_rate'),
        eps=_number(section, 'pde', 'constant_infiltration_eps', 1e-14),
        limit_by_rainfall=_boolean(section, 'pde', 'constant_limit_by_rainfall', False),
        limit_by_available_water=_boolean(
            section, 'pde', 'constant_limit_by_available_water', True),
    )


def build_pde(config: dict, topography, wet_dry):
    """Construct SWME1D / HSWME1D / RechargeSWME1D from the `pde:` section."""
    section = _section(config, 'pde', required=True)
    numerics = _section(config, 'numerics', required=True)

    model = _choice(section, 'pde', 'type',
                    {'SWME1D', 'HSWME1D', 'RechargeSWME1D'})
    initial_condition = _text(section, 'pde', 'initial_condition')
    viscosity = _number(section, 'pde', 'viscosity')
    slip_length = _number(section, 'pde', 'slip_length')
    linear_source = _boolean(section, 'pde', 'linear_source', False)

    # `linear_source` returns a matrix rather than a vector, which only
    # `ImplicitEuler` knows how to apply - so pairing it with any other
    # integrator is a silent wrong answer. main.py enforced this by
    # composition; state it.
    integrator = _text(numerics, 'numerics', 'time_integrator')
    linear_source_implicit = linear_source and integrator == 'ImplicitEuler'
    if linear_source and not linear_source_implicit:
        _fail(
            f"pde.linear_source is true but numerics.time_integrator is "
            f"'{integrator}'. The linear source returns a matrix that only "
            "ImplicitEuler applies; pair them or set linear_source: false."
        )

    if model in ('SWME1D', 'HSWME1D'):
        # HSWME1D is the same class with the hyperbolic closure; `hyperbolic:`
        # may also be given explicitly and must agree if both are present.
        hyperbolic = _boolean(section, 'pde', 'hyperbolic', model == 'HSWME1D')
        if model == 'HSWME1D' and not hyperbolic:
            _fail("pde.type is HSWME1D but pde.hyperbolic is false; they conflict.")
        return pde.SWME1D(
            initial_condition, viscosity, slip_length, hyperbolic,
            linear_source_implicit,
            topography=topography, wet_dry=wet_dry,
        )

    if not HAS_RECHARGE:
        raise ImportError(
            "Config requests pde.type 'RechargeSWME1D' but the recharge package "
            "is not importable in this environment."
        )

    mixing = _choice(section, 'pde', 'mixing_friction_model',
                     {'admissible'}, 'admissible')
    mixing_model = AdmissibleMixingFriction(
        alpha_R=_number(section, 'pde', 'alpha_R'),
        alpha_I=_number(section, 'pde', 'alpha_I'),
    )
    return RechargeSWME1D(
        initial_condition, viscosity, slip_length,
        _boolean(section, 'pde', 'hyperbolic', False),
        linear_source,
        _number(section, 'pde', 'rainfall_rate'),
        _build_infiltration_model(section),
        mixing_model,
        topography=topography,
        wet_dry=wet_dry,
    )


_SCHEMES = {
    'PRICE': spatialDiscretization.PRICE,
    'LF': spatialDiscretization.LF,
    'Roe': spatialDiscretization.Roe,
    'Osher': spatialDiscretization.Osher,
}

_INTEGRATORS = {'ExplicitEuler', 'ImplicitEuler', 'Exact'}


def build_scheme(config: dict):
    numerics = _section(config, 'numerics', required=True)
    _choice(numerics, 'numerics', 'fvm_type', {'PVM'}, 'PVM')
    name = _choice(numerics, 'numerics', 'pvm', set(_SCHEMES))
    return _SCHEMES[name]()


def build_time_integration(config: dict):
    numerics = _section(config, 'numerics', required=True)
    pde_section = _section(config, 'pde', required=True)
    name = _choice(numerics, 'numerics', 'time_integrator', _INTEGRATORS)
    if name == 'ImplicitEuler':
        return timeIntegration.ImplicitEuler(
            _boolean(pde_section, 'pde', 'linear_source', False))
    if name == 'ExplicitEuler':
        return timeIntegration.ExplicitEuler()
    return timeIntegration.Exact()


def build_simulation(config: dict, *, verbose: bool = False):
    """Build a ready-to-run simulation from an already-parsed config.

    Split out of `run()` so a caller - a test, a notebook - can get a real
    simulation object without also acquiring `run()`'s output directory,
    CSV writing and console printing. `run()` is now this plus I/O.
    """
    numerics = _section(config, 'numerics', required=True)
    postprocessing = _section(config, 'postprocessing')

    _choice(numerics, 'numerics', 'method', {'classical'}, 'classical')
    boundary_condition = _choice(
        numerics, 'numerics', 'boundary_condition',
        {'PERIODIC', 'INFLOW_OUTFLOW'})
    order = _integer(numerics, 'numerics', 'order')

    _topography = build_topography(config)
    _wet_dry = build_wet_dry(config)
    _pde = build_pde(config, _topography, _wet_dry)
    _mesh = build_mesh(config, _topography, boundary_condition)

    _simulation = simulation.ClassicalSimulation1D(
        order, _pde, _mesh, boundary_condition,
        _text(_section(config, 'pde'), 'pde', 'initial_condition'),
        build_scheme(config), build_time_integration(config),
    )

    # Diagnostics capture, for every model rather than recharge only.
    _simulation.verbose = verbose
    _simulation.store_history = _boolean(
        postprocessing, 'postprocessing', 'store_history', True)
    _simulation.store_hyperbolicity = _boolean(
        postprocessing, 'postprocessing', 'store_hyperbolicity', False)
    _simulation.history_stride = _integer(
        postprocessing, 'postprocessing', 'history_stride', 1)
    _simulation.hyperbolicity_stride = _integer(
        postprocessing, 'postprocessing', 'hyperbolicity_stride', 1)

    return _simulation


def build_mesh(config: dict, topography, boundary_condition):
    grid = _section(config, 'grid', required=True)
    _mesh = mesh.UniformRectangularMesh1D(
        [_number(grid, 'grid', 'x1'), _number(grid, 'grid', 'x2')],
        _integer(grid, 'grid', 'resolution_x'),
    )
    # Sample the bed onto the grid, ghost cells filled with the same boundary
    # condition as the state - otherwise the two edge interfaces see an
    # inconsistent (U, Z) pair (Step 5).
    if topography.bed_elevation is not None:
        _mesh.set_bed_elevation(topography.bed_elevation, boundary_condition)
    return _mesh


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

def _primitive_columns(order: int) -> list[str]:
    """Columns of the post-processed array: [x, h, u_m, a1..aN]."""
    return ["x", "h", "u_m"] + [f"a{i}" for i in range(1, order + 1)]


def _output_prefix(output_dir: str, model: str, hyperbolic: bool, order: int,
                   infiltration_type: str | None) -> str:
    """Filename stem for this run's CSVs.

    The recharge pattern is preserved exactly, because `processing/*.py`'s
    comparisons match these names inside a per-case subfolder (Step 4.5).
    Deliberately carries no config-name prefix: uniqueness across runs is the
    caller's job via --output-dir, as `scripts/run_thesis_configs.sh` does.
    Two configs sharing one output directory still overwrite each other.
    """
    tag = "hswme" if hyperbolic else "swme"
    if model == 'RechargeSWME1D':
        return os.path.join(output_dir, f"recharge_{tag}_N{order}_{infiltration_type}")
    return os.path.join(output_dir, f"{tag}_N{order}")


SIDECAR_SCHEMA = 1


def build_run_metadata(sim, config: dict, *, config_name=None, config_path=None,
                       elapsed_seconds=None) -> dict:
    """The `{prefix}_run.json` sidecar: what this run actually did.

    The CSVs carry only `[x, h, u_m, a1..aN]`, which leaves a report rebuilt
    from disk unable to say which flux scheme produced them or what the bed
    looked like. Those are not cosmetic gaps: the interface-matrix
    hyperbolicity counters are uninterpretable without the scheme name (Roe
    records one path average per interface, Osher five weight-scaled node
    matrices, LF and PRICE none at all), and the topography page has no bed to
    draw. Rather than widen the CSVs - which would break byte-comparison
    against every result already in `results/` - this records the run
    alongside them, additively.

    It is deliberately a record of the *run*, not a copy of the config: the
    config on disk can be edited afterwards, and `--output-dir` means the
    config-to-directory mapping cannot be inverted anyway.

    The bed is stored as a profile name plus its parameters rather than a
    sampled array. `topography.get_bed_profile(name, **params)` evaluated at
    the CSV's `x` column reproduces `mesh.set_bed_elevation` exactly, because
    both sample `mesh.cell_center_positions` - and `TopographySettings`
    carries a closure, which is not serialisable.
    """
    pde_section = _section(config, 'pde', required=True)
    numerics = _section(config, 'numerics', required=True)
    grid = _section(config, 'grid', required=True)
    topography_section = config.get('topography') or {}

    _pde = sim.pde_type
    _mesh = sim.mesh
    scheme = sim.spatial_discretization
    wet_dry = getattr(_pde, 'wet_dry', None)

    metadata = {
        'schema': SIDECAR_SCHEMA,
        'config_name': config_name,
        'config_path': str(config_path) if config_path is not None else None,
        'model': _text(pde_section, 'pde', 'type'),
        'hyperbolic': bool(getattr(_pde, 'hyperbolic', False)),
        'order': int(sim.order),
        'resolution': int(_mesh.resolution),
        'domain': [float(_mesh.boundaries[0]), float(_mesh.boundaries[1])],
        'initial_condition': sim.initial_condition,
        'boundary_condition': sim.boundary_condition,
        'scheme': type(scheme).__name__,
        'fvm_type': _text(numerics, 'numerics', 'fvm_type', 'PVM'),
        'time_integrator': type(sim.time_integration).__name__,
        'scheme_well_balanced': bool(getattr(scheme, 'well_balanced', False)),
        'scheme_viscosity_depends_on_timestep': bool(
            getattr(scheme, 'viscosity_depends_on_timestep', False)),
        'viscosity': _number(pde_section, 'pde', 'viscosity'),
        'slip_length': _number(pde_section, 'pde', 'slip_length'),
        't_end': _number(numerics, 'numerics', 't_end'),
        'elapsed_seconds': elapsed_seconds,
        'store_history': bool(sim.store_history),
        'history_stride': int(sim.history_stride),
        'store_hyperbolicity': bool(sim.store_hyperbolicity),
        'hyperbolicity_stride': int(sim.hyperbolicity_stride),
        'hyperbolicity_tol': float(sim.hyperbolicity_tol),
        # End-of-run scheme counters. Recorded even when zero, because a zero
        # means different things per scheme and the report has to be able to
        # tell them apart (LF/PRICE never eigendecompose at all).
        'scheme_counters': {
            'spectra_examined': int(getattr(scheme, 'spectra_examined', 0)),
            'nonhyperbolic_count': int(getattr(scheme, 'nonhyperbolic_count', 0)),
            'max_abs_imaginary_eigenvalue': float(
                getattr(scheme, 'max_abs_imaginary_eigenvalue', 0.0)),
            'tolerance': float(getattr(scheme, 'hyperbolicity_tolerance', 0.0)),
        },
        'mass_created_by_clamping': float(
            getattr(sim, 'mass_created_by_clamping', 0.0)),
        'grid': {key: _number(grid, 'grid', key)
                 for key in ('x1', 'x2') if key in grid},
        'bed_profile': topography_section.get('bed_profile') if topography_section else None,
        'bed_params': {
            key: _number(topography_section, 'topography', key)
            for key in topography_section if key not in _TOPOGRAPHY_RESERVED
        } if topography_section else {},
        'reference_water_level': (
            _number(topography_section, 'topography', 'reference_water_level', 1.0)
            if topography_section else None),
        'has_topography': bool(_mesh.has_topography),
        'wet_dry': ({'eps_div': wet_dry.eps_div, 'h_dry': wet_dry.h_dry,
                     'h_wet': wet_dry.h_wet} if wet_dry is not None else None),
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }

    if metadata['model'] == 'RechargeSWME1D':
        metadata['infiltration_type'] = _text(
            pde_section, 'pde', 'infiltration_type', 'horton').lower()
        metadata['rainfall_rate'] = _number(pde_section, 'pde', 'rainfall_rate')

    return metadata


def write_outputs(sim, data_array, output_dir: str, prefix: str, order: int,
                  metadata: dict | None = None) -> None:
    """Write the run's CSVs, and the metadata sidecar when one is supplied.

    Model-agnostic since Step 7: this used to sit inside the RechargeSWME1D
    branch, so a plain SWME1D/HSWME1D run produced nothing at all.
    """
    columns = _primitive_columns(order)
    if data_array.shape[1] != len(columns):
        raise ValueError(
            f"Mismatch between primitive output shape and expected columns. "
            f"Got {data_array.shape}, expected {len(columns)} columns."
        )

    pd.DataFrame(data_array, columns=columns).to_csv(f"{prefix}_final.csv", index=False)

    if getattr(sim, "history", None):
        history_frames = []
        summary_rows = []
        for entry in sim.history:
            step, time, snapshot = entry["step"], entry["time"], entry["data"]
            snapshot_df = pd.DataFrame(snapshot, columns=columns)
            snapshot_df.insert(0, "time", time)
            snapshot_df.insert(0, "step", step)
            history_frames.append(snapshot_df)

            summary = {
                "step": step,
                "time": time,
                "mean_h": snapshot_df["h"].mean(),
                "mean_u_m": snapshot_df["u_m"].mean(),
                "min_h": snapshot_df["h"].min(),
                "max_h": snapshot_df["h"].max(),
            }
            for i in range(1, order + 1):
                ai = f"a{i}"
                summary[f"mean_{ai}"] = snapshot_df[ai].mean()
                summary[f"min_{ai}"] = snapshot_df[ai].min()
                summary[f"max_{ai}"] = snapshot_df[ai].max()
            summary_rows.append(summary)

        pd.concat(history_frames, ignore_index=True).to_csv(
            f"{prefix}_field_history.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(
            f"{prefix}_summary_history.csv", index=False)

    # Gated on their own contents, not on the field history. Nesting these
    # inside the history block meant store_hyperbolicity without store_history
    # silently wrote nothing, while the common default wrote two empty files on
    # every run (RESTRUCTURE_PLAN.md Step 8.5, defect D2).
    #
    # Named from `prefix` like the other three (defect D6). They used to carry a
    # hardcoded `recharge_` stem, so a plain SWME run wrote files claiming to be
    # recharge output, and two models sharing an output directory collided on
    # them even though their other CSVs did not.
    if getattr(sim, "hyperbolicity_history", None):
        pd.DataFrame(sim.hyperbolicity_history).to_csv(
            f"{prefix}_hyperbolicity_history.csv", index=False)
    if getattr(sim, "hyperbolicity_summary", None):
        pd.DataFrame(sim.hyperbolicity_summary).to_csv(
            f"{prefix}_hyperbolicity_summary.csv", index=False)

    if metadata is not None:
        with open(f"{prefix}_run.json", 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2, sort_keys=False)
            handle.write("\n")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='moment-sw',
        description='1D finite-volume solver for the Shallow Water Moment '
                    'Equations (SWME/HSWME) and their rainfall-runoff '
                    '(recharge) extension.',
    )
    parser.add_argument(
        '-c', '--config', default=None,
        help='Config to run: a path, or the name of one shipped in '
             f'swme/config/. Default: {DEFAULT_CONFIG.name}')
    parser.add_argument(
        '-o', '--output-dir', default=None,
        help="Directory for result CSVs. Overrides postprocessing.output_dir.")
    parser.add_argument(
        '--list-configs', action='store_true',
        help='List the configs shipped in swme/config/ and exit.')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Print per-timestep progress. Off by default: it is four lines '
             'per step, which buries anything useful in a batch sweep.')
    parser.add_argument(
        '--plot', action='store_true',
        help='Show the interactive summary figure when the run finishes. Off '
             'by default: it blocks until the window is closed, which is wrong '
             'for scripted runs.')
    args = parser.parse_args(argv)

    if args.list_configs:
        for name in shipped_config_names():
            print(name)
        return None

    run(args.config, output_dir=args.output_dir, plot=args.plot,
        verbose=args.verbose)

    # Deliberately returns None. The console-script wrapper calls
    # `sys.exit(main())`, so returning the data array here would hand
    # `sys.exit` a non-int: it prints the object to stderr and exits 1, which
    # looks exactly like a crash even though the run and its CSVs are fine.
    # Callers who want the array should call `run()` directly.
    return None


def run(config=None, *, output_dir=None, plot=False, verbose=False):
    """Build and run a simulation from a config, returning the final state.

    The importable counterpart to `main()`: same behaviour, but returns the
    post-processed array `[x, h, u_m, a_1..a_N]` instead of a process exit code.
    """
    config_path = resolve_config(config)
    print(f"config: {config_path}")
    config = load_config(config_path)

    numerics = _section(config, 'numerics', required=True)
    postprocessing = _section(config, 'postprocessing')
    order = _integer(numerics, 'numerics', 'order')
    t_end = _number(numerics, 'numerics', 't_end')

    # Default to a per-config subdirectory of results/. Runs used to share
    # one directory and one set of filenames, so a second run silently
    # overwrote the first; segregating by config name makes that impossible
    # without anyone having to remember --output-dir.
    output_dir = (
        output_dir
        or postprocessing.get('output_dir')
        or os.path.join('results', config_path.stem)
    )
    os.makedirs(output_dir, exist_ok=True)

    _simulation = build_simulation(config, verbose=verbose)
    _mesh = _simulation.mesh
    _pde = _simulation.pde_type

    if _mesh.has_topography:
        print(f"topography: {config['topography'].get('bed_profile', 'flat')}, "
              f"Z in [{_mesh.bed_elevation.min():.6g}, {_mesh.bed_elevation.max():.6g}]")

    start = timeit.default_timer()
    data_array = _simulation.run_simulation(t_end)
    elapsed = timeit.default_timer() - start

    model = _text(_section(config, 'pde'), 'pde', 'type')
    infiltration_type = None
    if model == 'RechargeSWME1D':
        infiltration_type = _text(
            _section(config, 'pde'), 'pde', 'infiltration_type', 'horton').lower()
    prefix = _output_prefix(
        output_dir, model, _pde.hyperbolic, order, infiltration_type)
    metadata = build_run_metadata(
        _simulation, config, config_name=config_path.stem,
        config_path=config_path, elapsed_seconds=elapsed)
    write_outputs(_simulation, data_array, output_dir, prefix, order, metadata)

    print(f"Time: {elapsed}")
    print(f"model: {model}, hyperbolic: {_pde.hyperbolic}, N: {order}")
    print(f"output: {output_dir}")

    if plot:
        # Imported here, not at module scope: swme.plotting pulls in pyplot,
        # which every scripted run would otherwise pay for and never use.
        from . import plotting
        plotting.SWME1DPlotClassical(_pde, _mesh, _simulation).plot(data_array)

    return data_array


if __name__ == '__main__':
    main()
