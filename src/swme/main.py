# Packages & Local Imports
import argparse
import configparser
import os
import timeit
from pathlib import Path

import pandas as pd

from . import simulation
from . import pde
from . import mesh
from . import spatialDiscretization
from . import timeIntegration
from . import plotting
from . import topography as topography_module
from .wetdry import WetDryThresholds

_PACKAGE_DIR = Path(__file__).resolve().parent

# Recharge Specific Imports. Everything is written as a try-except block to
# avoid import errors in case the recharge module never merges with the main
# branch.
try:
    from recharge.initial_conditions import RechargeSWME1D_CustomIC as RechargeSWME1D
    from recharge.laws import (
        HortonInfiltration, ConstantInfiltration, AdmissibleMixingFriction,
    )

    # Define a global boolean flag that establishes if the recharge module 
    # was introduced at runtime.
    HAS_RECHARGE = True

    # Helper function to define the primitive output columns for SWME-style models.
    def _primitive_columns_for_swme(order : int) -> list[str]:
        """
        Primitive output columns produced by Simulation._post_processing(...)
        for SWME-style 1D models:
            [x, h, u_m, a1, ..., aN]
        """
        return ["x", "h", "u_m"] + [f"a{i}" for i in range(1, order + 1)]

except ImportError:
    # Silent fail, don't print a statement
    HAS_RECHARGE = False

DEFAULT_CONFIG = _PACKAGE_DIR / 'config' / 'config.ini'

# Keys of the optional [topography] section that configure the *water*, not
# the bed shape. Everything else in that section is forwarded to the chosen
# bed profile as a float parameter.
_TOPOGRAPHY_NON_PROFILE_KEYS = frozenset({
    'bed_profile',
    'reference_water_level',
    'perturbation_amplitude',
    'perturbation_center',
    'perturbation_width',
})


def _build_topography(config) -> topography_module.TopographySettings:
    """Build TopographySettings from the config's optional [topography] section.

    A config without that section - i.e. every config written before
    RESTRUCTURE_PLAN.md Step 5 - gets the default flat bed at zero, which
    leaves `mesh.has_topography` False and the solver on its original
    non-augmented path.

    Recognized keys::

        [topography]
        bed_profile           = gaussian_bump   # see topography.get_bed_profile
        reference_water_level = 1.0             # free surface H of a lake at rest
        perturbation_amplitude = 0.0            # for 'perturbedLakeAtRest'
        perturbation_center    = 0.0
        perturbation_width     = 1.0
        # ... any remaining keys are the bed profile's own float parameters,
        # e.g. amplitude / center / width for gaussian_bump

    An unknown profile parameter raises rather than being ignored: a silently
    dropped typo would produce a plausible-looking but wrong bed.
    """
    if not config.has_section('topography'):
        return topography_module.TopographySettings()

    section = config['topography']
    profile_name = section.get('bed_profile', fallback='flat')

    profile_params = {}
    for key in section:
        if key in _TOPOGRAPHY_NON_PROFILE_KEYS:
            continue
        try:
            profile_params[key] = section.getfloat(key)
        except ValueError as exc:
            raise ValueError(
                f"[topography] parameter '{key}' must be a number, got "
                f"'{section[key]}'."
            ) from exc

    bed_elevation = topography_module.get_bed_profile(
        profile_name, **profile_params)

    return topography_module.TopographySettings(
        bed_elevation = bed_elevation,
        reference_water_level = section.getfloat(
            'reference_water_level', fallback=1.0),
        perturbation_amplitude = section.getfloat(
            'perturbation_amplitude', fallback=0.0),
        perturbation_center = section.getfloat(
            'perturbation_center', fallback=0.0),
        perturbation_width = section.getfloat(
            'perturbation_width', fallback=1.0),
    )


def _build_wet_dry(config) -> WetDryThresholds:
    """Build WetDryThresholds from the config's optional [wet_dry] section.

    A config without that section gets the defaults (h_dry = 1e-4,
    h_wet = 1e-3), which are sized for the thesis' h ~ O(1) cases and never
    activate in a run that stays well above them - so every pre-Step-6 config
    is unaffected. These are absolute depths, not ratios: scale them to the
    depth scale of the problem.

        [wet_dry]
        h_dry   = 1e-4      # at/below: dry, no moments, velocity driven to 0
        h_wet   = 1e-3      # at/above: ordinary wet flow, no regularization
        eps_div = 1e-14     # machine-precision division guard
    """
    if not config.has_section('wet_dry'):
        return WetDryThresholds()

    section = config['wet_dry']
    defaults = WetDryThresholds()
    return WetDryThresholds(
        eps_div = section.getfloat('eps_div', fallback=defaults.eps_div),
        h_dry = section.getfloat('h_dry', fallback=defaults.h_dry),
        h_wet = section.getfloat('h_wet', fallback=defaults.h_wet),
    )


def _resolve_config(name_or_path) -> Path:
    """Resolve a --config value to a readable file.

    Accepts a filesystem path, or the bare name of a config shipped in
    `swme/config/` (with or without the .ini suffix), so thesis cases can be
    run as e.g. `--config thesis_5p3_pulse_N1` from any working directory.
    """
    if name_or_path is None:
        return DEFAULT_CONFIG

    candidates = [Path(name_or_path)]
    stem = Path(name_or_path).name
    candidates += [
        _PACKAGE_DIR / 'config' / stem,
        _PACKAGE_DIR / 'config' / f'{stem}.ini',
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    available = sorted(p.stem for p in (_PACKAGE_DIR / 'config').glob('*.ini'))
    raise FileNotFoundError(
        f"Config '{name_or_path}' not found. Tried: "
        + ", ".join(str(c) for c in candidates)
        + ".\nAvailable shipped configs: "
        + ", ".join(available)
    )


def main(argv=None):

    parser = argparse.ArgumentParser(
        prog='moment-sw',
        description='1D finite-volume solver for the Shallow Water Moment '
                    'Equations (SWME/HSWME) and their rainfall-runoff '
                    '(recharge) extension.',
    )
    parser.add_argument(
        '-c', '--config', default=None,
        help='Config file to run: a path, or the name of one shipped in '
             f'swme/config/. Default: {DEFAULT_CONFIG.name}',
    )
    parser.add_argument(
        '-o', '--output-dir', default=None,
        help='Directory for result CSVs. Overrides the config\'s '
             'postprocessing/recharge_output_dir.',
    )
    parser.add_argument(
        '--list-configs', action='store_true',
        help='List the config files shipped in swme/config/ and exit.',
    )
    args = parser.parse_args(argv)

    if args.list_configs:
        for path in sorted((_PACKAGE_DIR / 'config').glob('*.ini')):
            print(path.stem)
        return

    config_path = _resolve_config(args.config)
    print(f"config: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)
    pde_information = config['pde_information']
    grid_information = config['grid_information']
    numerical_method_information = config['numerical_method_information']

    # Output directory: --output-dir wins, else the config, else the default.
    postprocessing = config['postprocessing'] if config.has_section('postprocessing') else {}
    output_dir = (
        args.output_dir
        or (postprocessing.get('recharge_output_dir') if postprocessing else None)
        or 'Data-processing/Results/Recharge'
    )
    os.makedirs(output_dir, exist_ok=True)


    linear_source = pde_information.getboolean('linear_source')
    time_integrator = numerical_method_information['timeIntegrator']
    linear_source_implicit = linear_source and time_integrator == 'ImplicitEuler'

    # Optional bottom topography; defaults to a flat bed at zero, i.e. exactly
    # the pre-Step-5 behavior for every config without a [topography] section.
    _topography = _build_topography(config)

    # Wet-dry thresholds; defaults never activate above h ~ 1e-3, so configs
    # written before Step 6 behave exactly as they did.
    _wet_dry = _build_wet_dry(config)

    if pde_information['pde_type'] == 'SWME1D':
        _pde = pde.SWME1D(pde_information['initialCondition'],
                        pde_information.getfloat('viscosity'),
                        pde_information.getfloat('slipLength'),
                        False,
                        linear_source_implicit,
                        topography = _topography,
                        wet_dry = _wet_dry)
    elif pde_information['pde_type'] == 'HSWME1D':
        _pde = pde.SWME1D(pde_information['initialCondition'],
                        pde_information.getfloat('viscosity'),
                        pde_information.getfloat('slipLength'),
                        True,
                        linear_source_implicit,
                        topography = _topography,
                        wet_dry = _wet_dry)

    elif pde_information['pde_type'] == 'RechargeSWME1D':
        if not HAS_RECHARGE:
            raise ImportError(
                "Config requests pde_type='RechargeSWME1D' but the recharge module  "
                "is not available in this branch/environment."
            )
        
        # Choose infiltration model from config
        infiltration_type = pde_information.get('infiltration_type').lower()
        
        # Branches that define the settings of the infiltration model
        if infiltration_type == "horton":
            infiltration_model = HortonInfiltration(
                pde_information.getfloat('horton_f0'),
                pde_information.getfloat('horton_fc'),
                pde_information.getfloat('horton_k')
            )
        elif infiltration_type == "constant":
            infiltration_model = ConstantInfiltration(
                I0 = pde_information.getfloat('constant_infiltration_rate'),
                eps = pde_information.getfloat('constant_infiltration_eps',
                                               fallback=1e-14),
                limit_by_rainfall = pde_information.getboolean(
                    "constant_limit_by_rainfall", fallback=False),
                limit_by_available_water = pde_information.getboolean(
                    "constant_limit_by_available_water", fallback=True)
            )
        else: 
            raise ValueError(
                f"Unknown infiltration_type = '{infiltration_type}'.  "
                "Supported choices are : 'horton' and 'constant'."
            )

        # Choose mixing-friction model from config with a fallback option 
        # always being the admissible model defined by the control-volume.
        # Admissible closure is evaluated locally inside the source term layer
        mixing_friction_type = pde_information.get(
            "mixing_friction_model",
            fallback="admissible"
        ).lower()

        # Build the mixing friction model
        if mixing_friction_type == "admissible":
            mixing_friction_model = AdmissibleMixingFriction(
                alpha_R = pde_information.getfloat("alpha_R"),
                alpha_I = pde_information.getfloat("alpha_I"),
            )
        else:
            raise ValueError(
                f"Unknown mixing_friction_model = '{mixing_friction_type}'.  "
                "Supported choices are: 'admissible'."
            )
            

        # Build the pde object
        _pde = RechargeSWME1D(
            pde_information['initialCondition'],
            pde_information.getfloat('viscosity'),
            pde_information.getfloat('slipLength'),
            pde_information.getboolean('hyperbolic'),    
            pde_information.getboolean('linear_source', fallback=False),
            pde_information.getfloat('rainfall_rate'),
            infiltration_model,
            mixing_friction_model,
            topography = _topography,
            wet_dry = _wet_dry,
            )
    else:
        print('PDE_type is not implemented yet')
    
    ##########################################################################

    if numerical_method_information['fvm_type'] == 'PVM':
        if numerical_method_information['pvm'] == 'PRICE':
            _spatialDiscretization = spatialDiscretization.PRICE()
        elif numerical_method_information['pvm'] == 'LF':
            _spatialDiscretization = spatialDiscretization.LF()
        elif numerical_method_information['pvm'] == 'Roe':
            _spatialDiscretization = spatialDiscretization.Roe()
        elif numerical_method_information['pvm'] == 'Osher':
            _spatialDiscretization = spatialDiscretization.Osher()
        else:
            print('this pvm method is not implemented yet')
    else:
        print('this finite volume type is not implemented yet')

    if numerical_method_information['timeIntegrator'] == 'ImplicitEuler':
        _time_integration = timeIntegration.ImplicitEuler(linear_source)
    elif numerical_method_information['timeIntegrator'] == 'ExplicitEuler':
        _time_integration = timeIntegration.ExplicitEuler()
    elif numerical_method_information['timeIntegrator'] == 'Exact':
        _time_integration = timeIntegration.Exact()

    #########################################################################

    if pde_information.getboolean('1D'):

        _mesh = mesh.UniformRectangularMesh1D([grid_information.getfloat('x1boundary'),grid_information.getfloat('x2boundary')],
                                               grid_information.getint('resolutionX')) #TODO: Implement different grids

        # Sample the bed onto the grid (including ghost cells). Filled with
        # the same boundary condition as the state, or the interface
        # fluctuations at the domain edges would see an inconsistent (U, Z)
        # pair. A bed that evaluates to zero everywhere leaves
        # `_mesh.has_topography` False and the solver on its original path.
        if _topography.bed_elevation is not None:
            _mesh.set_bed_elevation(
                _topography.bed_elevation,
                numerical_method_information['boundaryCondition'],
            )
            if _mesh.has_topography:
                print(
                    "topography: "
                    f"{config['topography'].get('bed_profile', 'flat')}, "
                    f"Z in [{_mesh.bed_elevation.min():.6g}, "
                    f"{_mesh.bed_elevation.max():.6g}]"
                )

        # Only the classical (non-adaptive) driver remains; the spatially
        # adaptive and micro-macro drivers were removed in
        # RESTRUCTURE_PLAN.md Step 4 along with the models they served.
        if numerical_method_information['method'] != 'classical':
            raise ValueError(
                f"Unsupported method = '{numerical_method_information['method']}'. "
                "Only 'classical' is available."
            )

        _simulation = simulation.ClassicalSimulation1D(
            numerical_method_information.getint('order'),
            _pde,
            _mesh,
            numerical_method_information['boundaryCondition'],
            pde_information['initialCondition'],
            _spatialDiscretization,
            _time_integration)

        _plotting = plotting.SWME1DPlotClassical(_pde, _mesh, _simulation)
    
        start = timeit.default_timer()
        if (
            HAS_RECHARGE 
            and pde_information['pde_type'] == 'RechargeSWME1D'
            and numerical_method_information['method'] == 'classical'
        ):
            # Store CSVs of general solution & hyperbolicity history.
            # Defaults let a config omit the [postprocessing] section entirely.
            _simulation.store_history = config.getboolean(
                'postprocessing', 'store_history', fallback=True)
            _simulation.store_hyperbolicity = config.getboolean(
                'postprocessing', 'store_hyperbolicity', fallback=False)

            # Store every 1 time step. Adjust to larger values to reduce storage.
            _simulation.history_stride = config.getint(
                'postprocessing', 'history_stride', fallback=1)
            _simulation.hyperbolicity_stride = config.getint(
                'postprocessing', 'hyperbolicity_stride', fallback=1)

        data_array = _simulation.run_simulation(numerical_method_information.getfloat('t_end'))

        # Recharge specific post-processing
        if HAS_RECHARGE and pde_information['pde_type'] == 'RechargeSWME1D':
            if numerical_method_information['method'] != 'classical':
                raise NotImplementedError(
                    "Recharge post-processing is currently implemented only for  "
                    "the classical 1D solver."
                )
            # Extract order and infiltration type for output labeling
            order = getattr(_simulation, "order",
                            numerical_method_information.getint('order'))
            infiltration_type = pde_information.get(
                'infiltration_type', fallback="horton").lower()
            
            # Define the expected primitive output columns for SWME models
            primitive_columns = _primitive_columns_for_swme(order)

            # Check if the data array has the expected shape
            if data_array.shape[1] != len(primitive_columns):
                raise ValueError(
                    "Mismatch between primitive output shape and expected SWME  "
                    f"columns. Got {data_array.shape}, expected  "
                    f"{len(primitive_columns)} columns."
                )

            # Define output prefix for recharge results. Deliberately the
            # plain legacy pattern (no config-name prefix): the processing/
            # scripts' ROOT_DIR-based comparisons expect exactly this naming
            # inside a per-case subfolder, e.g.
            # <root>/Non_Wrapping_Pulse_N1/recharge_swme_N1_constant_field_history.csv.
            # Uniqueness across runs is the CALLER's responsibility via
            # --output-dir (one directory per case) - see
            # scripts/run_thesis_configs.sh, which does exactly that. Two
            # configs sharing one --output-dir will still silently overwrite
            # each other; this was a real bug found while validating the
            # thesis §5.1 cases (RESTRUCTURE_PLAN.md Step 4.5), traded off
            # here in favor of matching the already-written comparison
            # scripts instead of updating all of them to a new naming scheme.
            model_tag = "hswme" if _pde.hyperbolic else "swme"
            output_prefix = os.path.join(
                output_dir, f"recharge_{model_tag}_N{order}_{infiltration_type}"
            )

            # Final snapshot
            final_df = pd.DataFrame(data_array, columns=primitive_columns)
            final_df.to_csv(f"{output_prefix}_final.csv", index=False)

            # Full time history
            if hasattr(_simulation, "history") and len(_simulation.history) > 0:
                history_frames = []
                summary_rows = []

                # Iterate through the stored history and build dataframes
                for entry in _simulation.history:
                    step = entry["step"]
                    time = entry["time"]
                    snapshot = entry["data"]

                    # Build a dataframe for this snapshot and append to the history list
                    snapshot_df = pd.DataFrame(snapshot, columns=primitive_columns)
                    snapshot_df.insert(0, "time", time)
                    snapshot_df.insert(0, "step", step)
                    history_frames.append(snapshot_df)

                    # Build a summary row for this snapshot
                    summary = {
                        "step": step,
                        "time": time,
                        "mean_h": snapshot_df["h"].mean(),
                        "mean_u_m": snapshot_df["u_m"].mean(),
                        "min_h": snapshot_df["h"].min(),
                        "max_h": snapshot_df["h"].max(),
                    }

                    # Add summary statistics for the a_i's
                    for i in range(1, order + 1):
                        ai = f"a{i}"
                        summary[f"mean_{ai}"] = snapshot_df[ai].mean()
                        summary[f"min_{ai}"] = snapshot_df[ai].min()
                        summary[f"max_{ai}"] = snapshot_df[ai].max()

                    summary_rows.append(summary)

                # Concatenate all snapshot dataframes into a single history dataframe and save
                field_history_df = pd.concat(history_frames, ignore_index=True)
                field_history_df.to_csv(
                    f"{output_prefix}_field_history.csv", index=False,
                )
                
                # Create a summary dataframe with one row per time step and save
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(
                    f"{output_prefix}_summary_history.csv", index=False,
                )

                # Store hyperbolicity CSVs. Plain (not prefixed by order/tag)
                # to match what processing/*.py's ROOT_DIR-based comparisons
                # expect inside each case's own output directory:
                # "recharge_hyperbolicity_summary.csv", not an order-suffixed
                # name - one output directory per case (see
                # scripts/run_thesis_configs.sh) is what keeps these unique.
                pd.DataFrame(_simulation.hyperbolicity_history).to_csv(
                    os.path.join(output_dir, "recharge_hyperbolicity_history.csv"),
                    index = False,
                )

                pd.DataFrame(_simulation.hyperbolicity_summary).to_csv(
                    os.path.join(output_dir, "recharge_hyperbolicity_summary.csv"),
                    index = False,
                )

        stop = timeit.default_timer()
        print('Time: ', stop - start)
        print(
            f"RechargeHSWME: {pde_information.getboolean('hyperbolic')}")
        _plotting.plot(data_array)
    else:
        print('2D not implemented yet')

if __name__ == '__main__':
    main()