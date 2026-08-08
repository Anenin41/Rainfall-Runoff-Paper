# Packages & Local Imports
import simulation
import pde
import mesh
import spatialDiscretization
import timeIntegration
import plotting
import pandas as pd
import configparser
import timeit

# Recharge Specific Imports. Everything is written as a try-except block to
# avoid import errors in case the recharge module never merges with the main
# branch. 
try: 
    import os
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

def main():

    config = configparser.ConfigParser()
    config.read('config/config.txt')
    pde_information = config['pde_information']
    grid_information = config['grid_information']
    numerical_method_information = config['numerical_method_information']

    if HAS_RECHARGE:
        # Check if there exist a 'postprocessing' section on config.txt
        postprocessing = config['postprocessing']
    
        # Make a post-processing storade directory if one doesn't exist. 
        # Compartmentalize the recharge results in a separate folder but in 
        # the same Results/ directory.
        os.makedirs('Data-processing/Results/Recharge', exist_ok=True)
        

    linear_source = pde_information.getboolean('linear_source')
    time_integrator = numerical_method_information['timeIntegrator']
    linear_source_implicit = linear_source and time_integrator == 'ImplicitEuler'
    exact_source_computation = time_integrator == 'Exact'

    if pde_information['pde_type'] == 'SWME1D':
        _pde = pde.SWME1D(pde_information['initialCondition'],
                        pde_information.getfloat('viscosity'),
                        pde_information.getfloat('slipLength'),
                        False,
                        linear_source_implicit)
    elif pde_information['pde_type'] == 'HSWME1D':
        _pde = pde.SWME1D(pde_information['initialCondition'],
                        pde_information.getfloat('viscosity'),
                        pde_information.getfloat('slipLength'),
                        True,
                        linear_source_implicit)
        
    elif pde_information['pde_type'] == 'VegetationSWME1D':
        _pde = pde.VegetationSWME1D(pde_information['initialCondition'],
                                pde_information.getfloat('viscosity'),
                                pde_information.getfloat('slipLength'),
                                False,
                                linear_source_implicit,
                                0.008,
                                0.97,
                                800,
                                0.4)
    elif pde_information['pde_type'] == 'HME':
        _pde = pde.HermiteMomentEquations(
                        pde_information['initialCondition'],
                        pde_information.getfloat('relaxation_time'),
                        True,
                        True,
                        exact_source_computation)
    elif pde_information['pde_type'] == 'Grad':
        _pde = pde.HermiteMomentEquations(
                        pde_information['initialCondition'],
                        pde_information.getfloat('relaxation_time'),
                        False,
                        True,
                        exact_source_computation)
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

        if numerical_method_information['method'] == 'spatially_adaptive':
            start_order = int(numerical_method_information['start_order'])

            if numerical_method_information['coupling'] == 'nonconservative':
                _simulation = simulation.NonConservativeAdaptiveSimulation1D(
                    start_order,
                    _pde,
                    _mesh,
                    numerical_method_information['boundaryCondition'],
                    pde_information['initialCondition'],
                    pde_information['breakdown_criterion'],
                    _spatialDiscretization,
                    _time_integration
                )
            elif numerical_method_information['coupling'] == 'conservative':
                _simulation = simulation.ConservativeAdaptiveSimulation1D(
                    start_order,
                    _pde,
                    _mesh,
                    numerical_method_information['boundaryCondition'],
                    pde_information['initialCondition'],
                    pde_information['breakdown_criterion'],
                    _spatialDiscretization,
                    _time_integration
                )
        elif numerical_method_information['method'] == 'smoothedAdaptive':
            if numerical_method_information['coupling'] == 'nonconservative':
                start_order = int(numerical_method_information['start_order'])
                _simulation = simulation.SmoothedConsAdaptiveSimulation1D(
                    start_order,
                    _pde,
                    _mesh,
                    numerical_method_information['boundaryCondition'],
                    pde_information['initialCondition'],
                    pde_information['breakdown_criterion'],
                    _spatialDiscretization,
                    _time_integration)
            elif numerical_method_information['coupling'] == 'nonconservative':
                start_order = int(numerical_method_information['start_order'])
                _simulation = simulation.SmoothedNonConsAdaptiveSimulation1D(
                    start_order,
                    _pde,
                    _mesh,
                    numerical_method_information['boundaryCondition'],
                    pde_information['initialCondition'],
                    pde_information['breakdown_criterion'],
                    _spatialDiscretization,
                    _time_integration)                
        elif numerical_method_information['method'] == 'interpolatedAdaptive':
            start_order = int(numerical_method_information['start_order'])
            _simulation = simulation.InterpolatedAdaptiveSimulation1D(
                start_order,
                _pde,
                _mesh,
                numerical_method_information['boundaryCondition'],
                pde_information['initialCondition'],
                pde_information['breakdown_criterion'],
                _spatialDiscretization,
                _time_integration) 
        elif numerical_method_information['method'] == 'classical':
            _simulation = simulation.ClassicalSimulation1D(
                numerical_method_information.getint('order'),
                _pde,
                _mesh,
                numerical_method_information['boundaryCondition'],
                pde_information['initialCondition'],
                _spatialDiscretization,
                _time_integration)
            
        elif numerical_method_information['method'] == 'micro_macro':
            _simulation = simulation.Micro_macro(
                [int(order) for order in numerical_method_information['orders'].split(',')],
                _pde,
                _mesh,
                numerical_method_information['boundaryCondition'],
                pde_information['initialCondition'],
                _spatialDiscretization,
                _time_integration)
        
        # Modify the old plotting code to be slighly more readable.
        swme_plot_types = ['SWME1D', 'HSWME1D']
        if HAS_RECHARGE:
            swme_plot_types.append('RechargeSWME1D')

        if pde_information['pde_type'] in swme_plot_types:
            if numerical_method_information['method'] in [
                'spatially_adaptive',
                'smoothedAdaptive',
                'interpolatedAdaptive'
            ]:
                _plotting = plotting.SWME1DPlotAdaptive(_pde, _mesh, _simulation)
            elif numerical_method_information['method'] == 'classical':
                _plotting = plotting.SWME1DPlotClassical(_pde, _mesh, _simulation)
        elif pde_information['pde_type'] in ['HME', 'Grad']:
            if numerical_method_information['method'] in [
                'spatially_adaptive',
                'smoothedAdaptive',
                'interpolatedAdaptive'
            ]:
                _plotting = plotting.HME1DPlotAdaptive(_pde, _mesh, _simulation)
            elif numerical_method_information['method'] == 'classical':
                _plotting = plotting.HME1DPlotClassical(_pde, _mesh, _simulation)
    
        start = timeit.default_timer()
        if (
            HAS_RECHARGE 
            and pde_information['pde_type'] == 'RechargeSWME1D'
            and numerical_method_information['method'] == 'classical'
        ):
            # Store CSVs of general solution & hyperbolicity history
            _simulation.store_history = postprocessing.getboolean('store_history')
            _simulation.store_hyperbolicity = postprocessing.getboolean('store_hyperbolicity')

            # Store every 1 time step. Adjust to larger values to reduce storage.
            _simulation.history_stride = postprocessing.getint('history_stride')
            _simulation.hyperbolicity_stride = postprocessing.getint('hyperbolicity_stride')

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

            # Define output prefix for recharge results
            model_tag = "hswme" if _pde.hyperbolic else "swme"
            output_prefix = (
                f"Data-processing/Results/Recharge/"
                f"recharge_{model_tag}_N{order}_{infiltration_type}"
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

                # Store hyperbolicity CSVs
                pd.DataFrame(_simulation.hyperbolicity_history).to_csv(
                    "Data-processing/Results/Recharge/recharge_hyperbolicity_history.csv",
                    index = False,
                )

                pd.DataFrame(_simulation.hyperbolicity_summary).to_csv(
                    "Data-processing/Results/Recharge/recharge_hyperbolicity_summary.csv",
                    index = False,
                )

        stop = timeit.default_timer()
        print('Time: ', stop - start)
        print(
            f"RechargeHSWME: {pde_information.getboolean('hyperbolic')}")
        data_frame = pd.DataFrame(data_array)

        # Making the plotting call safe
        if '_plotting' in locals():
            _plotting.plot(data_array)
        else: 
            print("No plotting class defined for pde_type =", 
                  pde_information['pde_type'])
        # data_frame.to_csv('Data-processing/Output/test.csv', index=False,header=False)
        # data_frame.to_csv(
        #     'Data-processing/Results/KineticMomentEquations/smoothAndShockTube_order10_relaxation0.1_time1.0_3000.csv',
        #     index=False,
        #     header=False)
    else:
        print('2D not implemented yet')

if __name__ == '__main__':
    main()