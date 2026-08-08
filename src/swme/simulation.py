from abc import ABC, abstractmethod
import numpy as np
from . import pde
from . import mesh
from . import spatialDiscretization
from . import timeIntegration

class Simulation(ABC):

    """
    This interface represents a simulation.

    ...

    Attributes
    ----------
    pde_type : str
        the partial differential equations that is simulated
    mesh : RectangularMesh
        the used mesh
    boundary_condition : str
        the used boundary condition
    initial_condition : str
        the initial condition for the simulation
    spatial_discretization : spatial_discretization
        the numerical method for the spatial discretization

    
    Abstract methods
    -------
    def __init__(self):
        initializes the simulation object
    def run_simulation(self,t_end):
        runs the simulation and outputs the end values
    def _get_initial_conditions(self,cell_centers_x):
        constructs the initial values in each grid cell
    def _update_boundary_conditions(self,values_boundary):
        updates the boundary conditions
    def _post_processing(self,values):
        post processed the end data of the simulation and prepares it for plotting
    """

    @abstractmethod
    def __init__(self):
        """
        Implemented in the child classes.
        """
        pass

    @abstractmethod
    def run_simulation(self,
                       t_end: float) -> np.ndarray:
        """
        Runs the simulation until the end time t_end and returns the end values of the variables

        Parameters
        ----------
        t_end : float
            end time of the simulation
        
        Returns
        -------
        values: numpy arrays
            data array containing the positions of the grid cells and the values of the variables at the end of the simulation

        """
        pass

    @abstractmethod
    def _get_initial_conditions(self,
                               cell_centers):
        """
        Implemented and documented in the child classes. 
        """
        pass

    @abstractmethod
    def _update_boundary_conditions(self,
                                   values_boundary):
        """
        Implemented and documented in the child classes.
        """
        pass

    @abstractmethod
    def _post_processing(self,
                         end_values):
        """
        Post processes the end simulation data and prepares it for plotting

        Parameters
        ----------
        end_values : numpy array
            end values of the simulation
        
        Returns
        -------
        data_array: numpy arrays
            post processed data array containing values of the variables at the end of the simulation as 
            well as the cell center positions

        """
        pass

class ClassicalSimulation1D(Simulation):

    """
    This class represents a classical (not spatially adaptive) simulation in 1D.

    ...

    Attributes
    ----------
    order: int
        order of the moment model
    pde_type : str
        the partial differential equations that is simulated
    number_of_variables : int
        number of state variables
    mesh : RectangularMesh
        the used mesh
    boundary_condition: str
        the used boundary condition
    initial_condition: str
        the initial condition for the simulation
    spatial_discretization: SpatialDiscretization
        the numerical method for the spatial discretization
    time_integration: TimeIntegration
        the time integration method for the right-hand side source term

    
    Implemented methods from interface Simulation
    -------
    def run_simulation(self,t_end):
        runs the simulation and outputs the end values
    def _get_initial_conditions(self,cell_centers_x):
        constructs the initial values in each grid cell
    def _update_boundary_conditions(self,values_boundary):
        updates the boundary conditions
    def _post_processing(self,values):
        post processed the end data of the simulation and prepares it for plotting
    """

    def __init__(self,
                 order: int,
                 pde_type: pde.PDE,
                 mesh: mesh.RectangularMesh,
                 boundary_condition: str,
                 initial_condition: str,
                 spatial_discretization: spatialDiscretization.SpatialDiscretization,
                 time_integration: timeIntegration.TimeIntegration):
        """
        Constructs all the necessary attributes for the ClassicalSimulation1D object.

        Parameters
        ----------
        order: int
            order of the moment model
        pde_type : str
            the partial differential equations that is simulated
        number_of_variables : int
            number of state variables
        mesh : RectangularMesh
            the used mesh
        boundary_condition: str
            the used boundary condition
        initial_condition: str
            the initial condition for the simulation
        spatial_discretization: spatial_discretization
            the numerical method for the spatial discretization
        time_integration: TimeIntegration
            the time integration method for the right-hand side source term

        """
        self.order = order
        self.pde_type = pde_type
        self.number_of_variables = pde_type.compute_number_of_variables(self.order)
        self.mesh = mesh
        self.boundary_condition = boundary_condition
        self.initial_condition = initial_condition
        self.spatial_discretization = spatial_discretization
        self.time_integration = time_integration

        # Optional generic history storage
        # If enabled, post-processed snapshots of the solution are stored every
        # `self.history_stride` time steps. This mechanism is solver-agnostic
        # and can be used by any PDE model that needs time-history output for
        # debugging, post-processing, visualization or generic export.
        self.store_history = False                      # boolean storage flag
        self.history_stride = 10                        # stride for storing
        self.history = []                               # array to store in

        # Optional hyperbolicity diagnostics
        # If enabled, the solver checks the local transport matrix in each 
        # physical cell, computes its eigenvalues and stores both a detailed
        # cellwise log as well as a compact per-time summary.
        # Introduced to check hyperbolicity breakdown in the recharge model.
        self.store_hyperbolicity = False
        self.hyperbolicity_stride = 10
        self.hyperbolicity_tol = 1e-10
        self.hyperbolicity_history = []
        self.hyperbolicity_summary = []

    def _store_snapshot(self,
                        values : np.ndarray,
                        step : int,
                        time : float) -> None:
        """
        Optionally store a post-processed snapshot of the current solution.

        This method provides a generic time-history mechanism for the classical
        solver. If history storage is enabled, the current solution is 
        post-processed and appended to `self.history` at the prescribed stride
        interval.

        Parameters
        ----------
        values : numpy.ndarray
            Current state array, including ghost cells.
        step : int
            Current time-step index.
        time : float
            Physical time associated with the stored state..

        Returns
        -------
        None
        """
        if not self.store_history:
            return
        
        if step % self.history_stride != 0:
            return
        
        snapshot = self._post_processing(values.copy())
        self.history.append({
            "step" : step,
            "time" : time,
            "data" : snapshot,
        })

    def _store_hyperbolicity_snapshot(self,
                                      values : np.ndarray,
                                      step : int,
                                      time : float) -> None:
        """
        Optionally store hyperbolicity diagnostics for the current solution.

        For every physical cell, compute the local transport matrix
            A(U) = compute_system_matrix(order, U),
        evaluate its eigenvalues and store a cellwise diagnostic row. This is
        expected to be a costly operation, so enable it only in test-cases and 
        not when you want to solve large scale simulations. As a result, a 
        compact per-time-step summary is also stored.

        Parameters
        ----------
        values : numpy.ndarray
            Current state array, including ghost cells.
        step : int
            Current time-step index.
        time : float
            Physical time associated with the stored state.

        Returns
        -------
        None
        """
        # Check if the user wants a hyperbolicity check
        if not self.store_hyperbolicity:
            return
        
        # Perform hyperbolicity check only at the prescribed stride interval
        if step % self.hyperbolicity_stride != 0:
            return

        # Initialize util variables to store diagnostics
        n_bad = 0
        max_abs_imag_global = -1.0
        worst_cell_index = -1
        worst_x = np.nan
        worst_eigenvals = None

        # Iterate over physical cells and compute eigenvalues of the local
        # transport matrix. Store diagnostics.
        for i in range(1, self.mesh.resolution + 1):
            local_values = values[i, :].copy()
            x_i = float(self.mesh.cell_center_positions[i - 1])

            # Skip dry states before trying to build the transport matrix
            if ((not np.all(np.isfinite(local_values))) or
                local_values[0] <= self.hyperbolicity_tol):
                eigvals = np.full(self.number_of_variables, np.nan)
                real_parts = np.full(self.number_of_variables, np.nan)
                imag_parts = np.full(self.number_of_variables, np.nan)
                max_abs_imag = np.nan
                min_real = np.nan
                max_real = np.nan
                is_hyperbolic = 0
            else:
                try:
                    A = self.pde_type.compute_system_matrix(self.order, local_values)
                    eigvals = np.linalg.eigvals(A)

                    real_parts = np.real(eigvals)
                    imag_parts = np.imag(eigvals)

                    max_abs_imag = float(np.max(np.abs(imag_parts)))
                    min_real = float(np.min(real_parts))
                    max_real = float(np.max(real_parts))

                    is_hyperbolic = int(
                        np.isfinite(max_abs_imag) and
                        max_abs_imag < self.hyperbolicity_tol
                    )
                except Exception as e:
                    print("\n[hyperbolicity-check exception]")
                    print(f"step        = {step}")
                    print(f"time        = {time}")
                    print(f"cell        = {i - 1}")     # cell index
                    print(f"x           = {x_i}")       # cell center
                    print(f"values      = {local_values}")
                    print(f"error       = {type(e).__name__}: {e}\n")

                    eigvals = np.full(self.number_of_variables, np.nan + 1j * np.nan)
                    real_parts = np.full(self.number_of_variables, np.nan)
                    imag_parts = np.full(self.number_of_variables, np.nan)
                    max_abs_imag = np.nan
                    min_real = np.nan
                    max_real = np.nan
                    is_hyperbolic = 0
    
            if not is_hyperbolic: 
                n_bad += 1

            if np.isnan(max_abs_imag) or max_abs_imag > max_abs_imag_global:
                max_abs_imag_global = max_abs_imag
                worst_cell_index = i - 1
                worst_x = x_i
                worst_eigenvals = eigvals

            self.hyperbolicity_history.append({
                "step" : step,
                "time" : time,
                "cell_index" : i - 1,
                "x" : x_i,
                "eigvals" : eigvals,
                "max_abs_imag_eig" : max_abs_imag,
                "min_real_eig" : min_real,
                "max_real_eig" : max_real,
                "is_hyperbolic" : is_hyperbolic,
                "eigvals_real": ";".join([f"{val:.16e}" for val in real_parts]),
                "eigvals_imag": ";".join([f"{val:.16e}" for val in imag_parts]),
            })

        if worst_eigenvals is None:
            worst_eigvals_real = ""
            worst_eigvals_imag = ""
        else: 
            worst_eigvals_real = ";".join(
                [f"{val:.16e}" for val in np.real(worst_eigenvals)]
            )
            worst_eigvals_imag = ";".join(
                [f"{val:.16e}" for val in np.imag(worst_eigenvals)]
            )

        self.hyperbolicity_summary.append({
            "step": step,
            "time": time,
            "num_nonhyperbolic_cells": n_bad,
            "fraction_nonhyperbolic_cells": n_bad / self.mesh.resolution,
            "max_abs_imag_eig": max_abs_imag_global,
            "worst_cell_index": worst_cell_index,
            "worst_x": worst_x,
            "worst_eigvals_real": worst_eigvals_real,
            "worst_eigvals_imag": worst_eigvals_imag,
        })

    def run_simulation(self,
                       t_end: float,
                       g = 1) -> np.ndarray:

        delta_x = (self.mesh.boundaries[1] - self.mesh.boundaries[0])/self.mesh.resolution #TODO: include the possibility of nonuniform grids

        values = self._get_initial_conditions(self.mesh.cell_center_positions)
        fluctuations_min = np.zeros((self.mesh.resolution+1,self.number_of_variables))
        fluctuations_plus = np.zeros((self.mesh.resolution+1,self.number_of_variables))

        CFL = 0.5 #TODO: put CFL number in config file
        t = 0
        
        # Initialize history storage if enabled.
        # The initial condition is stored as snapshot 0 so that exported histories
        # include both the starting state and the later evolved states.
        step = 0 
        self.history = []
        self.hyperbolicity_history = []
        self.hyperbolicity_summary = []
        self._store_snapshot(values, step=0, time=t)
        self._store_hyperbolicity_snapshot(values, step=0, time=t)

        def system_matrix(cell_values):
            return self.pde_type.compute_system_matrix(self.order,cell_values)

        def source_term(cell_values,delta_t):
            return self.pde_type.compute_source_term(self.order,cell_values,delta_t)

        while t < t_end:

            # update boundary conditions
            values[0,:] = self._update_boundary_conditions(values,'left')
            values[self.mesh.resolution+1,:] = self._update_boundary_conditions(values,'right')
            
            max_speed = self.pde_type.compute_max_wavespeed(self.order,
                                                            values)

            delta_t = CFL*delta_x/max_speed 

            for i in range(self.mesh.resolution+1):
                fluctuations_min[i,:],fluctuations_plus[i,:] = self.spatial_discretization.compute_fluctuation(
                    values[i,:],
                    values[i+1,:],
                    system_matrix,
                    delta_t,
                    delta_x)          

            for i in range(1,self.mesh.resolution+1):
                values[i,:] = values[i,:] - delta_t/delta_x*(fluctuations_plus[i-1,:]+fluctuations_min[i,:])

                # Generic source-context hook.
                # The recharge PDE, as well as some other PDE models may require
                # runtime metadata in addition to the local cell state. Variables
                # like current time, timestep or cell position.
                # If the PDE object provides a `set_source_context(...)` method,
                # pass that information before the source integration step.
                if hasattr(self.pde_type, "set_source_context"):
                    x_i = self.mesh.cell_center_positions[i - 1]
                    self.pde_type.set_source_context(
                        time = t,
                        dt = delta_t,
                        cell_index = i - 1,
                        x = x_i,
                    )
                values[i,:] = self.time_integration.integrate(
                    values[i,:],
                    source_term,
                    delta_t)
                
                # Check is state is still finite (and physically meaningful)
                if not np.all(np.isfinite(values[i, :])):
                    raise RuntimeError(
                        f"Non-finite state produced after source integration  "
                        f"at step={step}, time={t}, cell={i-1}, x={self.mesh.cell_center_positions[i-1]},  "
                        f"values={values[i, :]}"
                    )
                if values[i, 0] <= 0.0:
                    raise RuntimeError(
                        f"Non-positive height produced after update  "
                        f"at step={step}, time={t}, cell={i-1}, x={self.mesh.cell_center_positions[i-1]},  "
                        f"h={values[i, 0]}, values={values[i, :]}"
                    )
            print()
            print('time: '+str(t))
            print('step size: '+str(delta_t))
            print()


            t += delta_t
            step += 1

            # Store history snapshot if enabled
            self._store_snapshot(values, step = step, time = t)
            self._store_hyperbolicity_snapshot(values, step = step, time = t)

        simulation_data = self._post_processing(values)
        return simulation_data

    def _get_initial_conditions(self,
                               cell_centers_x: np.ndarray) -> np.ndarray:

        """
        constructs the initial values for the variables

        Parameters
        ----------
        cell_centers_x : numpy 1D array
            the centers of the cells
        
        Returns
        -------
        initial_values: numpy 2D array
            initial values of the variables in each grid cell

        """
        
        initial_values = np.zeros((self.mesh.resolution+2,self.number_of_variables))

        for i in range(0,self.mesh.resolution):
            initial_values[i+1,:] = self.pde_type.get_initial_values(self.order,self.initial_condition,cell_centers_x[i])            
        
        return initial_values
    
    def _update_boundary_conditions(self,
                                   values: np.ndarray,
                                   boundary) -> np.ndarray:
        """
        update the boundary conditions

        Parameters
        ----------
        values : numpy 2D array
            values of the variables in each mesh cell
        boundary : str
            the boundary at which we are prescribing a boundary condition
        
        Returns
        -------
        values_ghost: numpy 1D array
            the values of the variables in the ghost cell

        """

        if self.boundary_condition == 'INFLOW_OUTFLOW':
            if boundary == 'left':
                values_ghost = values[1,:]
            else:
                values_ghost = values[-2,:]
        elif self.boundary_condition == 'PERIODIC':
            if boundary == 'left':
                values_ghost = values[-2,:]
            else:
                values_ghost = values[1,:]

        return values_ghost 
    
    def _post_processing(self,
                         values) -> np.ndarray:

        data_array = np.zeros((self.mesh.resolution,self.pde_type.compute_number_of_variables(self.order)+1)) # rewrite this such that it can be generalized to other PDE models

        for i in range(self.mesh.resolution):
            data_array[i,0] = self.mesh.cell_center_positions[i]

        data_array[:,1:] = values[1:-1,:]

        data_array[:,1:] = self.pde_type.convert_to_primitive(self.order,data_array[:,1:])

        return data_array

