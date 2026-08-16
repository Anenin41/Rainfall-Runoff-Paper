from abc import ABC, abstractmethod
import warnings

import numpy as np
from . import pde
from . import mesh
from . import spatialDiscretization
from . import timeIntegration

# How many times the positivity limiter may re-derive the timestep for schemes
# whose viscosity depends on it (LF, PRICE). Shrinking delta_t raises their
# viscosity, so the fixed point is not guaranteed to be reached in one pass;
# a small bound plus the clamp below is more robust than iterating to
# convergence. Roe and Osher need zero iterations.
_MAX_POSITIVITY_ITERATIONS = 3

# Depth below zero that counts as round-off rather than a real failure of the
# positivity limiter. Scaled to double precision on O(1) depths.
_NEGATIVE_HEIGHT_TOLERANCE = 1e-12


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

        # Per-timestep progress output. Off by default: printing four lines
        # per step produced a 1.9 MB log for a single thesis case, which is
        # useless in a batch sweep and drowns the diagnostics that matter.
        self.verbose = False

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

            # Skip dry states before trying to build the transport matrix.
            # The threshold is the model's *physical* dry threshold, not the
            # eigenvalue tolerance that happened to be reused here before
            # Step 6 - a cell below h_dry has had its moments ramped away, so
            # its spectrum says nothing about the flow.
            dry_threshold = getattr(
                getattr(self.pde_type, 'wet_dry', None), 'h_dry',
                self.hyperbolicity_tol)
            if ((not np.all(np.isfinite(local_values))) or
                local_values[0] <= dry_threshold):
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

        # Fresh counters per run, so a reused scheme object does not carry a
        # previous run's hyperbolicity loss into this one's report.
        if hasattr(self.spatial_discretization, 'reset_hyperbolicity_diagnostics'):
            self.spatial_discretization.reset_hyperbolicity_diagnostics()

        # Wet-dry bookkeeping (RESTRUCTURE_PLAN.md Step 6). `mass_created_by_
        # clamping` accumulates the water conjured by clamping round-off
        # negative depths back to zero. It should stay negligible; reporting it
        # is what turns "we clamp and hope" into a checkable claim.
        thresholds = getattr(self.pde_type, 'wet_dry', None)
        dry_threshold = getattr(thresholds, 'h_dry', 0.0)
        self.mass_created_by_clamping = 0.0
        
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

        # Bottom topography (RESTRUCTURE_PLAN.md Step 5).
        #
        # A non-flat bed is coupled in by widening the path-conservative state
        # from U to W = (U, Z) and integrating the augmented system matrix
        # A~(W) along the same linear path the moment transport already uses -
        # see pde._compute_augmented_system_matrix_generic. The Z row of the
        # resulting fluctuation is identically zero (the bed does not evolve),
        # so only the first `number_of_variables` entries are kept.
        #
        # `mesh.has_topography` is False for a bed that is zero everywhere, so
        # a flat-bed run never enters the augmented path and reproduces
        # pre-topography results bit for bit.
        use_topography = bool(getattr(self.mesh, "has_topography", False))
        if use_topography:
            bed_elevation = np.asarray(self.mesh.bed_elevation, dtype=np.float64)
            if bed_elevation.shape != (self.mesh.resolution + 2,):
                raise ValueError(
                    "mesh.bed_elevation must have one entry per cell including "
                    f"ghost cells, i.e. shape ({self.mesh.resolution + 2},), "
                    f"got {bed_elevation.shape}."
                )
            bed_boundary = getattr(self.mesh, "bed_boundary_condition", None)
            if bed_boundary is not None and bed_boundary != self.boundary_condition:
                raise ValueError(
                    f"The bed was sampled with boundary condition "
                    f"'{bed_boundary}' but this simulation runs with "
                    f"'{self.boundary_condition}'. The two ghost cells of Z "
                    "must be filled the same way the state's are, or the "
                    "interfaces at the domain edges see an inconsistent "
                    "(U, Z) pair."
                )
            if not hasattr(self.pde_type, "compute_augmented_system_matrix"):
                raise NotImplementedError(
                    f"{type(self.pde_type).__name__} has no "
                    "compute_augmented_system_matrix, so it cannot be run over "
                    "non-flat bottom topography."
                )
            if not getattr(self.spatial_discretization, "well_balanced", False):
                warnings.warn(
                    f"{type(self.spatial_discretization).__name__} is not "
                    "well balanced over topography: its numerical viscosity "
                    "has a nonzero constant term, so a lake at rest will not "
                    "be preserved exactly. Use Roe or Osher for topography "
                    "runs (see SpatialDiscretization.well_balanced).",
                    RuntimeWarning,
                    stacklevel=2,
                )

            # Reused per interface instead of reallocating (order+3,) twice
            # per interface per step; compute_fluctuation only reads them.
            augmented_left = np.empty(self.number_of_variables + 1)
            augmented_right = np.empty(self.number_of_variables + 1)

            def augmented_system_matrix(augmented_values):
                return self.pde_type.compute_augmented_system_matrix(
                    self.order, augmented_values)

        while t < t_end:

            # update boundary conditions
            values[0,:] = self._update_boundary_conditions(values,'left')
            values[self.mesh.resolution+1,:] = self._update_boundary_conditions(values,'right')
            
            max_speed = self.pde_type.compute_max_wavespeed(self.order,
                                                            values)

            delta_t = CFL*delta_x/max_speed

            def compute_all_fluctuations(step_size):
                if use_topography:
                    n_var = self.number_of_variables
                    for i in range(self.mesh.resolution+1):
                        augmented_left[:n_var] = values[i,:]
                        augmented_left[n_var] = bed_elevation[i]
                        augmented_right[:n_var] = values[i+1,:]
                        augmented_right[n_var] = bed_elevation[i+1]

                        fluctuation_min, fluctuation_plus = self.spatial_discretization.compute_fluctuation(
                            augmented_left,
                            augmented_right,
                            augmented_system_matrix,
                            step_size,
                            delta_x)

                        # Drop the Z row: it is zero by construction, and the
                        # state array has no Z column to write it into.
                        fluctuations_min[i,:] = fluctuation_min[:n_var]
                        fluctuations_plus[i,:] = fluctuation_plus[:n_var]
                else:
                    for i in range(self.mesh.resolution+1):
                        fluctuations_min[i,:],fluctuations_plus[i,:] = self.spatial_discretization.compute_fluctuation(
                            values[i,:],
                            values[i+1,:],
                            system_matrix,
                            step_size,
                            delta_x)

            compute_all_fluctuations(delta_t)

            # Positivity-preserving timestep limiter (RESTRUCTURE_PLAN.md
            # §2.3(c)). The mass update of cell i is
            #     h_i^{n+1} = h_i - dt/dx * (F+_{i-1,0} + F-_{i,0}),
            # so wherever that net outflow is positive it must not exceed the
            # water the cell actually holds. Capping dt pre-emptively is what
            # replaces the old crash-after-the-fact on non-positive height.
            #
            # For a wet, CFL-limited flow the cap is far looser than the CFL
            # condition and never binds, so this changes nothing about runs
            # that stay wet - the reason existing results are unaffected.
            # Cells already at or below h_dry are excluded. They hold no water
            # to protect, and a wet-dry interface can hand a dry cell a tiny
            # spurious positive outflow - for which the only admissible
            # timestep is exactly zero, which would deadlock the run over a
            # quantity smaller than h_dry. Their round-off excursions are
            # caught by the clamp after the update instead, and the mass that
            # creates is tracked and reported below.
            heights = values[1:self.mesh.resolution+1, 0]
            net_outflow = (fluctuations_plus[:self.mesh.resolution, 0]
                           + fluctuations_min[1:self.mesh.resolution+1, 0])
            draining = (net_outflow > 0.0) & (heights > dry_threshold)
            if np.any(draining):
                available = heights[draining]
                admissible = np.min(available * delta_x / net_outflow[draining])
                if admissible < delta_t:
                    delta_t = max(admissible, 0.0)
                    # Roe and Osher build their viscosity from |A| alone, so
                    # their fluctuations do not depend on delta_t and are still
                    # valid. LF and PRICE do depend on it, and shrinking dt
                    # *raises* their viscosity, so re-limit a bounded number of
                    # times rather than assuming one pass converges.
                    if getattr(self.spatial_discretization,
                               'viscosity_depends_on_timestep', True):
                        for _ in range(_MAX_POSITIVITY_ITERATIONS):
                            if delta_t <= 0.0:
                                break
                            compute_all_fluctuations(delta_t)
                            net_outflow = (
                                fluctuations_plus[:self.mesh.resolution, 0]
                                + fluctuations_min[1:self.mesh.resolution+1, 0])
                            draining = ((net_outflow > 0.0)
                                        & (heights > dry_threshold))
                            if not np.any(draining):
                                break
                            available = heights[draining]
                            admissible = np.min(
                                available * delta_x / net_outflow[draining])
                            if admissible >= delta_t:
                                break
                            delta_t = max(admissible, 0.0)
                    if delta_t <= 0.0:
                        raise RuntimeError(
                            "Positivity limiter drove the timestep to zero at "
                            f"step={step}, time={t}. The state is draining "
                            "faster than any positive timestep can follow; "
                            "check the wet-dry thresholds against the depth "
                            "scale of this problem."
                        )

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
                # h == 0 is a legitimate dry cell since Step 6, so this is no
                # longer a crash-on-sight. It is kept as the assertion that
                # the positivity limiter above did its job: a *negative* height
                # is still impossible, and reaching one means the limiter has a
                # bug rather than the problem being hard.
                if values[i, 0] < -_NEGATIVE_HEIGHT_TOLERANCE:
                    hint = ""
                    if getattr(self.pde_type, 'viscosity', 0.0) and not \
                            getattr(self.pde_type, 'linear_source', False):
                        hint = (
                            " NOTE: this run has nonzero viscosity and an "
                            "explicit source term. The Navier-slip friction "
                            "carries a nu/h^2 factor, so it becomes stiff near "
                            "a drying front and explicit integration of it "
                            "goes unstable - which shows up here, as a blown-up "
                            "moment dragging the height negative, rather than "
                            "as a failure of the positivity limiter. Use the "
                            "implicit source path (linear_source = True with "
                            "timeIntegrator = ImplicitEuler), which is exactly "
                            "what it is for."
                        )
                    raise RuntimeError(
                        f"Negative height produced after update  "
                        f"at step={step}, time={t}, cell={i-1}, x={self.mesh.cell_center_positions[i-1]},  "
                        f"h={values[i, 0]}, values={values[i, :]}. The "
                        f"positivity-preserving timestep limiter guarantees "
                        f"this cannot come from the flux update.{hint}"
                    )
                if values[i, 0] < 0.0:
                    # Round-off below the tolerance: clamp so the state stays
                    # physically meaningful rather than carrying a tiny
                    # negative depth into the next step's divisions. Water is
                    # created here, so keep a running total of how much.
                    self.mass_created_by_clamping -= values[i, 0] * delta_x
                    values[i, 0] = 0.0
            if self.verbose:
                print(f'  step {step}  t = {t:.6g}  dt = {delta_t:.6g}')


            t += delta_t
            step += 1

            # Store history snapshot if enabled
            self._store_snapshot(values, step = step, time = t)
            self._store_hyperbolicity_snapshot(values, step = step, time = t)

        # Hyperbolicity report. The scheme has already eigendecomposed the
        # transport matrix at every interface to build its viscosity, so this
        # costs nothing extra to collect - and without it a run whose spectrum
        # left the real axis finishes looking exactly like one that did not.
        # SWME is only unconditionally hyperbolic for N <= 1; from N = 2 up it
        # can lose hyperbolicity at large moments, which is what HSWME
        # (`hyperbolic=True`) is for. See tests/test_hyperbolicity.py.
        # Wet-dry mass audit. The positivity limiter guarantees h >= 0 for
        # every cell holding real water; cells already below h_dry are exempt
        # from it and can pick up round-off negatives that get clamped, which
        # creates water. Say so if it ever amounts to anything, rather than
        # leaving a silent conservation error.
        total_mass = float(np.sum(values[1:self.mesh.resolution+1, 0]) * delta_x)
        if (self.mass_created_by_clamping > 0.0 and total_mass > 0.0
                and self.mass_created_by_clamping > 1e-8 * total_mass):
            warnings.warn(
                f"Wet-dry clamping created "
                f"{self.mass_created_by_clamping:.3e} of mass "
                f"({self.mass_created_by_clamping / total_mass:.2e} of the "
                "total) by resetting round-off-negative depths to zero. This "
                "should be negligible; a large value means h_dry is too "
                "coarse for this problem's depth scale.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Be precise about what this measures: the *path-averaged* interface
        # matrix, not A(U) at any single state. Those are different questions,
        # and the averaged one leaves the real axis at a strong enough jump
        # even for models that are unconditionally hyperbolic pointwise - a
        # wet-dry front does it at N=0, i.e. for plain shallow water. The
        # wording below says so rather than blaming the model, because the
        # previous phrasing ("the N=... SWME system is not globally
        # hyperbolic") would have been flatly false in exactly the case Step 6
        # makes reachable.
        nonhyperbolic = getattr(self.spatial_discretization,
                                'nonhyperbolic_count', 0)
        if nonhyperbolic:
            examined = self.spatial_discretization.spectra_examined
            warnings.warn(
                f"Complex spectra at {nonhyperbolic} of {examined} interfaces "
                f"(largest |Im(lambda)| = "
                f"{self.spatial_discretization.max_abs_imaginary_eigenvalue:.3e}"
                "). This is the path-averaged interface matrix, which can "
                "leave the real axis at a strong jump - a wet-dry front does "
                "so even for plain shallow water - and is therefore not on "
                "its own evidence that the model lost hyperbolicity. To check "
                f"the model itself (the N={self.order} SWME system is only "
                "unconditionally hyperbolic for N <= 1; HSWME always is), set "
                "store_hyperbolicity = True for the cell-by-cell spectrum of "
                "A(U).",
                RuntimeWarning,
                stacklevel=2,
            )

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

