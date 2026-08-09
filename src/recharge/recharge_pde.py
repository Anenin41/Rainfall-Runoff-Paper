# Packages
import numpy as np

# Local imports
from swme.pde import SWME1D
from swme.topography import TopographySettings
from .context import SourceContext
from .source_terms import compute_recharge_mass_source, compute_total_friction

class RechargeSWME1D(SWME1D):
    """
    Minimal SWME model equipped with rainfall and infiltration physics.

    Current implementation:
    - Inherits the SWME transport structure.
    - Evaluated mixing-friction through a configurable law object.
    - Builds a recharge mass production source term S_{R, I} in the back end.
    - Builds a composite friction vector P(U) in the back end.
    - Yields the total source S_{R, I} - P(U).
    - Supports arbitrary moment order N (the former N = 0, 1, 2 restriction was
      lifted in RESTRUCTURE_PLAN.md Step 3 by the generic coefficient engine).
    """

    # Constructor
    # If arguments are parsed with a different type it yields a TypeError
    def __init__(
            self,
            initial_condition : str,
            viscosity : float,
            slip_length : float,
            hyperbolic : bool,
            linear_source : bool,
            rainfall_rate : float,
            infiltration_model : object,
            mixing_friction_model : object,
            eps_dry : float = 1e-14,
            topography : TopographySettings | None = None,
    ):
        # Inherit from the parent class stuff which are the same
        super().__init__(
            initial_condition = initial_condition,
            viscosity = viscosity,
            slip_length = slip_length,
            hyperbolic = hyperbolic,
            linear_source = linear_source,
            topography = topography,
        )

        # Initialize recharge specific attributes
        self.rainfall_rate = rainfall_rate
        self.infiltration_model = infiltration_model
        self.source_context = SourceContext()
        self.mixing_friction_model = mixing_friction_model

        # Division-guard / dry-cell tolerance. This is the machine-precision
        # guard inherited from the legacy source terms, NOT a physically
        # meaningful dry threshold - the latter (h_dry / h_wet, with a proper
        # wetting-front treatment) arrives in RESTRUCTURE_PLAN.md Step 6.
        self.eps_dry = eps_dry

    # Manually set the context for the source terms
    # Probably going to be needed in the future
    def set_source_context(
            self,
            time : float,
            dt : float,
            cell_index : int | None = None,
            x : float | None = None,
    ) -> None:
        self.source_context.time = time
        self.source_context.dt = dt
        self.source_context.cell_index = cell_index
        self.source_context.x = x

    # Fetch rainfall rate
    # Context is necessary here in case of non-continuous rainfall profile
    # Such a profile is not yet implemented
    def get_rainfall_rate(self, context : SourceContext) -> float:
        return self.rainfall_rate
    
    # Compute the total source term S_{R, I}(U) - P(U), for arbitrary order N
    def compute_source_term(
            self,
            order : int,
            values : np.ndarray,
            delta_t : float) -> np.ndarray:
        """
        Total recharge source S_total(U) = S_{R, I}(U) - P(U), generic in the
        moment order N.

        P(U) = P_slip(U) + P_mix(U) combines the base model's Navier-slip bed
        friction (from `swme.source_terms`, present with or without recharge)
        with the rainfall/exfiltration-induced mixing friction contributed by
        this extension. The former N = 0, 1, 2 restriction is gone: arbitrary
        N follows from the generic shifted-Legendre coefficient engine (see
        RESTRUCTURE_PLAN.md Step 3).
        """
        # Get dt from the context, so the infiltration closures can apply
        # their available-water limiting over the current step
        self.source_context.dt = delta_t

        values = np.asarray(values, dtype = np.float64)
        h = values[0]

        # Dry-cell treatment: no exchange, no friction, no mixing
        if h <= self.eps_dry:
            return np.zeros(order + 2, dtype = np.float64)

        # Evaluate the local rainfall and bed-exchange rates
        R = self.get_rainfall_rate(self.source_context)
        I = self.infiltration_model.rate(
            t = self.source_context.time,
            rainfall = R,
            h = h,
            dt = self.source_context.dt,
        )

        # Evaluate the mixing-friction coefficients from the chosen closure
        f_R, f_I = self.mixing_friction_model.evaluate(
            R = R,
            I = I,
            context = self.source_context,
            values = values,
        )

        # S_{R, I}(U) - P(U)
        return compute_recharge_mass_source(
            order, values, R = R, I = I, eps_div = self.eps_dry,
        ) - compute_total_friction(
            order,
            values,
            f_R = f_R,
            f_I = f_I,
            viscosity = self.viscosity,
            slip_length = self.slip_length,
            eps_div = self.eps_dry,
        )