# Packages
import numpy as np

# Local imports
from pde import SWME1D
from recharge.context import SourceContext
from recharge.source_terms import compute_total_source

class RechargeSWME1D(SWME1D):
    """
    Minimal SWME model equipped with rainfall and infiltration physics.

    Current implementation:
    - Inherits the SWME transport structure.
    - Evaluated mixing-friction through a configurable law object.
    - Builds a recharge mass production source term S_{R, I} in the back end.
    - Builds a composite friction vector P(U) in the back end.
    - Yields the total source S_{R, I} - P(U).
    - It supports N = 1 and N = 2. 
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
    ):
        # Inherit from the parent class stuff which are the same
        super().__init__(
            initial_condition = initial_condition,
            viscosity = viscosity,
            slip_length = slip_length,
            hyperbolic = hyperbolic,
            linear_source = linear_source,
        )

        # Initialize recharge specific attributes
        self.rainfall_rate = rainfall_rate
        self.infiltration_model = infiltration_model
        self.source_context = SourceContext()
        self.mixing_friction_model = mixing_friction_model

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
    
    # Compute the source term S_{R, I}
    # Order N = 0, 1, 2  are supported for now
    def compute_source_term(
            self, 
            order : int,
            values : np.ndarray,
            delta_t : float) -> np.ndarray:

        # If higher order is parsed, raise an error
        if order not in (0, 1, 2):
            raise NotImplementedError(
                f"RechargeSWME1D only supports order N=0, N=1 and N=2, got order={order}"
            )  
        
        # Get dt from the context
        self.source_context.dt = delta_t

        # Compute the rainfall-infiltration mass production source term
        source, _ = compute_total_source(
            order = order,
            values = values,
            rainfall= self.get_rainfall_rate,
            infiltration_model = self.infiltration_model,
            context = self.source_context,
            mixing_friction_model = self.mixing_friction_model,
            viscosity = self.viscosity,
            slip_length = self.slip_length,
        )

        return source