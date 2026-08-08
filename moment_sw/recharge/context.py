# Packages
from dataclasses import dataclass

@dataclass 
class SourceContext:
    """
    Context for the recharge source term in the nonlinear system.
    """

    time : float = 0.0                  # absolute time
    dt : float = 0.0                    # timestep
    cell_index : int | None = None      # cell index
    x : float | None = None             # cell center