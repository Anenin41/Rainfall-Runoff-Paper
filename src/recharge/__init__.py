"""Rainfall / infiltration / exfiltration extension of the SWME solver.

Adds the boundary-exchange source terms (rainfall mass input, bed exchange,
and the associated mixing friction) on top of the base transport model in the
sibling `swme` package, which is left untouched: the extension enters purely
through source terms, for any moment order N.
"""

from .context import SourceContext
from .laws import AdmissibleMixingFriction, ConstantInfiltration, HortonInfiltration
from .recharge_pde import RechargeSWME1D
from .source_terms import (
    compute_mixing_friction,
    compute_recharge_mass_source,
    compute_total_friction,
)

__all__ = [
    "RechargeSWME1D",
    "SourceContext",
    "HortonInfiltration",
    "ConstantInfiltration",
    "AdmissibleMixingFriction",
    "compute_recharge_mass_source",
    "compute_mixing_friction",
    "compute_total_friction",
]
