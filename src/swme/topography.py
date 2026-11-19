"""Bottom topography for the 1D solver: the bed-elevation profiles Z(x) and the
settings bundle that carries them into the PDE object.

Where the pieces live (RESTRUCTURE_PLAN.md Step 5):

* the *shape* Z(x) is a plain callable built here by `get_bed_profile`;
* the *sampled* Z on the grid (including ghost cells) lives on the mesh, via
  `mesh.UniformRectangularMesh1D.set_bed_elevation` - the PDE objects stay
  stateless with respect to the grid;
* the *coupling* into the scheme is the augmented (U, Z) path-conservative
  system matrix, `pde.SWME1D.compute_augmented_system_matrix`;
* `TopographySettings` is the small bundle the PDE needs in order to build
  topography-aware initial conditions (`lakeAtRest`, `perturbedLakeAtRest`,
  `lakeWithDryShores`), which cannot be written as a function of position
  alone.

A flat bed is the default everywhere, so nothing in this module changes the
behavior of a config that does not mention topography.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

BedElevation = Callable[[np.ndarray], np.ndarray]


def flat(elevation: float = 0.0) -> BedElevation:
    """Z(x) = elevation (0 by default). The no-topography case."""

    def z_of_x(x):
        return np.zeros_like(np.asarray(x, dtype=np.float64)) + elevation

    return z_of_x


def linear_slope(slope: float = 0.0, x_ref: float = 0.0,
                 elevation: float = 0.0) -> BedElevation:
    """Z(x) = elevation + slope*(x - x_ref). A constantly inclined bed."""

    def z_of_x(x):
        return elevation + slope * (np.asarray(x, dtype=np.float64) - x_ref)

    return z_of_x


def gaussian_bump(amplitude: float = 0.2, center: float = 0.5,
                  width: float = 0.1) -> BedElevation:
    """Z(x) = amplitude * exp(-((x - center)/width)^2).

    Smooth and everywhere-differentiable, so it isolates the well-balancing
    question from any discontinuity effects.
    """
    if width <= 0.0:
        raise ValueError(f"gaussian_bump width must be > 0, got {width}.")

    def z_of_x(x):
        xi = (np.asarray(x, dtype=np.float64) - center) / width
        return amplitude * np.exp(-xi * xi)

    return z_of_x


def parabolic_bump(amplitude: float = 0.2, center: float = 10.0,
                   half_width: float = 2.0) -> BedElevation:
    """Z(x) = max(0, amplitude*(1 - ((x - center)/half_width)^2)).

    The classical Goutal-Maisonneuve bump used across the well-balanced
    shallow-water literature. Continuous but only C^0 at the bump edges, which
    makes it a slightly harder test than `gaussian_bump`. Default parameters
    match the standard [0, 25] channel benchmark; scale them to the domain in
    use.
    """
    if half_width <= 0.0:
        raise ValueError(
            f"parabolic_bump half_width must be > 0, got {half_width}."
        )

    def z_of_x(x):
        xi = (np.asarray(x, dtype=np.float64) - center) / half_width
        return np.maximum(0.0, amplitude * (1.0 - xi * xi))

    return z_of_x


def sinusoidal(amplitude: float = 0.1, wavelength: float = 1.0,
               phase: float = 0.0, elevation: float = 0.0) -> BedElevation:
    """Z(x) = elevation + amplitude*sin(2*pi*(x - phase)/wavelength).

    Periodic in x, so it is the profile to use with `boundaryCondition =
    PERIODIC` - pick a wavelength that divides the domain length, otherwise
    the periodic ghost cells introduce an artificial bed jump at the wrap.
    """
    if wavelength <= 0.0:
        raise ValueError(f"sinusoidal wavelength must be > 0, got {wavelength}.")

    def z_of_x(x):
        xi = np.asarray(x, dtype=np.float64)
        return elevation + amplitude * np.sin(
            2.0 * np.pi * (xi - phase) / wavelength
        )

    return z_of_x


def step(amplitude: float = 0.1, position: float = 0.5,
         elevation: float = 0.0) -> BedElevation:
    """Z(x) = elevation + (amplitude if x >= position else 0).

    A genuinely discontinuous bed. Useful precisely because it is the hardest
    case for well-balancing: the equilibrium jump across the step interface is
    O(1) rather than O(dx).
    """

    def z_of_x(x):
        xi = np.asarray(x, dtype=np.float64)
        return elevation + np.where(xi >= position, amplitude, 0.0)

    return z_of_x


def tanh_step(amplitude: float = 0.1, position: float = 0.5,
              width: float = 0.05, elevation: float = 0.0) -> BedElevation:
    """Z(x) = elevation + amplitude/2 * (1 + tanh((x - position)/width)).

    A smoothed `step`; the smoothing width controls how many cells the
    transition is resolved over.
    """
    if width <= 0.0:
        raise ValueError(f"tanh_step width must be > 0, got {width}.")

    def z_of_x(x):
        xi = np.asarray(x, dtype=np.float64)
        return elevation + 0.5 * amplitude * (
            1.0 + np.tanh((xi - position) / width)
        )

    return z_of_x


def basin_dam(bank_height: float = 0.6, bank_position: float = 1.0,
              bank_width: float = 0.6, dam_height: float = 0.5,
              dam_position: float = 6.0, dam_width: float = 0.4,
              downstream_slope: float = 0.05) -> BedElevation:
    """A basin between a ridge on the left and a dam on the right.

        Z(x) = bank_height * exp(-((x - bank_position)/bank_width)^2)
               + dam_height * exp(-((x - dam_position)/dam_width)^2)
               - downstream_slope * max(x - dam_position, 0)

    Three pieces:

    * the bank: a Gaussian ridge. Its crest is a divide: water on its inner
      side runs into the basin, water on its outer side runs away from it,
      so an open (zero-gradient) boundary placed beyond the crest lets water
      out and feeds none in;
    * the dam: a Gaussian hump whose crest, at `dam_position`, stands
      `dam_height` above the basin floor;
    * the valley below the dam: the bed falls away linearly from the crest
      line, so water that overtops the dam keeps running to the outflow.

    Between the two humps the floor is at Z = 0 up to their tails. A lake
    fills the basin up to the lower of the two crests and then spills.
    """
    if bank_width <= 0.0:
        raise ValueError(f"basin_dam bank_width must be > 0, got {bank_width}.")
    if dam_width <= 0.0:
        raise ValueError(f"basin_dam dam_width must be > 0, got {dam_width}.")
    if dam_position <= bank_position:
        raise ValueError(
            f"basin_dam needs dam_position ({dam_position}) to the right of "
            f"bank_position ({bank_position})."
        )

    def z_of_x(x):
        xi = np.asarray(x, dtype=np.float64)
        xi_bank = (xi - bank_position) / bank_width
        bank = bank_height * np.exp(-xi_bank * xi_bank)
        eta = (xi - dam_position) / dam_width
        dam = dam_height * np.exp(-eta * eta)
        valley = downstream_slope * np.maximum(xi - dam_position, 0.0)
        return bank + dam - valley

    return z_of_x


_PROFILE_BUILDERS: dict[str, Callable[..., BedElevation]] = {
    "basin_dam": basin_dam,
    "flat": flat,
    "linear_slope": linear_slope,
    "gaussian_bump": gaussian_bump,
    "parabolic_bump": parabolic_bump,
    "sinusoidal": sinusoidal,
    "step": step,
    "tanh_step": tanh_step,
}


def available_bed_profiles() -> list[str]:
    """Names accepted by `get_bed_profile`, sorted."""
    return sorted(_PROFILE_BUILDERS)


def get_bed_profile(name: str, **params) -> BedElevation:
    """Build a bed-elevation callable Z(x) by name.

    Parameters
    ----------
    name : str
        One of `available_bed_profiles()`, case-insensitive.
    **params
        Profile-specific parameters; see the individual builders above. An
        unknown parameter name raises `TypeError`, deliberately - a silently
        ignored typo in a config would produce a plausible-looking but wrong
        bed.

    Returns
    -------
    Callable mapping x (scalar or array) to Z(x) as float64.
    """
    key = str(name).strip().lower()
    if key not in _PROFILE_BUILDERS:
        raise ValueError(
            f"Unknown bed profile '{name}'. Available: "
            + ", ".join(available_bed_profiles())
        )
    try:
        return _PROFILE_BUILDERS[key](**params)
    except TypeError as exc:
        raise TypeError(
            f"Bad parameters for bed profile '{key}': {exc}"
        ) from exc


@dataclass(frozen=True)
class TopographySettings:
    """What the PDE object needs to know about the bed.

    Only used to build topography-aware *initial conditions*: the bed's effect
    on the *dynamics* enters through the augmented system matrix and the
    mesh-resident sampled elevation, not through this object.

    Attributes
    ----------
    bed_elevation : callable or None
        Z(x). None means a flat bed at zero.
    reference_water_level : float
        The free-surface level H of a lake at rest, so h(x) = H - Z(x).
    perturbation_amplitude, perturbation_center, perturbation_width : float
        Gaussian free-surface perturbation added by the
        `perturbedLakeAtRest` initial condition. An amplitude of 0 makes it
        identical to `lakeAtRest`.
    lake_left, lake_right : float
        Extent of the water body of the `lakeWithDryShores` initial
        condition: h(x) = max(H - Z(x), 0) for lake_left <= x <= lake_right
        and 0 outside. Unbounded by default. Needed when ground beyond a
        divide lies below the lake level, as the valley below a dam does.
    """

    bed_elevation: BedElevation | None = None
    reference_water_level: float = 1.0
    perturbation_amplitude: float = 0.0
    perturbation_center: float = 0.0
    perturbation_width: float = 1.0
    lake_left: float = -np.inf
    lake_right: float = np.inf

    def __post_init__(self):
        if not self.lake_left < self.lake_right:
            raise ValueError(
                f"lake_left ({self.lake_left}) must be < lake_right "
                f"({self.lake_right})."
            )

    def elevation_at(self, position) -> float:
        """Z(position) as a plain float; 0 for the default flat bed."""
        if self.bed_elevation is None:
            return 0.0
        return float(np.asarray(self.bed_elevation(position), dtype=np.float64))
