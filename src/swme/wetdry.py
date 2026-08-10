"""Wet-dry treatment: thresholds and desingularized primitive reconstruction.

The conserved state is (h, h*u_m, h*a_1, ..., h*a_N), so every closure in the
model needs primitives obtained by dividing by h. That division is the whole
difficulty of a drying front: h -> 0 makes u_m and the moments blow up long
before h actually reaches zero, and the friction closure divides by h again
(nu/h^2 terms), so it blows up faster still.

This module owns the single rule for extracting primitives, so the system
matrix, the wave speeds, the friction and the recharge sources all agree about
what a nearly-dry cell means. It is applied ONLY when reading primitives out of
a state - never to the conserved state array itself, which stays exactly what
the finite-volume update produced.

Three regimes, separated by two thresholds (RESTRUCTURE_PLAN.md §2.3):

* h >= h_wet   fully wet. Primitives are the plain q/h, evaluated by exactly
               the arithmetic used before wet-dry existed, so runs that never
               approach drying are bit-for-bit unchanged.
* h_dry <= h < h_wet   transition. Velocity is still exactly q/h (see below);
               the moments are ramped linearly to zero, so a vanishing film
               relaxes to plug flow instead of carrying a vertical profile it
               cannot support.
* h < h_dry    draining to dry. Kurganov-Petrova desingularization takes over
               and drives the velocity smoothly to zero; the moments are
               already zero from the ramp.

Deviation from the plan, deliberate. §2.3 wrote the desingularization with
`max(h, eps_div)` and separately demanded `u_m = 0 exactly` below `h_dry`.
Those two do not fit together: with eps_div ~ 1e-14 the desingularization never
activates above machine noise, so the only thing shaping the velocity would be
the hard cut at h_dry - which is a jump of size q/h_dry right at the wetting
front, the worst possible place to put a discontinuity. Using the *physical*
threshold in the Kurganov-Petrova denominator instead,

    u_m = 2*h*q / (h^2 + max(h, h_dry)^2)

gives the intended behaviour continuously and with no special case: it is
identically q/h for h >= h_dry (the denominator is exactly 2*h^2 there) and
decays quadratically to zero below it. `eps_div` keeps its original job as the
last-ditch guard against 0/0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WetDryThresholds:
    """The two-tier epsilon convention of RESTRUCTURE_PLAN.md §2.3(a).

    Attributes
    ----------
    eps_div : float
        Machine-precision division guard. Not a physical quantity - purely
        the value below which a cell is treated as exactly, numerically empty
        so that no division is attempted at all.
    h_dry : float
        Physical dry threshold. At and below it a cell carries no moments and
        its velocity has been driven to (essentially) zero.
    h_wet : float
        Physical wet threshold. At and above it the cell is treated as
        ordinary wet flow with no regularization whatsoever.

    The defaults suit the thesis' test cases, which run at h ~ O(1). They are
    absolute depths, not ratios, so scale them to the problem: a simulation
    with h ~ 1e-3 everywhere would be entirely "dry" under these defaults.

    Choosing h_dry is a real trade-off, not a formality, and it is worth
    measuring rather than guessing:

    * Too large and it truncates genuinely small depths. A vacuum front is the
      sharp case, because the exact solution itself goes to zero there:
      Ritter's h vanishes quadratically at the front, so the leading edge sits
      below any fixed h_dry and gets slowed down. Measured on the standard dry
      dam break at 800 cells, against an exact front speed of 2.0 -
      h_dry = 1e-4 gives 1.758, 1e-8 gives 1.934, 1e-12 gives 1.984. The error
      does not converge away under mesh refinement, because refining only
      resolves more of the truncated region.
    * Too small and the friction terms it floors stop being bounded. `nu/h^2`
      at h_dry = 1e-10 reaches nu*1e20, which is finite - so no isfinite check
      catches it - and meaningless.

    So: put h_dry comfortably below the smallest depth the problem needs to
    resolve, then check that nu/h_dry^2 is still a sane number. An inviscid
    problem has no lower limit and can afford a very small h_dry; a viscous
    one cannot.
    """

    eps_div: float = 1e-14
    h_dry: float = 1e-4
    h_wet: float = 1e-3

    def __post_init__(self):
        if not (self.eps_div > 0.0):
            raise ValueError(f"eps_div must be > 0, got {self.eps_div}.")
        if not (self.h_dry >= self.eps_div):
            raise ValueError(
                f"h_dry ({self.h_dry}) must be >= eps_div ({self.eps_div}); "
                "the physical dry threshold cannot sit below the numerical one."
            )
        if not (self.h_wet > self.h_dry):
            raise ValueError(
                f"h_wet ({self.h_wet}) must be > h_dry ({self.h_dry}); the "
                "transition band needs positive width for the moment ramp."
            )

    def is_dry(self, h) -> bool:
        return h <= self.h_dry

    def moment_ramp(self, h: float) -> float:
        """Linear ramp, 0 at h_dry and 1 at h_wet, clipped outside."""
        if h >= self.h_wet:
            return 1.0
        if h <= self.h_dry:
            return 0.0
        return (h - self.h_dry) / (self.h_wet - self.h_dry)


DEFAULT_THRESHOLDS = WetDryThresholds()


def desingularized_primitives(order: int,
                              values: np.ndarray,
                              thresholds: WetDryThresholds = DEFAULT_THRESHOLDS):
    """Extract (h, u_m, alpha) from a conserved state, dry-safely.

    Parameters
    ----------
    order : int
        Moment order N; `alpha` comes back with N entries.
    values : np.ndarray
        Conserved state [h, h*u_m, h*a_1, ..., h*a_N].
    thresholds : WetDryThresholds

    Returns
    -------
    (h, u_m, alpha) : (float, float, np.ndarray)
        `h` is the raw conserved height, unmodified - callers that need a
        regularized height for their own division must ask for it explicitly
        (see `safe_height`). `u_m` and `alpha` are the regularized primitives.

    Notes
    -----
    For h >= h_wet this returns exactly `values[1]/h` and `values[2:]/h`, the
    same expressions and the same order of operations as before the wet-dry
    work, so wet runs reproduce previous results bit for bit.
    """
    h = float(values[0])

    if h <= thresholds.eps_div:
        # Numerically empty: no division at all, and no momentum or profile.
        return h, 0.0, np.zeros(order, dtype=np.float64)

    if h >= thresholds.h_wet:
        # Ordinary wet flow: untouched arithmetic.
        u_m = values[1] / h
        alpha = values[2:] / h if order else np.zeros(0, dtype=np.float64)
        return h, u_m, alpha

    # Transition band and below. `scale` is identically 1/h for h >= h_dry and
    # decays quadratically to 0 beneath it.
    floor = h if h > thresholds.h_dry else thresholds.h_dry
    scale = 2.0 * h / (h * h + floor * floor)

    u_m = values[1] * scale
    if order == 0:
        return h, u_m, np.zeros(0, dtype=np.float64)

    ramp = thresholds.moment_ramp(h)
    alpha = values[2:] * (scale * ramp)
    return h, u_m, alpha


def safe_height(h: float,
                thresholds: WetDryThresholds = DEFAULT_THRESHOLDS) -> float:
    """A height safe to divide by, for closures that need 1/h or 1/h^2.

    Floors at `h_dry` rather than `eps_div`: the viscous friction terms carry
    nu/h^2, which at h = eps_div = 1e-14 would be 1e28 - finite, and therefore
    not caught by any isfinite check, but complete nonsense. `h_dry` is the
    depth below which the cell is declared to have no meaningful internal
    structure, so it is also the right floor for the terms that describe that
    structure.
    """
    return h if h > thresholds.h_dry else thresholds.h_dry


def desingularized_primitives_array(order: int,
                                    values: np.ndarray,
                                    thresholds: WetDryThresholds = DEFAULT_THRESHOLDS):
    """Vectorized `desingularized_primitives` over a (M, order+2) state array.

    Used by `compute_max_wavespeed` and `convert_to_primitive`, which see the
    whole grid at once. Same three regimes, same values.
    """
    values = np.asarray(values, dtype=np.float64)
    h = values[:, 0]

    wet = h >= thresholds.h_wet
    if wet.all():
        # Same exact expressions as the scalar wet branch, so the two agree
        # bit for bit and neither differs from the pre-wet-dry code.
        u_m = values[:, 1] / h
        alpha = (values[:, 2:] / h[:, None] if order
                 else np.zeros((len(h), 0), dtype=np.float64))
        return h, u_m, alpha

    empty = h <= thresholds.eps_div
    floor = np.where(h > thresholds.h_dry, h, thresholds.h_dry)
    denominator = h * h + floor * floor
    scale = np.where(empty, 0.0,
                     2.0 * h / np.where(empty, 1.0, denominator))
    # Exact 1/h wherever the cell is fully wet, matching the fast path above.
    scale = np.where(wet, np.divide(1.0, np.where(empty, 1.0, h)), scale)

    u_m = values[:, 1] * scale

    if order == 0:
        return h, u_m, np.zeros((len(h), 0), dtype=np.float64)

    ramp = np.clip(
        (h - thresholds.h_dry) / (thresholds.h_wet - thresholds.h_dry), 0.0, 1.0)
    alpha = values[:, 2:] * (scale * ramp)[:, None]
    return h, u_m, alpha
