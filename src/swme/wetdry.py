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

Three regimes, separated by two thresholds (RESTRUCTURE_PLAN.md section 3):

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

Deviation from the plan, deliberate. The plan wrote the desingularization with
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

from . import coefficients


@dataclass(frozen=True)
class WetDryThresholds:
    """The two-tier epsilon convention of RESTRUCTURE_PLAN.md section 3.1.

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
    shoreline_treatment : bool
        Whether wet-dry interfaces across a bed step get the modified Roe
        treatment of Castro, Gonzalez-Vida and Pares (2006),
        `shoreline_fluctuation`, instead of the ordinary path-conservative
        fluctuation. On by default. It only ever acts over non-flat
        topography, at an interface with a dry cell and a bed step, so runs
        that have no such interface are unaffected bit for bit. Switching it
        off restores the untreated scheme, which produces negative depths
        there.

    The defaults suit the thesis' test cases, which run at h ~ O(1). They are
    absolute depths, not ratios, so scale them to the problem: a simulation
    with h ~ 1e-3 everywhere would be entirely "dry" under these defaults.

    Choosing h_dry is a real trade-off, not a formality, and it is worth
    measuring rather than guessing:

    * Too large and it truncates genuinely small depths. A vacuum front is the
        sharp case, because the exact solution itself goes to zero there:
        Ritter's h vanishes quadratically at the front, so the leading edge
        sits below any fixed h_dry and gets slowed down. Measured on the
        standard dry dam break at 800 cells, against an exact front speed of
        2.0 - h_dry = 1e-4 gives 1.758, 1e-8 gives 1.934, 1e-12 gives 1.984.
        The error does not converge away under mesh refinement, because
        refining only resolves more of the truncated region.
    * Too small and the friction terms it floors stop being bounded. `nu/h^2`
        at h_dry = 1e-10 reaches nu*1e20, which is finite - so no isfinite
        check catches it - and meaningless.

    So: put h_dry comfortably below the smallest depth the problem needs to
    resolve, then check that nu/h_dry^2 is still a sane number. An inviscid
    problem has no lower limit and can afford a very small h_dry; a viscous
    one cannot.
    """

    eps_div: float = 1e-14
    h_dry: float = 1e-4
    h_wet: float = 1e-3
    shoreline_treatment: bool = True

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

    Used by `compute_max_wavespeed`, which sees the whole grid at once. Same
    three regimes, same values.
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




# ---------------------------------------------------------------------------
# Shoreline treatment: Castro, Gonzalez-Vida and Pares (2006), Section 5
# ---------------------------------------------------------------------------
#
# Notation. Castro et al. write the bed as a depth H below a reference level;
# here it is the elevation Z = -H. Their Case 2 (H_L < H_R, wet cell L, dry
# cell R) is a bed that drops towards the dry cell; their Case 3
# (H_L > H_R) is a bed that rises towards it. All velocities below are the
# depth-averaged velocity u_m, taken positive towards the neighbouring cell.

def drains_a_dry_cell(h_left: float, h_right: float, mass_flux: float,
                      delta_t: float, delta_x: float,
                      thresholds: WetDryThresholds = DEFAULT_THRESHOLDS) -> bool:
    """Whether an interface mass flux takes too much water out of a dry cell.

    In flux form the mass update of cell i is
    `h_i <- h_i - dt/dx * (F_{i+1/2} - F_{i-1/2})`, where the mass flux of an
    interface follows from its fluctuations as
    `F = q_left + D-[0] = q_right - D+[0]`. A positive F leaves the left
    cell, a negative F leaves the right cell. The positivity-preserving
    timestep limiter skips cells with `h <= h_dry`, so for those cells this
    is the only guard: the flux is rejected when this one interface would
    remove more than half of what such a cell holds (half, because a cell
    has two interfaces). For an exactly dry cell any loss at all is rejected.

    This is the second trigger of the shoreline treatment. It catches what
    the case analysis of `one_sided_problem` hands back to the ordinary
    scheme (Castro's Case 3.3, a submerged step): water receding from
    a dry cell whose bed is only just submerged, where the linearised
    Riemann solution has a negative depth on the dry side (N = 0, Roe, wet
    cell L below a dry cell with bed step d: the dry cell is drained when
    `d > h_L * (1 + Fr_L)`, which for a receding flow, Fr_L < 0, happens
    before the step emerges), and a nearly dry cell that still stores a
    residue of momentum.
    """
    transported = mass_flux * delta_t / delta_x
    if h_left <= thresholds.h_dry and transported > 0.5 * h_left:
        return True
    if h_right <= thresholds.h_dry and -transported > 0.5 * h_right:
        return True
    return False


def wall_depth_change(h: float, u_normal: float, g: float = 1.0) -> float:
    """Depth at a solid wall facing a shallow-water state (h, u), minus h.

    The exact solution of the partial Riemann problem with q = 0 at the wall
    (Castro, Gonzalez-Vida, Pares 2006, problem (5.21)). `u_normal` is the
    velocity component towards the wall, c = sqrt(g h), and h_w the depth at
    the wall:

    * `u_normal <= -2c`: the water leaves faster than it can expand and a
      dry zone opens at the wall, h_w = 0 (eq. 5.23);
    * `-2c < u_normal <= 0`: a rarefaction, `h_w = (u_normal/2 + c)^2 / g`
      (eq. 5.25), from the Riemann invariant u + 2c;
    * `u_normal > 0`: a reflected shock, h_w the greatest root of
      `x^3 - h x^2 - (h^2 + 2 h u^2 / g) x + h^3` (eq. 5.27), which is the
      Rankine-Hugoniot condition between (h, u) and (x, 0).

    The difference d = h_w - h is returned rather than h_w, because the wall
    pressure needs `h_w^2 - h^2 = d (2h + d)` and at small velocities that
    subtraction would lose every digit. For the same reason the cubic is not
    handed to a root finder: at u = 0 it has the double root x = h. Written
    in d it reads

        f(d) = d^2 (2h + d) - (2 h u^2 / g) (h + d) = 0,

    which has a single positive root and is convex for d > 0, so Newton's
    iteration from a point with f > 0 converges monotonically.
    """
    if h <= 0.0:
        return 0.0
    c = np.sqrt(g * h)
    if u_normal <= -2.0 * c:
        return -h
    if u_normal <= 0.0:
        return u_normal * (0.25 * u_normal + c) / g

    k = 2.0 * h * u_normal * u_normal / g
    froude = u_normal / c
    d = h * (froude + froude * froude)       # f(d) > 0 here
    for _ in range(60):
        f = d * d * (2.0 * h + d) - k * (h + d)
        slope = d * (4.0 * h + 3.0 * d) - k
        step = f / slope
        d -= step
        if abs(step) <= 1e-15 * d:
            break
    return d


def regularised_state(values: np.ndarray,
                      thresholds: WetDryThresholds = DEFAULT_THRESHOLDS
                      ) -> np.ndarray:
    """The conserved state rebuilt from the desingularised primitives,
    `(h, h*u_m, h*a_1, ..., h*a_N)`.

    At or above `h_wet` this is the state itself, returned unchanged. Below
    it, the momentum and the moments are the ones the system matrix actually
    sees (`desingularized_primitives`), and an empty cell (`h <= eps_div`)
    carries none at all. A cell can hold a residue of momentum without the
    water to carry it, left behind when it drained; the shoreline treatment
    must not read that residue as a discharge. `run_simulation` also applies
    this to every cell at or below `h_dry` after each step when the
    treatment is on, which is the usual consistency step of the
    Kurganov-Petrova desingularisation (recompute h*u from the desingularised
    u): below `h_dry` it scales the momentum by `2 h^2 / (h^2 + h_dry^2)`,
    continuously from 1 at `h_dry` to 0 for an empty cell, and removes the
    moments.
    """
    values = np.asarray(values, dtype=np.float64)
    if values[0] >= thresholds.h_wet:
        return values
    order = values.shape[0] - 2
    h, u_m, alpha = desingularized_primitives(order, values, thresholds)
    state = np.empty_like(values)
    state[0] = h
    state[1] = h * u_m
    state[2:] = h * alpha
    return state



def momentum_flux(state: np.ndarray, g: float = 1.0,
                  hyperbolic: bool = False) -> float:
    """Momentum flux of the SWME/HSWME momentum equation,

        F_mom(U) = q^2/h + g h^2/2 + sum_i (h a_i)^2 / ((2i+1) h).

    The momentum row of the system matrix A(U) is exactly the gradient of
    this function, so the momentum equation is in conservation form apart
    from the bed term. HSWME (`hyperbolic`) assembles A(U) with a_2..a_N set
    to zero, so its momentum flux has the a_1 term only. An empty state has
    no flux.
    """
    h = state[0]
    if h <= 0.0:
        return 0.0
    flux = state[1] * state[1] / h + 0.5 * g * h * h
    order = state.shape[0] - 2
    if order:
        moments = state[2:3] if hyperbolic else state[2:]
        weights = coefficients.get_coefficients(order).inv_two_i_plus_1
        flux += (moments * moments) @ weights[:moments.shape[0]] / h
    return flux


def vacuum_edge_state(h: float, u_normal: float, g: float = 1.0):
    """Depth and velocity at x = 0 of the rarefaction into a dry bed.

    Castro et al. eq. (5.8): the Riemann problem between the wet state
    (h, u) on the left and the vacuum on the right, on a flat bed, sampled
    on the interface. With c = sqrt(g h):

    * `u <= -2c`: the water leaves faster than it expands, (0, 0);
    * `-2c < u < c`: the critical state of the rarefaction,
      u0 = c0 = (u + 2c)/3, h0 = u0^2/g, from the Riemann invariant u + 2c;
    * `u >= c`: the flow is supercritical towards the dry bed and the
      interface sees the cell state itself.

    Returns (h0, u0), with u0 positive towards the dry bed.
    """
    if h <= 0.0:
        return 0.0, 0.0
    c = np.sqrt(g * h)
    if u_normal <= -2.0 * c:
        return 0.0, 0.0
    if u_normal >= c:
        return h, u_normal
    u0 = (u_normal + 2.0 * c) / 3.0
    return u0 * u0 / g, u0


def bernoulli_depth(discharge: float, energy: float, g: float = 1.0,
                    supercritical: bool = True) -> float:
    """Depth with discharge q and specific energy E, h + q^2/(2 g h^2) = E.

    This is the steady contact discontinuity at a bed step, Castro et al.
    eq. (2.30) with q unchanged: the polynomials (5.15) and (5.18) are this
    relation multiplied by h^2. It has two positive roots when
    E > (3/2) h_c, h_c = (q^2/g)^(1/3) the critical depth: the
    supercritical one (h < h_c, the least positive root of (5.15)) and the
    subcritical one (h > h_c, the greatest). Below (3/2) h_c there is none,
    and the critical depth is returned, which is where both roots meet.

    f(h) = h + q^2/(2 g h^2) - E is convex, decreasing below h_c and
    increasing above it. Newton's iteration started at q/sqrt(2 g E)
    (f > 0, left of the supercritical root) or at E (f > 0, right of the
    subcritical root) therefore converges monotonically.
    """
    q2 = discharge * discharge
    if q2 == 0.0:
        return 0.0 if supercritical else max(energy, 0.0)
    critical = (q2 / g) ** (1.0 / 3.0)
    if energy <= 1.5 * critical:
        return critical
    h = np.sqrt(q2 / (2.0 * g * energy)) if supercritical else energy
    for _ in range(200):
        f = h + q2 / (2.0 * g * h * h) - energy
        slope = 1.0 - q2 / (g * h * h * h)
        step = f / slope
        h -= step
        if abs(step) <= 1e-15 * h:
            break
    return h


def wall_state(state: np.ndarray, direction: int, g: float = 1.0) -> np.ndarray:
    """Castro's state at a solid wall in `direction` from a cell.

    `(h_w, 0, h_w a_1, ..., h_w a_N)` with `h_w = h + wall_depth_change(h,
    u)`, u the depth-averaged velocity towards the wall: no discharge, the
    exact shallow-water wall depth of eqs (5.23)-(5.27), and the moments per
    unit depth of the cell unchanged (a modelling choice for N >= 1, see
    `one_sided_problem`). An empty cell, or a wall the water runs away from
    faster than it can expand (h_w = 0), gives the empty state.
    """
    result = np.zeros_like(state, dtype=np.float64)
    h = state[0]
    if h <= 0.0:
        return result
    depth = h + wall_depth_change(h, direction * state[1] / h, g)
    if depth > 0.0:
        result[0] = depth
        result[2:] = state[2:] * (depth / h)
    return result


def one_sided_problem(state: np.ndarray, direction: int, step: float,
                      g: float = 1.0,
                      thresholds: WetDryThresholds = DEFAULT_THRESHOLDS):
    """Castro's nonlinear Riemann problem for one cell facing an empty one.

    `state` is the (regularised) conserved state of a cell, the neighbour
    in `direction` (+1: to the right, -1: to the left) is taken to be
    empty, and `step = Z_neighbour - Z_cell` is the bed step between them.
    Returns `(kind, near, far)`: the states on the cell's own side and on
    the neighbour's side of the interface, each on its own bed, with the
    discharge signed in x. By Castro et al. eq. (5.1) the numerical fluxes
    are the physical fluxes of these two states. With u the depth-averaged
    velocity towards the empty cell, c = sqrt(g h), the specific energy
    E = h + u^2/(2g) and the critical depth h_c = (q^2/g)^(1/3):

    Bed rising towards the empty cell (Castro Case 3, `step > 0`):

    * `climb` (Case 3.2): supercritical flow towards the step, `u >= c`,
      with the mechanical energy to climb it, `step <= E - (3/2) h_c` (the
      negation of eq. 5.20). The near state is the cell itself, the far
      state the supercritical state on the step with the same discharge
      (eq. 5.29, least positive root of eq. 5.15, `bernoulli_depth`).
    * `wall` (Case 3.1): the bed emerges, `step >= h`, and either the flow
      does not move towards it (`u <= 0`) or it does without the energy to
      climb it (eq. 5.20). The near state is `wall_state` (eqs 5.23-5.27),
      the far state is empty.
    * `ordinary` (Case 3.3, every remaining case): Castro et al. keep the
      unmodified Roe scheme. No states are returned.

    Reading of Case 3.1, deliberate. The paper's second condition for 3.1
    ("u_L >= 0 and the mechanical energy at the wet cell is not enough to
    make the fluid go up the step", eq. 5.20) does not repeat the
    emergence condition of the first, but its Fig. 8(b) draws the bed
    emerged, and the case is introduced as "the step acts as an obstacle".
    Read literally, it also walls off a dry cell whose bed lies below the
    free surface whenever the discharge is too large for a steady flow over
    the step, although the water standing above that bed would spread onto
    it. Measured on Thacker's sloshing (60, 120, 240 cells) the literal
    reading raises the mean depth error from 0.0118, 0.0058, 0.0037 to
    0.0278, 0.0111, 0.0088. This code requires the bed to emerge.

    Bed dropping towards the empty cell (Case 2, `step < 0`), `overfall`:
    the near state is the vacuum-rarefaction state W0 of eq. (5.8)
    (`vacuum_edge_state`), the far state the state below the step on the
    same steady contact, with the discharge of W0 and the energy of W0
    plus the drop (eqs 5.11-5.14). The supercritical root is used in every
    case. Castro et al. take it when the cell is supercritical (their case
    (b), eq. 5.14), but the subcritical root when W0 is critical (their
    case (c), eqs 5.17-5.18, for `-2c < u < c`). This code does not follow
    them there, for two reasons. The subcritical state has a negative
    characteristic speed u - c, so the rarefaction that has to connect it
    to the dry bed on its right would run back through the stationary
    step: the wave structure the state is supposed to belong to does not
    exist (their Remark 5.1 notes the resonance). And its depth grows like
    the drop while its discharge stays that of W0, so the empty cell would
    receive a momentum flux of order g drop^2 / 2 with almost no water.
    The supercritical state is the free overfall: the water accelerates
    down the step. `vacuum` is the case `u <= -2c`, where no water reaches
    the interface (eqs 5.8, 5.10).

    For N >= 1 the moments per unit depth, a_i, are carried unchanged into
    every interface state, and every depth is the shallow-water one,
    computed from the depth-averaged velocity. The paper treats N = 0
    only; these are two modelling choices.

    An empty cell (`h <= eps_div`) gives `('empty', None, None)`.
    """
    h = state[0]
    if h <= thresholds.eps_div:
        return 'empty', None, None
    if step == 0.0:
        raise ValueError("one_sided_problem needs a bed step; a flat "
                         "interface keeps the ordinary scheme.")
    discharge = direction * state[1]
    u = discharge / h
    c = np.sqrt(g * h)
    alpha = state[2:] / h

    def at_interface(depth, normal_discharge):
        result = np.zeros_like(state, dtype=np.float64)
        if depth > 0.0:
            result[0] = depth
            result[1] = direction * normal_discharge
            result[2:] = depth * alpha
        return result

    if step > 0.0:
        enough_energy = False
        if u > 0.0:
            energy = h + 0.5 * u * u / g
            critical = (discharge * discharge / g) ** (1.0 / 3.0)
            enough_energy = step <= energy - 1.5 * critical
            if enough_energy and u >= c:
                depth = bernoulli_depth(discharge, energy - step, g, True)
                return ('climb', np.array(state, dtype=np.float64),
                        at_interface(depth, discharge))
        if step >= h and not enough_energy:
            return ('wall', wall_state(state, direction, g),
                    np.zeros_like(state, dtype=np.float64))
        return 'ordinary', None, None

    h0, u0 = vacuum_edge_state(h, u, g)
    if h0 == 0.0:
        empty = np.zeros_like(state, dtype=np.float64)
        return 'vacuum', empty, empty.copy()
    discharge0 = h0 * u0
    near = (np.array(state, dtype=np.float64) if u >= c
            else at_interface(h0, discharge0))
    energy = h0 + 0.5 * u0 * u0 / g - step
    depth = bernoulli_depth(discharge0, energy, g, True)
    return 'overfall', near, at_interface(depth, discharge0)


def _jump(scheme, system_matrix, left, right, delta_t, delta_x, g, hyperbolic):
    """F(right) - F(left) in the conservative rows, the path integral of A
    from left to right in the moment rows (which have no flux)."""
    jump = np.zeros_like(left, dtype=np.float64)
    if np.array_equal(left, right):
        return jump
    if left.shape[0] > 2:
        matrix_min, matrix_plus = scheme.compute_generalized_roe_and_viscosity(
            left, right, system_matrix, delta_t, delta_x)
        jump[2:] = ((matrix_min + matrix_plus) @ (right - left))[2:]
    jump[0] = right[1] - left[1]
    jump[1] = (momentum_flux(right, g, hyperbolic)
               - momentum_flux(left, g, hyperbolic))
    return jump


def shoreline_fluctuation(scheme, system_matrix, value_left: np.ndarray,
                          value_right: np.ndarray, bed_left: float,
                          bed_right: float, delta_t: float, delta_x: float,
                          thresholds: WetDryThresholds = DEFAULT_THRESHOLDS,
                          g: float = 1.0, hyperbolic: bool = False,
                          force_wall: bool = False):
    """Fluctuations at a wet-dry interface across a bed step, or None.

    The wet-dry treatment of Castro, Gonzalez-Vida and Pares (M3AS 16 (2006)
    897-931, Section 5) in fluctuation form, for the moment variables. It
    acts where at least one of the two cells is dry (`h <= h_dry`) and the
    bed is not flat (`bed_left != bed_right`); everywhere else it returns
    None and the caller keeps the ordinary path-conservative fluctuation.

    The Riemann problem across the step is split into two of Castro's
    nonlinear problems, one for the water of each cell with the other cell
    taken to be empty (`one_sided_problem`): the lower cell's water facing
    a rising bed (Case 3) and the higher cell's water facing a drop
    (Case 2). Where one cell is exactly empty, as in the paper, one of the
    two is trivial and the scheme is Castro's. Where the dry cell holds a
    film below `h_dry`, the two are superposed. With near and far states
    W_near, W_far of a problem posed for cell X, X receives the jump from
    its state to W_near and the other cell the jump from W_far to empty:

        left  cell X: D- += J(U_X -> W_near),  right cell: D+ += J(W_far -> 0)
        right cell X: D+ += J(W_near -> U_X),  left  cell: D- += J(0 -> W_far)

    J is the flux difference in the mass and momentum rows (eq. 5.1: the
    numerical flux is the flux of the interface state) and, for the moment
    rows, which have no flux, the scheme's path integral of A(U). The
    stationary contact at the step between W_near and W_far carries the bed
    force and gives no fluctuation, as in the generalised Rankine-Hugoniot
    condition (2.26) with zero speed.

    If the lower cell's problem is Castro's Case 3.3 (`ordinary`), the
    whole interface keeps the ordinary scheme (None is returned), unless
    `force_wall` is set: then the lower cell gets the wall. The caller sets
    it when the ordinary fluctuation would drain a dry cell
    (`drains_a_dry_cell`), the positivity failure the paper notes for
    Case 3.3.

    Both problems are posed on `regularised_state` of the two cells. The
    mass rows are written in flux form with the sum of the two problems'
    discharges as the single interface flux F, D-[0] = F - q_left and
    D+[0] = q_right - F with the stored discharges, so mass is conserved
    exactly whatever the two states are.

    Returns
    -------
    (fluctuation_min, fluctuation_plus) or None
    """
    if value_left[0] > thresholds.h_dry and value_right[0] > thresholds.h_dry:
        return None
    step = bed_right - bed_left
    if step == 0.0:
        return None

    left = regularised_state(value_left, thresholds)
    right = regularised_state(value_right, thresholds)
    # +1 if the higher cell is on the right
    up = 1 if step > 0.0 else -1
    lower, higher = (left, right) if up == 1 else (right, left)
    rise = abs(step)

    kind, near_lower, far_lower = one_sided_problem(lower, up, rise, g, thresholds)
    if kind == 'ordinary':
        if not force_wall:
            return None
        near_lower = wall_state(lower, up, g)
        far_lower = np.zeros_like(lower, dtype=np.float64)
    _, near_higher, far_higher = one_sided_problem(higher, -up, -rise, g,
                                                   thresholds)

    empty = np.zeros_like(left, dtype=np.float64)
    fluctuation_min = np.zeros_like(left, dtype=np.float64)
    fluctuation_plus = np.zeros_like(left, dtype=np.float64)
    mass_flux = 0.0
    for state, near, far, on_left in ((lower, near_lower, far_lower, up == 1),
                                      (higher, near_higher, far_higher, up == -1)):
        if near is None:
            continue
        if on_left:
            fluctuation_min += _jump(scheme, system_matrix, state, near,
                                     delta_t, delta_x, g, hyperbolic)
            fluctuation_plus += _jump(scheme, system_matrix, far, empty,
                                      delta_t, delta_x, g, hyperbolic)
        else:
            fluctuation_plus += _jump(scheme, system_matrix, near, state,
                                      delta_t, delta_x, g, hyperbolic)
            fluctuation_min += _jump(scheme, system_matrix, empty, far,
                                     delta_t, delta_x, g, hyperbolic)
        mass_flux += near[1]

    fluctuation_min[0] = mass_flux - value_left[1]
    fluctuation_plus[0] = value_right[1] - mass_flux
    return fluctuation_min, fluctuation_plus
