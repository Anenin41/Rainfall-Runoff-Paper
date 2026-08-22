"""The recharge initial conditions, after they were made generic in N.

Three things are pinned here, and the reason for each is the same reason the
change was risky in the first place.

**The N <= 2 values must not move, at all.** These initial conditions seed every
validated thesis configuration in `swme/config/`. Generalizing the moment ladder
into a loop is only safe if it is a refactor at the orders that already existed,
so the check is `array_equal`, not `allclose`: the legacy N=0/1/2 expressions are
transcribed below and must be reproduced bit for bit. (They are: the ladder was
`alpha_2 = -0.5*alpha_1`, and halving is exact in binary floating point, so
`0.10 * -0.5` *is* `-0.05`.)

**The profiles must stay nested.** The whole point of these cases is comparing
moment orders against each other under identical forcing. If raising N perturbed
h, u_m, or any lower moment, an N-to-N comparison would silently become a
comparison of two different problems.

**The seeded ray must stay outside the non-hyperbolic wedge.** Per
RESTRUCTURE_PLAN.md Section 6, SWME from N=2 up loses hyperbolicity on a narrow
wedge of moment *ratios*, |alpha_2/alpha_1| in [1.14, 1.40] - not on a magnitude
threshold. So this suite checks the ratio the initial conditions actually use, at
magnitudes far past anything a run reaches, and includes a wedge state that must
fail. Without that second half the hyperbolicity test could not fail and would be
worth nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from recharge.initial_conditions import MOMENT_DECAY_RATIO, RechargeSWME1D_CustomIC
from swme.pde import SWME1D
from swme.spatialDiscretization import SpatialDiscretization

# The documented N=2 unstable wedge, RESTRUCTURE_PLAN.md Section 6.
UNSTABLE_WEDGE = (1.14, 1.40)

# The suite's own bar for "on the real axis", from
# `SpatialDiscretization.hyperbolicity_tolerance`. Exact zero is the wrong test at the tail of the
# Gaussian, where the moments vanish and the spectrum becomes degenerate: LAPACK
# splits a repeated real eigenvalue into a conjugate pair off the axis by ~1e-17.
# That is the eigensolver, not the model, and asserting == 0.0 would pin
# roundoff. The wedge state below clears this bar by fifteen orders of
# magnitude, so nothing real hides under it.
HYPERBOLICITY_TOLERANCE = SpatialDiscretization.hyperbolicity_tolerance

CASES = {
    # name -> (h_ref, h_amp, u_ref, u_amp, alpha_amp)
    "smooth_nested_profile_pulse_aggressive": (1.0, 0.05, 1.0, 0.15, 0.10),
    "smooth_nested_profile_pulse_mild": (1.0, 0.03, 0.8, 0.08, 0.04),
    "horton_moment_order_pulse": (1.0, 0.05, 0.5, 0.10, 0.10),
}

POSITIONS = np.linspace(0.0, 1.0, 51)


class _NoInfiltration:
    def rate(self, t, rainfall, h, dt):
        return 0.0


class _NoMixingFriction:
    def evaluate(self, R, I, context, values):
        return 0.0, 0.0


@pytest.fixture(scope="module")
def pde():
    return RechargeSWME1D_CustomIC(
        initial_condition="horton_moment_order_pulse",
        viscosity=1e-3,
        slip_length=1.0,
        hyperbolic=False,
        linear_source=False,
        rainfall_rate=0.0,
        infiltration_model=_NoInfiltration(),
        mixing_friction_model=_NoMixingFriction(),
    )


def _legacy(name, order, position):
    """The pre-generalization implementation, transcribed verbatim."""
    h_ref, h_amp, u_ref, u_amp, alpha_amp = CASES[name]
    values = np.zeros(order + 2, dtype=np.float64)
    pulse = np.exp(-((position - 0.5) / 0.08) ** 2)
    h = h_ref + h_amp * pulse
    values[0] = h
    values[1] = h * (u_ref + u_amp * pulse)
    if order > 0:
        values[2] = h * (alpha_amp * pulse)
    if order > 1:
        values[3] = h * (-0.5 * alpha_amp * pulse)
    return values


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("order", [0, 1, 2])
def test_legacy_orders_are_bit_for_bit_unchanged(pde, name, order):
    """The validated thesis runs must be untouched by the generalization."""
    for position in POSITIONS:
        assert np.array_equal(
            pde.get_initial_values(order, name, float(position)),
            _legacy(name, order, float(position)),
        )


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("order", [3, 4, 5, 6, 8])
def test_orders_beyond_the_legacy_cap_are_available(pde, name, order):
    """`horton_moment_order_pulse` used to raise NotImplementedError above N=2,
    and the other two silently returned zeros for every moment past alpha_2."""
    values = pde.get_initial_values(order, name, 0.5)

    assert values.shape == (order + 2,)
    assert np.all(np.isfinite(values))
    # Every moment is seeded, not left at zero - the old silent-zero behavior
    # would make a high-N run indistinguishable from a low-N one.
    assert np.all(values[2:] != 0.0)


@pytest.mark.parametrize("name", sorted(CASES))
def test_profiles_are_nested_across_moment_order(pde, name):
    """Raising N adds a moment and changes nothing below it, which is what makes
    the cross-order comparison a comparison of models rather than of problems."""
    for position in (0.0, 0.37, 0.5, 0.83):
        for order in range(9):
            high = pde.get_initial_values(order, name, position)
            for lower in range(order + 1):
                low = pde.get_initial_values(lower, name, position)
                assert np.array_equal(high[: lower + 2], low)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("order", [1, 2, 4, 8])
def test_moments_decay_geometrically_on_the_documented_ray(pde, name, order):
    alpha_amp = CASES[name][4]
    values = pde.get_initial_values(order, name, 0.5)
    h = values[0]
    pulse = 1.0  # position 0.5 is the pulse center

    expected = [
        alpha_amp * MOMENT_DECAY_RATIO ** (i - 1) * pulse for i in range(1, order + 1)
    ]
    assert np.allclose(values[2:] / h, expected, rtol=0, atol=1e-15)

    # Higher moments enter at strictly smaller amplitude, so raising N perturbs
    # the initial velocity profile less and less.
    magnitudes = np.abs(values[2:])
    assert np.all(np.diff(magnitudes) < 0.0)


def test_the_seeded_ratio_is_clear_of_the_unstable_wedge():
    assert not (UNSTABLE_WEDGE[0] <= abs(MOMENT_DECAY_RATIO) <= UNSTABLE_WEDGE[1])


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("hyperbolic", [False, True])
def test_seeded_states_are_hyperbolic_at_every_order(order, hyperbolic):
    """Checked against the SWME closure too (`hyperbolic=False`), which is the
    fragile one - HSWME passing would prove nothing, it never loses it."""
    model = SWME1D("unused", 1e-3, 1.0, hyperbolic, False)
    ic = RechargeSWME1D_CustomIC(
        initial_condition="horton_moment_order_pulse",
        viscosity=1e-3,
        slip_length=1.0,
        hyperbolic=hyperbolic,
        linear_source=False,
        rainfall_rate=0.0,
        infiltration_model=_NoInfiltration(),
        mixing_friction_model=_NoMixingFriction(),
    )
    for name in sorted(CASES):
        for position in POSITIONS:
            values = ic.get_initial_values(order, name, float(position))
            eigenvalues = np.linalg.eigvals(
                model.compute_system_matrix(order, values)
            )
            assert np.max(np.abs(eigenvalues.imag)) <= HYPERBOLICITY_TOLERANCE


@pytest.mark.parametrize("order", [2, 3, 4, 6, 8])
def test_the_ray_stays_hyperbolic_far_past_any_reachable_magnitude(order):
    """The ray property, not a smallness accident: the largest scaled moment any
    thesis run reached is 0.71, and this sweeps to 200."""
    model = SWME1D("unused", 1e-3, 1.0, False, False)
    for alpha_1 in (0.5, 5.0, 50.0, 200.0):
        values = np.zeros(order + 2)
        values[0] = 1.0
        values[1] = 1.0
        for i in range(1, order + 1):
            values[i + 1] = alpha_1 * MOMENT_DECAY_RATIO ** (i - 1)
        eigenvalues = np.linalg.eigvals(model.compute_system_matrix(order, values))
        assert np.max(np.abs(eigenvalues.imag)) <= HYPERBOLICITY_TOLERANCE


@pytest.mark.parametrize("ratio", [1.2, -1.2, 1.3])
def test_the_hyperbolicity_check_can_actually_fail(ratio):
    """A ratio inside the documented wedge must break N=2 SWME. Without this the
    tests above would pass against a broken eigen-check and mean nothing."""
    model = SWME1D("unused", 1e-3, 1.0, False, False)
    values = np.array([1.0, 1.0, 1.5, 1.5 * ratio])
    eigenvalues = np.linalg.eigvals(model.compute_system_matrix(2, values))
    assert np.max(np.abs(eigenvalues.imag)) > 1e-3


def test_a_negative_moment_order_is_rejected(pde):
    with pytest.raises(ValueError, match="non-negative"):
        pde.get_initial_values(-1, "horton_moment_order_pulse", 0.5)
