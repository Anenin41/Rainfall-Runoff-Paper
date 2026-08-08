"""Step 2 regression suite (RESTRUCTURE_PLAN.md), recharge side: proves the
new generic-N implementations in `recharge.source_terms`
(`compute_recharge_mass_source`, `compute_mixing_friction`,
`compute_total_friction`) reproduce the still-present hardcoded N=0/1/2
functions (`compute_recharge_source_n0/n1/n2`, `compute_friction_matrix_n0/n1/n2`,
dispatched via `compute_recharge_source`/`compute_friction_matrix`) for the only
orders the legacy code supports, before Step 3/4 delete the hardcoded functions
and dispatchers and wire the generic engine into RechargeSWME1D.

Unlike the swme-side system matrix (see test_pde_regression.py), no bugs were
found here on the orders the legacy code actually supports (N=0,1,2) - the
hand-derivation for N=1 was independently cross-checked by hand against the
general formula while designing swme.source_terms/recharge.source_terms (see
RESTRUCTURE_PLAN.md), so this suite is confirmatory rather than bug-hunting.
"""

from __future__ import annotations

import numpy as np
import pytest

from recharge import source_terms
from recharge.context import SourceContext

ORDERS = [0, 1, 2]  # the only orders the legacy N=0/1/2 code supports
N_SAMPLES = 8
SEED = 20260808


class _FixedInfiltration:
    """Stand-in infiltration model returning a fixed I regardless of inputs -
    isolates the math being tested from recharge/laws.py's (already generic,
    not order-dependent, out of Step 2's scope) infiltration closures."""

    def __init__(self, I_value):
        self.I_value = I_value

    def rate(self, t, rainfall, h, dt):
        return self.I_value


class _FixedMixingFriction:
    """Stand-in mixing-friction model returning fixed (f_R, f_I) regardless
    of inputs - isolates the math from recharge/laws.py's closures, same
    reasoning as _FixedInfiltration above."""

    def __init__(self, f_R, f_I):
        self.f_R = f_R
        self.f_I = f_I

    def evaluate(self, R, I, context, values):
        return self.f_R, self.f_I


def _random_state(order, rng, h_range=(0.4, 2.5), vel_scale=0.6):
    n = order + 2
    values = np.empty(n, dtype=np.float64)
    h = rng.uniform(*h_range)
    um = rng.uniform(-vel_scale, vel_scale)
    values[0] = h
    values[1] = h * um
    if order > 0:
        alpha = rng.uniform(-vel_scale, vel_scale, size=order)
        values[2:] = h * alpha
    return values


def _random_states(order, n_samples=N_SAMPLES, seed=SEED):
    rng = np.random.default_rng(seed + 500 + order)
    return [_random_state(order, rng) for _ in range(n_samples)]


PARAM_COMBOS = [
    # (R, I, f_R, f_I, viscosity, slip_length)
    (0.1, -0.05, 0.1, 0.05, 1e-3, 1.0),
    (0.0, 0.02, 0.0, 0.01, 5e-2, 0.3),
    (0.3, 0.3, 0.2, 0.0, 2.0, 4.0),
]


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("R,I,f_R,f_I,viscosity,slip_length", PARAM_COMBOS)
def test_recharge_mass_source_matches_legacy(order, R, I, f_R, f_I, viscosity, slip_length):
    ctx = SourceContext(time=0.0, dt=1e-3)
    for values in _random_states(order):
        S_old, _ = source_terms.compute_recharge_source(
            order=order,
            values=values,
            rainfall=R,
            infiltration_model=_FixedInfiltration(I),
            context=ctx,
        )
        S_new = source_terms.compute_recharge_mass_source(order, values, R=R, I=I)
        assert np.allclose(S_old, S_new, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("R,I,f_R,f_I,viscosity,slip_length", PARAM_COMBOS)
def test_total_friction_matches_legacy(order, R, I, f_R, f_I, viscosity, slip_length):
    ctx = SourceContext(time=0.0, dt=1e-3)
    for values in _random_states(order):
        P_old, _ = source_terms.compute_friction_matrix(
            order=order,
            values=values,
            R=R,
            I=I,
            mixing_friction_model=_FixedMixingFriction(f_R, f_I),
            viscosity=viscosity,
            slip_length=slip_length,
            context=ctx,
        )
        P_new = source_terms.compute_total_friction(
            order, values, f_R=f_R, f_I=f_I, viscosity=viscosity, slip_length=slip_length
        )
        assert np.allclose(P_old, P_new, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("R,I,f_R,f_I,viscosity,slip_length", PARAM_COMBOS)
def test_total_source_matches_legacy(order, R, I, f_R, f_I, viscosity, slip_length):
    """End-to-end: S_total = S_{R,I}(U) - P(U), matching
    recharge.source_terms.compute_total_source (used by the still-present
    RechargeSWME1D.compute_source_term)."""
    ctx = SourceContext(time=0.0, dt=1e-3)
    for values in _random_states(order):
        S_total_old, _ = source_terms.compute_total_source(
            order=order,
            values=values,
            rainfall=R,
            infiltration_model=_FixedInfiltration(I),
            context=ctx,
            mixing_friction_model=_FixedMixingFriction(f_R, f_I),
            viscosity=viscosity,
            slip_length=slip_length,
        )

        S_new = source_terms.compute_recharge_mass_source(order, values, R=R, I=I)
        P_new = source_terms.compute_total_friction(
            order, values, f_R=f_R, f_I=f_I, viscosity=viscosity, slip_length=slip_length
        )
        assert np.allclose(S_total_old, S_new - P_new, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("order", [3, 4, 5, 6])
def test_generic_recharge_functions_run_beyond_legacy_cap(order):
    """No legacy code exists above N=2 to compare against (that IS the
    NotImplementedError being removed - RESTRUCTURE_PLAN.md decision to lift
    the N=0,1,2 cap). This just confirms the generic functions execute and
    produce finite output for higher N, exercising the arbitrary-N coefficient
    engine end-to-end on the recharge side."""
    rng = np.random.default_rng(SEED + 700 + order)
    values = _random_state(order, rng)

    S = source_terms.compute_recharge_mass_source(order, values, R=0.15, I=-0.05)
    P = source_terms.compute_total_friction(
        order, values, f_R=0.15, f_I=0.05, viscosity=1e-3, slip_length=1.0
    )

    assert S.shape == (order + 2,)
    assert P.shape == (order + 2,)
    assert np.all(np.isfinite(S))
    assert np.all(np.isfinite(P))


@pytest.mark.parametrize("order", ORDERS)
def test_total_friction_decomposes_into_slip_plus_mixing(order):
    """compute_total_friction must equal compute_navier_slip_friction (swme)
    plus compute_mixing_friction (recharge) - the exact decomposition the
    swme/recharge package boundary is built on."""
    from swme.source_terms import compute_navier_slip_friction

    rng = np.random.default_rng(SEED + 900 + order)
    values = _random_state(order, rng)
    viscosity, slip_length, f_R, f_I = 4e-3, 0.7, 0.2, 0.08

    total = source_terms.compute_total_friction(
        order, values, f_R=f_R, f_I=f_I, viscosity=viscosity, slip_length=slip_length
    )
    slip = compute_navier_slip_friction(order, values, viscosity=viscosity, slip_length=slip_length)
    mix = source_terms.compute_mixing_friction(order, values, f_R=f_R, f_I=f_I)

    assert np.allclose(total, slip + mix, atol=1e-12, rtol=1e-12)
