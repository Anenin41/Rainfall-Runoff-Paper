"""Regression suite pinning the generic-N recharge implementations to the
behavior of the legacy hardcoded N=0/1/2 code that Step 3 deleted.

Same mechanism as tests/test_pde_regression.py: Step 2 proved the generic
functions match the legacy `compute_recharge_source_n0/n1/n2` /
`compute_friction_matrix_n0/n1/n2` / `compute_total_source` by direct
comparison; before Step 3 deleted them, their outputs were captured to
`tests/data/legacy_recharge_golden.npz`, and this suite now checks the generic
functions against that captured reference so the safety net survives the
deletion. Unlike the swme side, no legacy bugs were found here on the orders
the legacy code supported (N=0,1,2) - one bug WAS found during Step 2 but it
was in the newly written generic code (a sign error on the F_ij infiltration
term), caught precisely because these comparisons existed.

See RESTRUCTURE_PLAN.md Step 2/3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from recharge import source_terms
from recharge.context import SourceContext
from recharge.recharge_pde import RechargeSWME1D
from swme.source_terms import compute_navier_slip_friction

GOLDEN = np.load(Path(__file__).parent / "data" / "legacy_recharge_golden.npz")

LEGACY_ORDERS = [0, 1, 2]  # the only orders the deleted legacy code supported
COMBOS = [
    # (R, I, f_R, f_I, viscosity, slip_length)
    (0.1, -0.05, 0.1, 0.05, 1e-3, 1.0),
    (0.0, 0.02, 0.0, 0.01, 5e-2, 0.3),
    (0.3, 0.3, 0.2, 0.0, 2.0, 4.0),
]
SEED = 20260808


class _FixedInfiltration:
    """Stand-in infiltration model: fixed I regardless of inputs, isolating the
    math under test from recharge/laws.py's closures (already order-generic)."""

    def __init__(self, I_value):
        self.I_value = I_value

    def rate(self, t, rainfall, h, dt):
        return self.I_value


class _FixedMixingFriction:
    """Stand-in mixing-friction model: fixed (f_R, f_I), same reasoning."""

    def __init__(self, f_R, f_I):
        self.f_R = f_R
        self.f_I = f_I

    def evaluate(self, R, I, context, values):
        return self.f_R, self.f_I


def _states(order):
    return GOLDEN[f"states_{order}"]


@pytest.mark.parametrize("order", LEGACY_ORDERS)
@pytest.mark.parametrize("k", range(len(COMBOS)))
def test_recharge_mass_source_matches_legacy_golden(order, k):
    R, I, _f_R, _f_I, _nu, _lam = COMBOS[k]
    expected = GOLDEN[f"S_{order}_{k}"]
    for values, S_ref in zip(_states(order), expected):
        S_new = source_terms.compute_recharge_mass_source(order, values, R=R, I=I)
        assert np.allclose(S_new, S_ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("order", LEGACY_ORDERS)
@pytest.mark.parametrize("k", range(len(COMBOS)))
def test_total_friction_matches_legacy_golden(order, k):
    _R, _I, f_R, f_I, nu, lam = COMBOS[k]
    expected = GOLDEN[f"P_{order}_{k}"]
    for values, P_ref in zip(_states(order), expected):
        P_new = source_terms.compute_total_friction(
            order, values, f_R=f_R, f_I=f_I, viscosity=nu, slip_length=lam
        )
        assert np.allclose(P_new, P_ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("order", LEGACY_ORDERS)
@pytest.mark.parametrize("k", range(len(COMBOS)))
def test_total_source_matches_legacy_golden(order, k):
    """End-to-end S_total = S_{R,I}(U) - P(U), against the legacy
    compute_total_source's captured output."""
    R, I, f_R, f_I, nu, lam = COMBOS[k]
    expected = GOLDEN[f"T_{order}_{k}"]
    for values, T_ref in zip(_states(order), expected):
        S = source_terms.compute_recharge_mass_source(order, values, R=R, I=I)
        P = source_terms.compute_total_friction(
            order, values, f_R=f_R, f_I=f_I, viscosity=nu, slip_length=lam
        )
        assert np.allclose(S - P, T_ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("order", LEGACY_ORDERS)
@pytest.mark.parametrize("k", range(len(COMBOS)))
def test_recharge_pde_source_term_matches_legacy_golden(order, k):
    """RechargeSWME1D.compute_source_term must now BE the generic path
    (Step 3 wiring) and still reproduce the legacy total source."""
    R, I, f_R, f_I, nu, lam = COMBOS[k]
    pde = RechargeSWME1D(
        initial_condition="unused",
        viscosity=nu,
        slip_length=lam,
        hyperbolic=False,
        linear_source=False,
        rainfall_rate=R,
        infiltration_model=_FixedInfiltration(I),
        mixing_friction_model=_FixedMixingFriction(f_R, f_I),
    )
    expected = GOLDEN[f"T_{order}_{k}"]
    for values, T_ref in zip(_states(order), expected):
        assert np.allclose(
            pde.compute_source_term(order, values, delta_t=1e-3), T_ref, atol=1e-10, rtol=1e-10
        )


@pytest.mark.parametrize("order", [3, 4, 5, 6, 8])
def test_recharge_runs_beyond_legacy_cap(order):
    """The legacy code raised NotImplementedError for order not in (0,1,2).
    Lifting that cap is a headline goal of this restructure."""
    rng = np.random.default_rng(SEED + 700 + order)
    h = rng.uniform(0.4, 2.5)
    values = np.empty(order + 2)
    values[0] = h
    values[1] = h * rng.uniform(-0.6, 0.6)
    values[2:] = h * rng.uniform(-0.6, 0.6, size=order)

    pde = RechargeSWME1D(
        initial_condition="unused",
        viscosity=1e-3,
        slip_length=1.0,
        hyperbolic=False,
        linear_source=False,
        rainfall_rate=0.15,
        infiltration_model=_FixedInfiltration(-0.05),
        mixing_friction_model=_FixedMixingFriction(0.15, 0.05),
    )
    total = pde.compute_source_term(order, values, delta_t=1e-3)
    assert total.shape == (order + 2,)
    assert np.all(np.isfinite(total))


@pytest.mark.parametrize("order", [0, 1, 2, 4])
def test_total_friction_decomposes_into_slip_plus_mixing(order):
    """compute_total_friction must equal swme's Navier-slip friction plus
    recharge's mixing friction - the exact decomposition the swme/recharge
    package boundary is built on."""
    rng = np.random.default_rng(SEED + 900 + order)
    h = rng.uniform(0.4, 2.5)
    values = np.empty(order + 2)
    values[0] = h
    values[1] = h * rng.uniform(-0.6, 0.6)
    if order:
        values[2:] = h * rng.uniform(-0.6, 0.6, size=order)
    nu, lam, f_R, f_I = 4e-3, 0.7, 0.2, 0.08

    total = source_terms.compute_total_friction(
        order, values, f_R=f_R, f_I=f_I, viscosity=nu, slip_length=lam
    )
    slip = compute_navier_slip_friction(order, values, viscosity=nu, slip_length=lam)
    mix = source_terms.compute_mixing_friction(order, values, f_R=f_R, f_I=f_I)
    assert np.allclose(total, slip + mix, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_zero_rainfall_and_infiltration_reduces_to_base_friction(order):
    """R=I=f_R=f_I=0 must collapse the recharge model onto the plain SWME
    source term (-P_slip), i.e. recharge is a strict extension."""
    rng = np.random.default_rng(SEED + 950 + order)
    h = rng.uniform(0.4, 2.5)
    values = np.empty(order + 2)
    values[0] = h
    values[1] = h * rng.uniform(-0.6, 0.6)
    if order:
        values[2:] = h * rng.uniform(-0.6, 0.6, size=order)
    nu, lam = 3e-3, 0.9

    S = source_terms.compute_recharge_mass_source(order, values, R=0.0, I=0.0)
    P = source_terms.compute_total_friction(
        order, values, f_R=0.0, f_I=0.0, viscosity=nu, slip_length=lam
    )
    base = -compute_navier_slip_friction(order, values, viscosity=nu, slip_length=lam)
    assert np.allclose(S, 0.0, atol=1e-14)
    assert np.allclose(S - P, base, atol=1e-12, rtol=1e-12)
