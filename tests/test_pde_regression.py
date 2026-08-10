"""Regression suite pinning the generic-N swme implementations to the behavior
of the legacy hardcoded per-order code that Step 3 deleted.

HOW THIS WORKS (and why it still has teeth after the legacy code is gone):
Step 2 proved, by direct comparison, that the generic implementations match the
hardcoded `if order == 0/1/.../6:` blocks in `pde.py` for N=0..5 (and found the
N=6 block to be buggy - see below). Before Step 3 deleted those blocks, their
outputs were captured to `tests/data/legacy_golden.npz`. This suite now checks
the generic implementations against that captured reference, so the safety net
survives the deletion permanently: if anyone later breaks
`_compute_system_matrix_generic`, `compute_navier_slip_friction`, or
`compute_friction_operator_matrix`, these tests fail against values that were
independently validated against the original Mathematica-derived code.

N=6 CAVEAT: the legacy order==6 system-matrix block disagreed with the generic
implementation at 22 of 64 entries - not a generic-implementation bug (ruled
out by the order-reduction consistency check below, which needs no legacy code
at all, plus the independent coefficient-tensor tests in test_coefficients.py),
but hand-transcription errors in the legacy block's ~90 dense polynomial
entries, the most obvious being `A[1][7] = (2*alpha5)/13.` where every other
order follows `A[1][i+1] = 2*alpha_i/(2i+1)` (i.e. it should be alpha6). The
N=6 system-matrix goldens were therefore captured from the GENERIC
implementation, not the legacy one; every other golden (all orders' friction
vectors and friction-operator inverses, and N=0..5 system matrices) came from
the legacy code. See RESTRUCTURE_PLAN.md Step 2/3.

REGENERATING THE GOLDENS: don't, unless you have deliberately changed the
physics. The whole point is that they encode the pre-refactor reference. Their
provenance is the legacy code, which no longer exists in the working tree (see
git history prior to the Step 3 commit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from swme import source_terms
from swme.pde import SWME1D, _compute_system_matrix_generic

GOLDEN = np.load(Path(__file__).parent / "data" / "legacy_golden.npz")

ORDERS = list(range(0, 7))
VISC_SLIP = [(1e-3, 1.0), (5e-2, 0.3), (2.0, 4.0)]
DTS = [1e-4, 1e-2, 0.5]
SEED = 20260808


def _states(order):
    return GOLDEN[f"states_{order}"]


# ============================================================================ #
# System matrix A(U)
# ============================================================================ #


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("hyperbolic", [False, True])
def test_system_matrix_matches_legacy_golden(order, hyperbolic):
    expected = GOLDEN[f"A_{order}_{int(hyperbolic)}"]
    for values, A_ref in zip(_states(order), expected):
        A_new = _compute_system_matrix_generic(order, values, hyperbolic=hyperbolic)
        assert np.allclose(A_new, A_ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("hyperbolic", [False, True])
def test_system_matrix_via_pde_class_matches_generic(order, hyperbolic):
    """SWME1D.compute_system_matrix must now BE the generic implementation
    (Step 3 wiring), including its input validation still firing."""
    pde = SWME1D("unused", 1e-3, 1.0, hyperbolic, False)
    for values in _states(order):
        assert np.allclose(
            pde.compute_system_matrix(order, values),
            _compute_system_matrix_generic(order, values, hyperbolic=hyperbolic),
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("hyperbolic", [False, True])
def test_system_matrix_order_reduction_consistency(order, hyperbolic):
    """Independent correctness evidence needing no legacy code or goldens at
    all (this is what backs N=6, where the legacy block was buggy): A_ijk and
    B_ijk depend only on the basis functions involved, not on the truncation
    order N, so the generic system matrix at order N with alpha_N = 0 must
    exactly reproduce the order (N-1) matrix on their shared block.
    """
    rng = np.random.default_rng(SEED + 300 + order)
    n_low = order + 1
    lower = np.empty(n_low)
    h = rng.uniform(0.4, 2.5)
    lower[0] = h
    lower[1] = h * rng.uniform(-0.6, 0.6)
    if order - 1 > 0:
        lower[2:] = h * rng.uniform(-0.6, 0.6, size=order - 1)

    higher = np.zeros(order + 2)
    higher[:n_low] = lower  # same h, u_m, alpha_1..alpha_{N-1}; alpha_N = 0

    A_lower = _compute_system_matrix_generic(order - 1, lower, hyperbolic=hyperbolic)
    A_higher = _compute_system_matrix_generic(order, higher, hyperbolic=hyperbolic)
    assert np.allclose(A_lower, A_higher[:n_low, :n_low], atol=1e-10, rtol=1e-10)


def test_system_matrix_validation_still_rejects_bad_input():
    """Input validation that the legacy compute_system_matrix performed must
    survive the Step 3 rewrite.

    One clause was deliberately relaxed in Step 6: h == 0 used to be rejected
    as "non-positive height", but a dry cell is now a legitimate state, so only
    a *negative* height is an error. See tests/test_wetdry.py.
    """
    pde = SWME1D("unused", 1e-3, 1.0, False, False)
    with pytest.raises(ValueError):
        pde.compute_system_matrix(1, np.array([[1.0, 0.0, 0.0]]))  # not 1D
    with pytest.raises(ValueError):
        pde.compute_system_matrix(1, np.array([1.0, np.nan, 0.0]))  # non-finite
    with pytest.raises(ValueError):
        pde.compute_system_matrix(1, np.array([-1.0, 0.0, 0.0]))  # h < 0

    # h == 0 is now accepted and yields a finite, dry transport matrix.
    dry = pde.compute_system_matrix(1, np.array([0.0, 0.0, 0.0]))
    assert np.isfinite(dry).all()


# ============================================================================ #
# Friction source term (explicit / vector branch)
# ============================================================================ #


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("k", range(len(VISC_SLIP)))
def test_friction_vector_matches_legacy_golden(order, k):
    viscosity, slip_length = VISC_SLIP[k]
    expected = GOLDEN[f"S_{order}_{k}"]
    for values, S_ref in zip(_states(order), expected):
        P_new = source_terms.compute_navier_slip_friction(
            order, values, viscosity=viscosity, slip_length=slip_length
        )
        # thesis eq. 3.38: source = S_{R,I}(U) - P(U); base SWME has S_{R,I}=0.
        assert np.allclose(-P_new, S_ref, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("k", range(len(VISC_SLIP)))
def test_friction_vector_via_pde_class_matches_golden(order, k):
    """SWME1D.compute_source_term (explicit branch) must now BE the generic
    implementation (Step 3 wiring)."""
    viscosity, slip_length = VISC_SLIP[k]
    pde = SWME1D("unused", viscosity, slip_length, False, False)
    expected = GOLDEN[f"S_{order}_{k}"]
    for values, S_ref in zip(_states(order), expected):
        assert np.allclose(
            pde.compute_source_term(order, values, delta_t=1e-3), S_ref, atol=1e-9, rtol=1e-9
        )


# ============================================================================ #
# Implicit friction operator / matrix inverse
# ============================================================================ #


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("k", range(len(VISC_SLIP)))
@pytest.mark.parametrize("m", range(len(DTS)))
def test_friction_operator_matrix_inverse_matches_legacy_golden(order, k, m):
    viscosity, slip_length = VISC_SLIP[k]
    delta_t = DTS[m]
    hs = GOLDEN[f"h_{order}_{k}"]
    expected = GOLDEN[f"Sinv_{order}_{k}_{m}"]
    n = order + 2
    for h, S_inv_ref in zip(hs, expected):
        S = source_terms.compute_friction_operator_matrix(order, h, viscosity, slip_length)
        S_inv_new = np.linalg.inv(np.eye(n) - delta_t * S)
        assert np.allclose(S_inv_new, S_inv_ref, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("k", range(len(VISC_SLIP)))
@pytest.mark.parametrize("m", range(len(DTS)))
def test_matrix_inverse_via_pde_class_matches_golden(order, k, m):
    """SWME1D._compute_source_matrix_inverse must now BE the generic runtime
    solve (Step 3 wiring), replacing ~1700 lines of Mathematica-derived
    closed forms."""
    viscosity, slip_length = VISC_SLIP[k]
    delta_t = DTS[m]
    pde = SWME1D("unused", viscosity, slip_length, False, True)
    hs = GOLDEN[f"h_{order}_{k}"]
    expected = GOLDEN[f"Sinv_{order}_{k}_{m}"]
    for h, S_inv_ref in zip(hs, expected):
        dummy = np.zeros(order + 2)
        dummy[0] = h
        assert np.allclose(
            pde._compute_source_matrix_inverse(order, dummy, delta_t),
            S_inv_ref,
            atol=1e-8,
            rtol=1e-8,
        )


@pytest.mark.parametrize("order", ORDERS)
def test_friction_operator_matrix_is_consistent_with_friction_vector(order):
    """Independent structural check (no goldens): linearizing
    compute_navier_slip_friction in the conserved state at fixed h must
    reproduce compute_friction_operator_matrix exactly, since the base
    friction term is genuinely linear in U at fixed h.
    """
    rng = np.random.default_rng(SEED + 200 + order)
    h = rng.uniform(0.4, 2.5)
    viscosity, slip_length = 7e-3, 0.6
    n = order + 2
    S = source_terms.compute_friction_operator_matrix(order, h, viscosity, slip_length)

    for _ in range(4):
        w = np.zeros(n)
        w[0] = h
        w[1:] = rng.uniform(-0.5, 0.5, size=n - 1)
        minus_P = -source_terms.compute_navier_slip_friction(
            order, w, viscosity=viscosity, slip_length=slip_length
        )
        assert np.allclose(S @ w, minus_P, atol=1e-9, rtol=1e-9)


# ============================================================================ #
# Arbitrary N beyond the legacy cap
# ============================================================================ #


@pytest.mark.parametrize("order", [7, 8, 10])
def test_generic_swme_runs_beyond_legacy_cap(order):
    """The legacy hardcoded blocks stopped at N=6 (silently returning a zero
    matrix above it). Arbitrary N is the point of this restructure."""
    rng = np.random.default_rng(SEED + 800 + order)
    h = rng.uniform(0.4, 2.5)
    values = np.empty(order + 2)
    values[0] = h
    values[1] = h * rng.uniform(-0.6, 0.6)
    values[2:] = h * rng.uniform(-0.6, 0.6, size=order)

    A = _compute_system_matrix_generic(order, values)
    P = source_terms.compute_navier_slip_friction(order, values, viscosity=1e-3, slip_length=1.0)
    S = source_terms.compute_friction_operator_matrix(order, h, 1e-3, 1.0)

    assert A.shape == (order + 2, order + 2)
    assert P.shape == (order + 2,)
    assert S.shape == (order + 2, order + 2)
    assert np.all(np.isfinite(A)) and np.all(np.isfinite(P)) and np.all(np.isfinite(S))
    # A must be non-trivial (the legacy code silently returned zeros above N=6)
    assert np.any(A != 0.0)
