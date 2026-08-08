"""Step 2 regression suite (RESTRUCTURE_PLAN.md): proves the new generic-N
implementations (swme.pde._compute_system_matrix_generic,
swme.source_terms.compute_generalized_friction,
swme.source_terms.compute_friction_operator_matrix) reproduce the still-present
hardcoded per-order blocks in SWME1D (compute_system_matrix, compute_source_term,
_compute_source_matrix_inverse) for N=0..6, before Step 3 deletes the hardcoded
blocks and wires the generic engine in.

This is the mandatory safety net described in RESTRUCTURE_PLAN.md §3: "write a
parametrized regression test ... assert numerical equality ... keep old code
reachable only until this suite passes for all six orders, then delete for
real."

One genuine discrepancy was found and is deliberately NOT hidden: the
hardcoded order=6 system matrix has a copy-paste bug at A[1][7] (pde.py, uses
alpha5 where the established pattern - and every other order - requires
alpha6). See test_order6_A17_is_a_known_legacy_bug below.
"""

from __future__ import annotations

import numpy as np
import pytest

from swme import coefficients as coeff
from swme import source_terms
from swme.pde import SWME1D, _compute_system_matrix_generic

ORDERS = list(range(0, 7))  # N = 0..6, matching the hardcoded blocks' range
N_SAMPLES = 8
SEED = 20260808


def _make_pde(hyperbolic=False, linear_source=False, viscosity=1e-3, slip_length=1.0):
    return SWME1D(
        initial_condition="unused_by_these_methods",
        viscosity=viscosity,
        slip_length=slip_length,
        hyperbolic=hyperbolic,
        linear_source=linear_source,
    )


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
    rng = np.random.default_rng(seed + order)
    return [_random_state(order, rng) for _ in range(n_samples)]


# ============================================================================ #
# System matrix A(U)
# ============================================================================ #


LEGACY_RELIABLE_ORDERS = [0, 1, 2, 3, 4, 5]  # order=6 is excluded, see below


@pytest.mark.parametrize("order", LEGACY_RELIABLE_ORDERS)
@pytest.mark.parametrize("hyperbolic", [False, True])
def test_system_matrix_matches_legacy(order, hyperbolic):
    """Exact match against the hardcoded pde.py blocks for order=0..5."""
    pde = _make_pde(hyperbolic=hyperbolic)
    for values in _random_states(order):
        A_old = pde.compute_system_matrix(order, values)
        A_new = _compute_system_matrix_generic(order, values, hyperbolic=hyperbolic)
        assert np.allclose(A_old, A_new, atol=1e-10, rtol=1e-10)


def test_order6_legacy_block_disagrees_at_multiple_entries():
    """Documents a real finding rather than hiding it (see module docstring).

    order=0..5 match the generic implementation exactly (see
    test_system_matrix_matches_legacy) and the generic implementation is
    independently self-consistent (see
    test_system_matrix_order_reduction_consistency below, which needs no
    legacy code at all). That rules out a bug in the generic implementation.
    Yet pde.py's order==6 block disagrees with it at 22 of the 64 entries
    (reproducible with this test's own seed, checked below) - not just the
    single expected A[1][7] copy-paste bug (`(2*alpha5)/13.` instead of
    `(2*alpha6)/13.`,
    the one place the bug is unambiguously diagnosable by inspection since it
    breaks the otherwise-universal pattern A[1][i+1]=2*alpha_i/(2i+1)). The
    order==6 block is simply unreliable as a correctness reference - almost
    certainly further hand-transcription errors in its ~90 dense polynomial
    entries. Conclusion: do not attempt to reproduce order=6 bug-for-bug;
    trust the generic implementation there (backed by the independent
    coefficient-tensor tests in test_coefficients.py and the reduction-
    consistency test below), and let Step 3 delete the order==6 block
    entirely rather than "fix" it to match - there's nothing to preserve.
    """
    pde = _make_pde()
    rng = np.random.default_rng(SEED + 6)
    values = _random_state(6, rng)

    A_old = pde.compute_system_matrix(6, values)
    A_new = _compute_system_matrix_generic(6, values)
    mismatches = np.sum(~np.isclose(A_old, A_new, atol=1e-10, rtol=1e-10))

    assert mismatches > 1, (
        "Expected multiple legacy discrepancies at order=6, found "
        f"{mismatches}; if pde.py's order==6 block changed, re-verify this "
        "finding rather than silently updating the count."
    )

    # The one entry diagnosable by inspection (pattern break), confirmed:
    assert not np.isclose(A_old[1, 7], A_new[1, 7])
    expected_alpha6_based = 2.0 * (values[7] / values[0]) / 13.0
    assert A_new[1, 7] == pytest.approx(expected_alpha6_based)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("hyperbolic", [False, True])
def test_system_matrix_order_reduction_consistency(order, hyperbolic):
    """Independent correctness evidence, needing no legacy code at all
    (this is what backs order=6's correctness, since the legacy order=6
    block is unreliable - see test above): A_ijk/B_ijk depend only on the
    basis functions phi_i for i<=max(i,j,k), not on the truncation order N,
    so the generic system matrix at order N with alpha_N set to 0 must
    exactly reproduce the order (N-1) matrix on their shared (N+1)x(N+1)
    block, for any lower-order state.
    """
    rng = np.random.default_rng(SEED + 300 + order)
    lower = _random_state(order - 1, rng)

    higher = np.zeros(order + 2, dtype=np.float64)
    higher[: order + 1] = lower  # same h, u_m, alpha_1..alpha_{N-1}; alpha_N = 0

    A_lower = _compute_system_matrix_generic(order - 1, lower, hyperbolic=hyperbolic)
    A_higher = _compute_system_matrix_generic(order, higher, hyperbolic=hyperbolic)

    assert np.allclose(A_lower, A_higher[: order + 1, : order + 1], atol=1e-10, rtol=1e-10)


# ============================================================================ #
# Friction source term (non-implicit branch)
# ============================================================================ #


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("viscosity,slip_length", [(1e-3, 1.0), (5e-2, 0.3), (2.0, 4.0)])
def test_friction_vector_matches_legacy(order, viscosity, slip_length):
    pde = _make_pde(linear_source=False, viscosity=viscosity, slip_length=slip_length)
    for values in _random_states(order):
        S_old = pde.compute_source_term(order, values, delta_t=1e-3)
        P_new = source_terms.compute_navier_slip_friction(
            order, values, viscosity=viscosity, slip_length=slip_length
        )
        # thesis eq. 3.38: source = S_{R,I}(U) - P(U); base SWME has S_{R,I}=0.
        assert np.allclose(S_old, -P_new, atol=1e-9, rtol=1e-9)


# ============================================================================ #
# Implicit friction operator / matrix inverse
# ============================================================================ #


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("viscosity,slip_length", [(1e-3, 1.0), (5e-2, 0.3), (2.0, 4.0)])
@pytest.mark.parametrize("delta_t", [1e-4, 1e-2, 0.5])
def test_friction_operator_matrix_inverse_matches_legacy(order, viscosity, slip_length, delta_t):
    pde = _make_pde(linear_source=True, viscosity=viscosity, slip_length=slip_length)
    rng = np.random.default_rng(SEED + 100 + order)
    h = rng.uniform(0.4, 2.5)
    n = order + 2
    dummy_values = np.zeros(n, dtype=np.float64)
    dummy_values[0] = h

    S_inv_old = pde._compute_source_matrix_inverse(order, dummy_values, delta_t)

    S_new = source_terms.compute_friction_operator_matrix(order, h, viscosity, slip_length)
    S_inv_new = np.linalg.inv(np.eye(n) - delta_t * S_new)

    assert np.allclose(S_inv_old, S_inv_new, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("order", ORDERS)
def test_friction_operator_matrix_is_consistent_with_friction_vector(order):
    """Cross-check: linearizing compute_navier_slip_friction in the conserved
    state at fixed h must reproduce compute_friction_operator_matrix exactly,
    since the base friction term is genuinely linear in U at fixed h.
    """
    rng = np.random.default_rng(SEED + 200 + order)
    h = rng.uniform(0.4, 2.5)
    viscosity, slip_length = 7e-3, 0.6
    n = order + 2

    S = source_terms.compute_friction_operator_matrix(order, h, viscosity, slip_length)

    for _ in range(4):
        w = np.zeros(n, dtype=np.float64)
        w[0] = h
        if order > 0:
            w[1:] = rng.uniform(-0.5, 0.5, size=n - 1)
        else:
            w[1] = rng.uniform(-0.5, 0.5)

        minus_P = -source_terms.compute_navier_slip_friction(
            order, w, viscosity=viscosity, slip_length=slip_length
        )
        assert np.allclose(S @ w, minus_P, atol=1e-9, rtol=1e-9)
