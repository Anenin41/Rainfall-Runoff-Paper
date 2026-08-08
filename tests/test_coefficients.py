"""Regression tests for moment_sw.coefficients.

Validates:
  - the closed forms for r, s, E, F against direct sympy integration
    (matching the derivation of thesis Appendix B.1/B.4), for N=0..8.
  - basic sanity/consistency properties of A, B, C, phi_at_0, phi_at_1.
  - eval_phi (float64 Bonnet recurrence) against the exact sympy basis.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from moment_sw import coefficients as coeff


N_VALUES = list(range(0, 9))  # N = 0..8


@pytest.mark.parametrize("N", N_VALUES)
def test_r_s_closed_form_matches_sympy(N):
    z = sp.Symbol("z", real=True)
    basis = coeff._build_shifted_legendre_basis(N, z)
    r_sym, s_sym = coeff._compute_r_s_sympy(basis)

    c = coeff.get_coefficients(N)

    for i in range(N + 1):
        assert float(r_sym[i]) == pytest.approx(c.r[i], abs=1e-12)
        assert float(s_sym[i]) == pytest.approx(c.s[i], abs=1e-12)


@pytest.mark.parametrize("N", N_VALUES)
def test_e_f_closed_form_matches_sympy(N):
    z = sp.Symbol("z", real=True)
    basis = coeff._build_shifted_legendre_basis(N, z)
    E_sym, F_sym = coeff._compute_E_F_sympy(basis)

    c = coeff.get_coefficients(N)

    for i in range(N + 1):
        for j in range(N + 1):
            assert float(E_sym[i][j]) == pytest.approx(c.E[i, j], abs=1e-12)
            assert float(F_sym[i][j]) == pytest.approx(c.F[i, j], abs=1e-12)


@pytest.mark.parametrize("N", [0, 1, 2, 3, 5])
def test_phi_endpoint_values(N):
    c = coeff.get_coefficients(N)
    for i in range(N + 1):
        assert c.phi_at_1[i] == pytest.approx((-1.0) ** i)
        assert c.phi_at_0[i] == pytest.approx(1.0)


def test_A_known_values_N0():
    # A_000 = (2*0+1) * integral(phi_0^3) = 1 * 1 = 1 (phi_0 == 1 identically)
    c = coeff.get_coefficients(0)
    assert c.A[0, 0, 0] == pytest.approx(1.0)


def test_A_vanishes_for_odd_index_sum_parity_case():
    # A_111 involves wigner_3j(1,1,1;0,0,0) which is zero (parity rule: the
    # 3j symbol with all-zero lower row vanishes unless i+j+k is even).
    c = coeff.get_coefficients(2)
    assert c.A[1, 1, 1] == pytest.approx(0.0)


def test_A_symmetric_in_last_two_indices():
    c = coeff.get_coefficients(4)
    N = c.N
    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(N + 1):
                assert c.A[i, j, k] == pytest.approx(c.A[i, k, j], abs=1e-12)


def test_shapes():
    N = 5
    c = coeff.get_coefficients(N)
    n = N + 1
    assert c.A.shape == (n, n, n)
    assert c.B.shape == (n, n, n)
    assert c.C.shape == (n, n)
    assert c.E.shape == (n, n)
    assert c.F.shape == (n, n)
    assert c.r.shape == (n,)
    assert c.s.shape == (n,)
    assert c.phi_at_1.shape == (n,)
    assert c.phi_at_0.shape == (n,)


def test_get_coefficients_is_cached():
    c1 = coeff.get_coefficients(3)
    c2 = coeff.get_coefficients(3)
    assert c1 is c2


def test_get_coefficients_rejects_negative_N():
    with pytest.raises(ValueError):
        coeff.get_coefficients(-1)


@pytest.mark.parametrize("N", [0, 1, 2, 3, 4])
def test_eval_phi_matches_exact_basis(N):
    z_points = np.array([0.0, 0.13, 0.5, 0.87, 1.0])
    phi_numeric = coeff.eval_phi(N, z_points)

    z = sp.Symbol("z", real=True)
    basis = coeff._build_shifted_legendre_basis(N, z)
    for i in range(N + 1):
        for k, zp in enumerate(z_points):
            expected = float(basis.phisP[i].eval(zp))
            assert phi_numeric[i, k] == pytest.approx(expected, abs=1e-10)


def test_eval_phi_rejects_negative_N():
    with pytest.raises(ValueError):
        coeff.eval_phi(-1, np.array([0.5]))
