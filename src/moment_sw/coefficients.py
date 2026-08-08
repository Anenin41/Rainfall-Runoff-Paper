"""Generic-N shifted-Legendre moment coefficient engine.

This module replaces the per-order hardcoded blocks that used to live in
``pde.py`` and ``recharge/source_terms.py`` (``if order == 0/1/.../6:``) with a
single engine that computes the projection tensors/matrices/vectors needed by
the Shallow Water Moment / Hyperbolic SWME / Recharge SWME closures for any
moment order N, following the general formulas of Appendix C of the thesis.

Basis convention (shifted Legendre polynomials on [0, 1]):

    phi_i(z) = P_i(1 - 2z),   phi_i(1) = (-1)^i,   phi_i(0) = 1

The tensors A, B, C are computed exactly with sympy (rational arithmetic) and
cast to float64; this logic was originally prototyped in
``symbolic_math/symbo.py`` and is ported here (not imported) since that
directory is deleted once its math is absorbed into the solver. The vectors
r, s and matrices E, F have closed forms (proven in thesis Appendix B) and are
implemented directly, with no sympy involved.

Coefficients are cached per N with ``functools.lru_cache`` — this is a
startup-time cost paid once per (order), not a per-cell/per-timestep cost, so
no persisted on-disk cache is used. See RESTRUCTURE_PLAN.md, Section 1, for
the full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
import sympy as sp
from sympy.physics.wigner import wigner_3j


@dataclass(frozen=True)
class Coefficients:
    """All (N+1)-indexed tensors needed by the SWME/HSWME/Recharge closures.

    Index 0 corresponds to the depth-averaged ("mean") Legendre mode; the
    physical moment variables alpha_1..alpha_N correspond to tensor indices
    1..N. Callers that only need the moment-moment block slice [1:, 1:, ...].

    Attributes
    ----------
    N : int
        Maximum moment order.
    A : np.ndarray, shape (N+1, N+1, N+1)
        Conservative quadratic-flux self-interaction tensor.
    B : np.ndarray, shape (N+1, N+1, N+1)
        Non-conservative vertical-coupling transport tensor.
    C : np.ndarray, shape (N+1, N+1)
        Viscous/friction (Navier-slip) coupling matrix.
    E : np.ndarray, shape (N+1, N+1)
        Rainfall-side mixing matrix (affine weight zeta).
    F : np.ndarray, shape (N+1, N+1)
        Bed-exchange-side mixing matrix (affine weight zeta - 1).
    r : np.ndarray, shape (N+1,)
        Rainfall-side auxiliary vector, r_i = (-1)^i for i>=1, r_0 = 0.
    s : np.ndarray, shape (N+1,)
        Bed-exchange-side auxiliary vector, s_i = 1 for i>=1, s_0 = 0.
    phi_at_1 : np.ndarray, shape (N+1,)
        Basis endpoint values at the free surface, phi_i(1) = (-1)^i.
    phi_at_0 : np.ndarray, shape (N+1,)
        Basis endpoint values at the wet boundary, phi_i(0) = 1.
    """

    N: int
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    E: np.ndarray
    F: np.ndarray
    r: np.ndarray
    s: np.ndarray
    phi_at_1: np.ndarray
    phi_at_0: np.ndarray


# ============================================================================ #
# Private sympy-backed basis/tensor machinery (ported from                     #
# symbolic_math/symbo.py). Not part of the public API - use get_coefficients.  #
# ============================================================================ #


@dataclass(frozen=True)
class _Basis:
    """Polynomial basis container for shifted Legendre polynomials on [0, 1]."""

    N: int
    z: sp.Symbol
    phisP: List[sp.Poly]
    dphisP: List[sp.Poly]
    zP: sp.Poly


def _build_shifted_legendre_basis(N: int, z: sp.Symbol) -> _Basis:
    """Build phi_i(z) = P_i(1 - 2z) for i = 0..N as exact rational polynomials."""
    phisP = [sp.Poly(sp.legendre(i, 1 - 2 * z), z, domain="QQ") for i in range(N + 1)]
    dphisP = [p.diff(z) for p in phisP]
    zP = sp.Poly(z, z, domain="QQ")
    return _Basis(N=N, z=z, phisP=phisP, dphisP=dphisP, zP=zP)


def _integrate_poly(poly_or_expr: sp.Expr | sp.Poly, z: sp.Symbol) -> sp.Rational:
    """Exact integral over z in [0, 1] via antiderivative evaluation (fast,
    avoids sympy's general-purpose integrate()/simplify())."""
    if isinstance(poly_or_expr, sp.Poly):
        p = poly_or_expr
    else:
        p = sp.Poly(poly_or_expr, z, domain="QQ")
    Pint = p.integrate()
    return sp.Rational(Pint.eval(1) - Pint.eval(0))


def _triple_int_phi(i: int, j: int, k: int) -> sp.Expr:
    """Closed form of integral(phi_i phi_j phi_k, z, 0, 1) via Wigner 3j symbols."""
    return (-1) ** (i + j + k) * (wigner_3j(i, j, k, 0, 0, 0) ** 2)


def _compute_A(N: int) -> List[List[List[sp.Expr]]]:
    """A_ijk = (2i+1) * integral(phi_i phi_j phi_k, z, 0, 1). No basis needed."""
    A = [[[None] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(N + 1):
                A[i][j][k] = (2 * i + 1) * _triple_int_phi(i, j, k)
    return A


def _compute_JP(basis: _Basis) -> List[sp.Poly]:
    """J_j(z) = integral(phi_j(xi), xi, 0, z), as polynomials in z."""
    z = basis.z
    JP: List[sp.Poly] = []
    for j in range(basis.N + 1):
        Pint = basis.phisP[j].integrate()
        JP.append(Pint - sp.Poly(Pint.eval(0), z, domain="QQ"))
    return JP


def _compute_B(
    basis: _Basis, JP: Optional[List[sp.Poly]] = None
) -> List[List[List[sp.Rational]]]:
    """B_ijk = (2i+1) * integral(dphi_i(z) J_j(z) phi_k(z), z, 0, 1)."""
    N, z = basis.N, basis.z
    if JP is None:
        JP = _compute_JP(basis)
    B = [[[None] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(N + 1):
                B[i][j][k] = (2 * i + 1) * _integrate_poly(
                    basis.dphisP[i] * JP[j] * basis.phisP[k], z
                )
    return B


def _compute_C(basis: _Basis) -> List[List[sp.Rational]]:
    """C_ij = integral(dphi_i(z) dphi_j(z), z, 0, 1)."""
    N, z = basis.N, basis.z
    C = [[None] * (N + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        for j in range(N + 1):
            C[i][j] = _integrate_poly(basis.dphisP[i] * basis.dphisP[j], z)
    return C


def _compute_r_s_sympy(basis: _Basis) -> Tuple[List[sp.Rational], List[sp.Rational]]:
    """Exact sympy computation of r_i, s_i - used only to validate the closed
    forms in tests, not on the hot path (see r_i/s_i closed forms below)."""
    N, z = basis.N, basis.z
    r = [None] * (N + 1)
    s = [None] * (N + 1)
    for i in range(N + 1):
        r[i] = _integrate_poly(basis.zP * basis.dphisP[i], z)
        s[i] = _integrate_poly((basis.zP - 1) * basis.dphisP[i], z)
    return r, s


def _compute_E_F_sympy(
    basis: _Basis,
) -> Tuple[List[List[sp.Rational]], List[List[sp.Rational]]]:
    """Exact sympy computation of E_ij, F_ij - used only to validate the
    closed forms in tests, not on the hot path (see E/F closed forms below)."""
    N, z = basis.N, basis.z
    E = [[None] * (N + 1) for _ in range(N + 1)]
    F = [[None] * (N + 1) for _ in range(N + 1)]
    for i in range(N + 1):
        for j in range(N + 1):
            E[i][j] = _integrate_poly(basis.zP * basis.dphisP[i] * basis.phisP[j], z)
            F[i][j] = _integrate_poly(
                (basis.zP - 1) * basis.dphisP[i] * basis.phisP[j], z
            )
    return E, F


# ============================================================================ #
# Public API                                                                    #
# ============================================================================ #


@lru_cache(maxsize=None)
def get_coefficients(N: int) -> Coefficients:
    """Return all projection tensors/matrices/vectors for moment order N.

    Cached per N (see module docstring for the caching rationale). N=0..~10
    is the realistic domain; sympy cost is O((N+1)^3) exact rational
    evaluations, paid once per distinct N per process.

    Parameters
    ----------
    N : int
        Maximum moment order (>= 0).

    Returns
    -------
    Coefficients
    """
    if N < 0:
        raise ValueError(f"Moment order N must be >= 0, got {N}.")

    z = sp.Symbol("z", real=True)
    basis = _build_shifted_legendre_basis(N, z)
    JP = _compute_JP(basis)

    A_sym = _compute_A(N)
    B_sym = _compute_B(basis, JP=JP)
    C_sym = _compute_C(basis)

    n = N + 1
    A = np.array(
        [[[float(A_sym[i][j][k]) for k in range(n)] for j in range(n)] for i in range(n)],
        dtype=np.float64,
    )
    B = np.array(
        [[[float(B_sym[i][j][k]) for k in range(n)] for j in range(n)] for i in range(n)],
        dtype=np.float64,
    )
    C = np.array([[float(C_sym[i][j]) for j in range(n)] for i in range(n)], dtype=np.float64)

    idx = np.arange(n)
    # phi_i(1) = (-1)^i, phi_i(0) = 1 for every i, including i=0 (phi_0 == 1).
    phi_at_1 = np.where(idx % 2 == 0, 1.0, -1.0)
    phi_at_0 = np.ones(n, dtype=np.float64)

    # r_i = (-1)^i, s_i = 1 for i >= 1 (thesis Appendix B.1, via integration
    # by parts). This does NOT extend to i=0: phi_0 is the constant function
    # 1, so dphi_0/dz == 0 identically and r_0 = s_0 = 0 by direct evaluation
    # (verified against sympy in tests/test_coefficients.py). Index 0 is the
    # depth-averaged mode and is not a physical moment variable regardless.
    r = np.where(idx == 0, 0.0, np.where(idx % 2 == 0, 1.0, -1.0))
    s = np.where(idx == 0, 0.0, 1.0)

    # Closed forms for E, F (thesis Appendix B.4): lower-triangular,
    # E_ii = F_ii = i/(2i+1); E_ij = (-1)^(i+j) for j<i, F_ij = 1 for j<i;
    # both 0 for j>i.
    E = np.zeros((n, n), dtype=np.float64)
    F = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        E[i, i] = i / (2 * i + 1)
        F[i, i] = i / (2 * i + 1)
        for j in range(i):
            E[i, j] = (-1.0) ** (i + j)
            F[i, j] = 1.0

    return Coefficients(
        N=N, A=A, B=B, C=C, E=E, F=F, r=r, s=s, phi_at_1=phi_at_1, phi_at_0=phi_at_0
    )


def eval_phi(N: int, z: np.ndarray) -> np.ndarray:
    """Evaluate phi_0..phi_N at the given z points via a float64 Bonnet
    recurrence (no sympy at runtime). Used only for post-processing (vertical
    velocity profile reconstruction), never on the hot simulation path.

    Parameters
    ----------
    N : int
        Maximum moment order (>= 0).
    z : array_like
        Points in [0, 1] at which to evaluate the basis.

    Returns
    -------
    np.ndarray, shape (N+1,) + np.shape(z)
        phi_i(z) for i = 0..N.
    """
    if N < 0:
        raise ValueError(f"Moment order N must be >= 0, got {N}.")

    z_arr = np.asarray(z, dtype=np.float64)
    x = 1.0 - 2.0 * z_arr  # phi_i(z) = P_i(1 - 2z), Bonnet recurrence in x
    phi = np.empty((N + 1,) + z_arr.shape, dtype=np.float64)
    phi[0] = 1.0
    if N >= 1:
        phi[1] = x
    for k in range(1, N):
        phi[k + 1] = ((2 * k + 1) * x * phi[k] - k * phi[k - 1]) / (k + 1)
    return phi
