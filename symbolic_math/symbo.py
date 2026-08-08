# Sympy Script for computing Matrices & Tensors of the SWME + Recharge Model #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Tue 02 Mar 2026 @ 12:12:05 +0100
# Modified: Tue 28 Apr 2026 @ 18:15:55 +0200

# Packages
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import sympy as sp
from sympy.physics.wigner import wigner_3j

# Data container
@dataclass(frozen=True)
class Basis:
    """
    Polynomial basis container for shifted Legendre polynomials on [0, 1].

    Attributes
    ----------
    N:
        Maximum polynomial order. Basis size is n = N+1.
    z:
        SymPy symbol used for the variable.
    phisP:
        List of polynomials phi_i(z) as sympy. Poly in z with rational domain.
    dphisP:
        List of derivatives d/dz phi_i(z) as sympy.Poly.
    zP:
        Polynomial representing z as sympy.Poly.
    """
    N: int
    z: sp.Symbol
    phisP: List[sp.Poly]
    dphisP: List[sp.Poly]
    zP: sp.Poly

# Building the basis and performing polynomial integration
def build_shifted_legendre_basis(N: int, z: sp.Symbol) -> Basis:
    """
    Build the shifted Legendre basis in polynomial form.

    Uses: phi_i(z) = P_i(1 - 2 z)  (connection between original and shifted
    Legendre polynomials)
    Source: Wikipedia

    Parameters
    ----------
    N:
        Maximum order.
    z:
        Variable symbol.

    Returns
    -------
    Basis
        Precomputed polynomials phi_i, their derivatives, and z as Poly.
    """
    phisP = [sp.Poly(sp.legendre(i, 1 - 2 * z), z, domain="QQ") 
             for i in range(N + 1)]
    dphisP = [p.diff(z) for p in phisP]
    zP = sp.Poly(z, z, domain="QQ")
    return Basis(N=N, z=z, phisP=phisP, dphisP=dphisP, zP=zP)


def integrate_poly(poly_or_expr: sp.Expr | sp.Poly, 
                   z: sp.Symbol) -> sp.Rational:
    """
    Exact integral over z in [0, 1] for polynomial expressions.

    This avoids SymPy's general-purpose integrate() and simplify() and is 
    practically faster for this specific application.

    Parameters
    ----------
    poly_or_expr:
        A polynomial in z (either sympy.Poly or expression).
    z:
        Variable symbol.

    Returns
    -------
    sympy.Rational
        The exact value of ∫_0^1 poly(z) dz.
    """
    if isinstance(poly_or_expr, sp.Poly):
        p = poly_or_expr
    else:
        p = sp.Poly(poly_or_expr, z, domain="QQ")

    Pint = p.integrate()
    return sp.Rational(Pint.eval(1) - Pint.eval(0))


# Core computations
def compute_r_s(basis: Basis) -> Tuple[List[sp.Rational], List[sp.Rational]]:
    """
    Compute vectors r and s.

    r_i = integral of z * dphi_i(z) over z in [0, 1]
    s_i = integral of (z - 1) * dphi_i(z) over z in [0, 1]

    Returns
    -------
    (r, s)
        Two lists of length N+1.
    """
    N, z = basis.N, basis.z
    r = [None] * (N + 1)
    s = [None] * (N + 1)

    for i in range(N + 1):
        r[i] = integrate_poly(basis.zP * basis.dphisP[i], z)
        s[i] = integrate_poly((basis.zP - 1) * basis.dphisP[i], z)

    return r, s


def compute_E_F_C(basis: Basis) -> Tuple[List[List[sp.Rational]],
                                        List[List[sp.Rational]],
                                        List[List[sp.Rational]]]:
    """
    Compute matrices E, F, C.

    E_ij = integral of z * dphi_i(z) * phi_j(z) over z in [0, 1]
    F_ij = integral of (z-1)  * dphi_i(z) * phi_j(z) over z in [0, 1]
    C_ij = integral of dphi_i(z) * dphi_j(z) over z in [0, 1]

    Returns
    -------
    (E, F, C)
        Three (N+1)x(N+1) lists-of-lists of exact rationals.
    """
    N, z = basis.N, basis.z
    E = [[None] * (N + 1) for _ in range(N + 1)]
    F = [[None] * (N + 1) for _ in range(N + 1)]
    C = [[None] * (N + 1) for _ in range(N + 1)]

    for i in range(N + 1):
        for j in range(N + 1):
            E[i][j] = integrate_poly(basis.zP * basis.dphisP[i] * basis.phisP[j], 
                                     z)
            F[i][j] = integrate_poly((basis.zP - 1) * basis.dphisP[i] * basis.phisP[j]
                                     , z)
            C[i][j] = integrate_poly(basis.dphisP[i] * basis.dphisP[j], z)

    return E, F, C


def triple_int_phi(i: int, j: int, k: int) -> sp.Expr:
    """
    It is possible to derive a closed form of the:
        integral of phi_i(z) phi_j(z) phi_k(z) over z in [0, 1]
        with phi_n(z) = P_n(1 - 2 z)
    by using spherical harmonics. It has been proven that the expression above
    is the same as Wigner 3j symbols, which is already implemented in the 
    library.

    Since P_n(1-2z) = P_n(-(2z-1)) = (-1)^n P_n(2z-1),
    the integral differs by (-1)^(i+j+k) from the P_n(2z-1) case.

    For the P_n(2z-1) definition:
        integral of P_i(2z-1) P_j(2z-1) P_k(2z-1) = (wigner_3j(i,j,k;0,0,0))^2

    Therefore for phi_n(z)=P_n(1-2z):
        integral of phi_i phi_j phi_k dz = (-1)^(i+j+k) * (wigner_3j(...))^2

    Notes:
        The 3j symbol is zero unless triangle and parity rules holdin nonzero 
        cases. In addition i+j+k is even, so the sign factor is +1. So, keeping 
        the factor is still correct.

    Returns
    -------
    sympy.Expr
        Exact expression (typically Rational) for the triple integral.
    """
    return (-1) ** (i + j + k) * (wigner_3j(i, j, k, 0, 0, 0) ** 2)


def compute_A(N: int) -> List[List[List[sp.Expr]]]:
    """
    Compute the A-tensor:

        A_{ijk} = (2i + 1) integral of phi_i(z) phi_j(z) phi_k(z) over z in [0, 1]

    Uses Wigner 3j closed form for the integral.

    Parameters
    ----------
    N:
        Maximum order.

    Returns
    -------
    A:
        3-tensor as a (N+1)x(N+1)x(N+1) nested list.
    """
    # Omit iterables here
    A = [[[None] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(N + 1):
                A[i][j][k] = (2 * i + 1) * triple_int_phi(i, j, k)
    return A


def compute_JP(basis: Basis) -> List[sp.Poly]:
    """
    Precompute J_j(z) = integral of phi_j(xi) for xi in [0, z] as polynomials.

    Returns
    -------
    JP:
        List of sympy.Poly objects (each is a polynomial in z).
    """
    z = basis.z
    JP: List[sp.Poly] = []
    for j in range(basis.N + 1):
        Pint = basis.phisP[j].integrate()
        JP.append(Pint - sp.Poly(Pint.eval(0), z, domain="QQ"))
    return JP


def compute_B(basis: Basis, 
              JP: Optional[List[sp.Poly]] = None) -> List[List[List[sp.Rational]]]:
    """
    Compute the B-tensor:

        B_{ijk} = (2i + 1) integral of dphi_i(z) * J_j(z) * phi_k(z) over z in [0, 1]
    where
        J_j(z) = integral of phi_j(xi) over xi in [0, z]

    Parameters
    ----------
    basis:
        Precomputed polynomials.
    JP:
        Optional precomputed list of J_j polynomials. If None, computed internally.

    Returns
    -------
    B:
        3-tensor as a (N+1)x(N+1)x(N+1) nested list.
    """
    N, z = basis.N, basis.z
    if JP is None:
        JP = compute_JP(basis)

    B = [[[None] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(N + 1):
                B[i][j][k] = (2 * i + 1) * integrate_poly(basis.dphisP[i] * JP[j] * basis.phisP[k], z)
    return B

#========================#
# Pretty terminal output #
#========================#

def _count_nonzero_matrix(M: List[List[sp.Expr]]) -> int:
    """
    Indicator of sparsity. Count how many non-zero entries exist in the matrix.
    """
    return sum(1 for row in M for x in row if x != 0)

def _count_nonzero_tensor3(T: List[List[List[sp.Expr]]]) -> int:
    """
    Indicator od sparsity. Count how many non-zero entries exist in the tensor.
    """
    n = len(T)
    return sum(
            1 
            for i in range(n) 
            for j in range(n) 
            for k in range(n) 
            if T[i][j][k] != 0
            )

def _moment_vector(v: List[sp.Expr]) -> List[sp.Expr]:
    """
    Drop the zeroth mode. Only moment-dependent entries are printed.
    """
    return v[1:]

def _moment_matrix(M: List[List[sp.Expr]]) -> List[List[sp.Expr]]:
    """
    Drop row/column 0. Only the moment-moment block is printed.
    """
    return [row[1:] for row in M[1:]]

def _moment_tensor3(
        T: List[List[List[sp.Expr]]]
        ) -> Tuple[List[List[List[sp.Expr]]], List[int]]:
    """
    Drop index 0 in all three tensor directions.
    Returns:
        trimmed tensor with only indices 1...N
        original slice labels [1, ..., N]
    """
    kept_indices = list(range(1, len(T)))
    trimmed = []
    for i in kept_indices:
        trimmed.append([row[1:] for row in T[i][1:]])
    return trimmed, kept_indices

def pprint_vector(v: List[sp.Expr], 
                  name: str, 
                  unicode: bool = True) -> None:
    """
    Pretty-print only the moment-dependent vector as a column matrix in the CLI.
    """
    vm = _moment_vector(v)
    if not vm:
        print(f"\n{name}: no moment-dependent entries for N=0.")
        return

    print(f"\n{name} (len={len(v)}):")
    sp.pprint(sp.Matrix(vm), use_unicode=unicode)

def pprint_matrix(M: List[List[sp.Expr]], 
                  name: str, 
                  unicode: bool = True) -> None:
    """
    Pretty-print only the moment-moment block of a matrix in the CLI.
    """
    Mm = _moment_matrix(M)
    n = len(Mm)
    if n == 0:
        print(f"\n{name}: no moment-dependent block for N=0.")
        return

    nz = _count_nonzero_matrix(Mm)
    print(f"\n{name} ({n}x{n})  nonzeros={nz}/{n*n}:")
    sp.pprint(sp.Matrix(Mm), use_unicode=unicode)

def pprint_tensor_slices(T: List[List[List[sp.Expr]]], 
                         name: str, 
                         unicode: bool = True) -> None:
    """
    Pretty-print only the purely moment-dependent tensor block, 
    i.e. indices 1...N in every direction.

    For a typical max case N=5 (size 6), this is terminal-feasible.
    """
    Tm, labels = _moment_tensor3(T)
    n = len(Tm)
    if n == 0:
        print(f"\n{name}: no moment-dependent slices for N=0.")
        return

    nz = _count_nonzero_tensor3(Tm)
    print(f"\n{name} ({n}x{n}x{n})  nonzeros={nz}/{n*n*n}:")
    for local_i, original_i in enumerate(labels):
        print(f"\n-- {name}[{original_i},1:,1:] --")
        sp.pprint(sp.Matrix(Tm[local_i]), use_unicode=unicode)

#========================================================# 
# Building a Command Line Interface with Python Argparse #
#========================================================#

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute and print SWME symbolic matrices/tensors (fast)."
    )
    p.add_argument("--N", type=int, default=5, help="Maximum order N (size = N+1). Default: 5")
    p.add_argument("--unicode", action="store_true", help="Use unicode pretty printing (default off).")

    # What to print
    p.add_argument("--r", action="store_true", help="Print vector r.")
    p.add_argument("--s", action="store_true", help="Print vector s.")
    p.add_argument("--E", action="store_true", help="Print matrix E.")
    p.add_argument("--F", action="store_true", help="Print matrix F.")
    p.add_argument("--C", action="store_true", help="Print matrix C.")
    p.add_argument("--A", action="store_true", help="Print tensor A as slices A[i,:,:].")
    p.add_argument("--B", action="store_true", help="Print tensor B as slices B[i,:,:].")
    p.add_argument("--all", action="store_true", help="Print everything.")

    return p

def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    N = args.N
    use_unicode = bool(args.unicode)

    # Decide what is requested
    want_any = (args.all or args.r or args.s or args.E or 
                args.F or args.C or args.A or args.B)
    if not want_any:
        # Show the --h option when nothing is selected
        parser.print_help()
        return

    if args.all:
        args.r = args.s = args.E = args.F = args.C = args.A = args.B = True

    print(f"N = {N} (size={N+1})")
    z = sp.Symbol("z", real=True)

    # Compute only what is needed:
    # - A does NOT require building the polynomial basis.
    # - r,s,E,F,C,B require the basis; B also requires JP.
    basis: Optional[Basis] = None
    JP: Optional[List[sp.Poly]] = None

    if args.r or args.s or args.E or args.F or args.C or args.B:
        basis = build_shifted_legendre_basis(N, z)

    # Vectors / matrices
    # Crash if somehow the polynomial basis computation was skipped
    if args.r or args.s:
        assert basis is not None
        r, s = compute_r_s(basis)
        if args.r:
            pprint_vector(r, "r", unicode=use_unicode)
        if args.s:
            pprint_vector(s, "s", unicode=use_unicode)

    if args.E or args.F or args.C:
        assert basis is not None
        E, F, C = compute_E_F_C(basis)
        if args.E:
            pprint_matrix(E, "E", unicode=use_unicode)
        if args.F:
            pprint_matrix(F, "F", unicode=use_unicode)
        if args.C:
            pprint_matrix(C, "C", unicode=use_unicode)

    # A tensor
    if args.A:
        A = compute_A(N)
        pprint_tensor_slices(A, "A", unicode=use_unicode)

    # B tensor
    # Crash if somehow the polynomial basis computation was skipped
    if args.B:
        assert basis is not None
        JP = compute_JP(basis)
        B = compute_B(basis, JP=JP)
        pprint_tensor_slices(B, "B", unicode=use_unicode)


if __name__ == "__main__":
    main()
