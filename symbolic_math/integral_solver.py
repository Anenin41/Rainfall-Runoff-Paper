# Sympy Script for computing Matrices & Tensors of the SWME + Recharge Model #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Tue 02 Mar 2026 @ 12:12:05 +0100
# Modified: Wed 25 Mar 2026 @ 21:42:26 +0100

# Packages
import sympy as sp
from pathlib import Path
from datetime import datetime

# Order
N = 5

# Variable
z = sp.Symbol('z', real=True)

# Building the shifted Legendre Basis Functions
# Wikipedia: phi_n(z) = P_n(2z - 1), relation between shifted and non-shifted pols
def phi(n):
    return sp.legendre(n, 1 - 2*z)

# Initialize polynomials and their derivatives
phis = [sp.expand(phi(i)) for i in range(N + 1)]
dphis = [sp.diff(phis[i], z) for i in range(N + 1)]

# Helper operator that defines the integral of f(z) over z in [0, 1]
# Mathematically: I[f] = int_{0}^{1} f(z) dz
def I(expr):
    return sp.simplify(sp.integrate(sp.expand(expr), (z, 0, 1)))

# Memory allocation for the containers (omit iterables)
# Tensors:
A = [[[None] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]
B = [[[None] * (N + 1) for _ in range(N + 1)] for __ in range(N + 1)]

# Matrices
E = [[None] * (N + 1) for _ in range(N + 1)]
F = [[None] * (N + 1) for _ in range(N + 1)]
C = [[None] * (N + 1) for _ in range(N + 1)]

# Vectors:
r = [None] * (N + 1)
s = [None] * (N + 1)

# Compute vectors & matrices (same nested loop)
# The math formulas can be found in the report/presentation
for i in range(N + 1):
    r[i] = I( z * dphis[i] )
    s[i] = I( (z - 1) * dphis[i] )
    for j in range(N + 1):
        E[i][j] = I( z * (dphis[i]) * phis[j] )
        F[i][j] = I( (z - 1) * (dphis[i]) * phis[j])
        C[i][j] = I( (dphis[i]) * (dphis[j]) )

# Compute the tensor A
for i in range(N + 1):
    for j in range(N + 1):
        for k in range(N + 1):
            A[i][j][k] = sp.simplify( (2*i + 1) * I(phis[i] * phis[j] * phis[k]) )

# Inside B there exists another integral:
# int_{0}^{z} phi_j (zhat) dzhat
# Define this as an auxiliary operator as well
J = [sp.integrate(phis[j].subs(z, sp.Symbol('zhat')), (sp.Symbol('zhat'), 0, z))
     for j in range(N + 1)]

# Compute the tensor B
for i in range(N + 1):
    for j in range(N + 1):
        for k in range(N + 1):
            B[i][j][k] = sp.simplify( (2*i + 1) * I(dphis[i] * J[j] * phis[k]) )


# Functions that control how the results are printed out
def _pretty(obj) -> str:
    """Pretty-print SymPy objects as ASCII (good for text files)."""
    return sp.pretty(obj, use_unicode=False)

def _as_matrix(M):
    return sp.Matrix(M)

def _as_colvec(v):
    return sp.Matrix(v)

def _count_nonzero_matrix(M):
    return sum(1 for row in M for x in row if x != 0)

def _count_nonzero_tensor(T):
    n = len(T)
    return sum(1 for i in range(n) for j in range(n) for k in range(n) if T[i][j][k] != 0)

def dump_all_txt(
        path: str | Path,
        N: int,
        r, s, E, F, C, A, B,
        include_metadata: bool = True):
    path = Path(path)
    n = N + 1

    with path.open("w", encoding="utf-8") as f:
        if include_metadata:
            f.write("Symbolic tensor/matrix dump\n")
            f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n")
            f.write(f"N = {N}  (size = {n})\n")
            f.write("=" * 80 + "\n\n")

        # Vectors
        f.write(f"=== r (len={len(r)}) ===\n")
        f.write(_pretty(_as_colvec(r)) + "\n\n")

        f.write(f"=== s (len={len(s)}) ===\n")
        f.write(_pretty(_as_colvec(s)) + "\n\n")

        # Matrices
        for name, M in [("E", E), ("F", F), ("C", C)]:
            nz = _count_nonzero_matrix(M)
            f.write(f"=== {name} ({n}x{n})  nonzeros={nz}/{n*n} ===\n")
            f.write(_pretty(_as_matrix(M)) + "\n\n")

        # 3-tensors: print as slices T[i,:,:]
        for name, T in [("A", A), ("B", B)]:
            nz = _count_nonzero_tensor(T)
            f.write(f"=== {name} ({n}x{n}x{n})  nonzeros={nz}/{n*n*n} ===\n\n")
            for i in range(n):
                f.write(f"-- {name}[{i},:,:] slice --\n")
                f.write(_pretty(_as_matrix(T[i])) + "\n\n")

    return path

# -----------------------------
# Use it in your __main__ block
# -----------------------------
if __name__ == "__main__":
    out = dump_all_txt("tensors_dump.txt", N, r, s, E, F, C, A, B)
    print(f"Wrote tensor/matrix dump to: {out.resolve()}")
