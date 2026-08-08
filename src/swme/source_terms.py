"""Generic-N base (Navier-slip) friction, and the boundary-velocity
reconstruction shared by the base SWME/HSWME model and the recharge extension.

Replaces the per-order hardcoded friction blocks in `SWME1D.compute_source_term`
and the ~1700-line `_compute_source_matrix_inverse` with the general formulas of
thesis Appendix C (the `f_R = f_I = 0` case of the full generalized friction
P(U); rainfall/infiltration-induced mixing friction is NOT part of this module
- that lives in `recharge/source_terms.py`, since it only exists when the
recharge extension is active. This module holds exactly what the base
SWME/HSWME transport-and-friction model needs on its own).

`reconstruct_boundary_velocities` is exposed publicly (not `swme`-private)
because `recharge/source_terms.py` needs the identical u_s/u_b reconstruction
for its own mixing-friction and mass-source terms - it is a kinematic fact
about the moment representation (how to read off the surface/bed velocity from
the moment coefficients), not friction physics, so sharing it does not blur the
swme/recharge boundary: the *transport* representation is shared, only the
extra *source* terms are recharge-specific.

NOT YET WIRED INTO `pde.py`: staged here so `tests/test_pde_regression.py` can
validate `compute_navier_slip_friction` and `compute_friction_operator_matrix`
against the still-present hardcoded `SWME1D.compute_source_term` /
`SWME1D._compute_source_matrix_inverse` (Step 2) before Step 3 wires this
module in and deletes the hardcoded blocks.
"""

from __future__ import annotations

import numpy as np

from . import coefficients


def reconstruct_boundary_velocities(order: int, values: np.ndarray, eps_div: float = 1e-12):
    """Extract (h, u_m, alpha, u_s, u_b) from a conserved state U.

    u_s, u_b are the reconstructed free-surface / wet-boundary horizontal
    velocities (thesis eq. 3.34): u_s = u_m + sum_i alpha_i*phi_i(1),
    u_b = u_m + sum_i alpha_i*phi_i(0).
    """
    values = np.asarray(values, dtype=np.float64)
    h = values[0]
    h_reg = h if h > eps_div else eps_div
    um = values[1] / h_reg

    if order == 0:
        alpha = np.zeros(0, dtype=np.float64)
        return h, um, alpha, um, um

    alpha = values[2:] / h_reg
    c = coefficients.get_coefficients(order)
    phi1 = c.phi_at_1[1:]  # phi_i(1), i=1..N
    phi0 = c.phi_at_0[1:]  # phi_i(0), i=1..N
    u_s = um + np.dot(alpha, phi1)
    u_b = um + np.dot(alpha, phi0)
    return h, um, alpha, u_s, u_b


def compute_navier_slip_friction(
    order: int,
    values: np.ndarray,
    viscosity: float,
    slip_length: float,
    eps_div: float = 1e-12,
) -> np.ndarray:
    """P_slip(U): the classical Navier-slip bed-friction contribution to the
    generalized friction term (thesis eq. 3.32-3.33 with f_R=f_I=0):

        P_slip[0] = 0
        P_slip[1] = (nu/lambda)*u_b
        P_slip[i+2] = (2i+1)*(nu/lambda)*phi_i(0)*u_b
                      + (2i+1)*(nu/h^2)*sum_j C[i,j]*U[j+2],   i = 1..N

    The RHS source contribution is `S_{R,I}(U) - P(U)` (thesis eq. 3.38); for
    plain SWME/HSWME (no recharge), P(U) = P_slip(U), so
    `SWME1D.compute_source_term` returns `-compute_navier_slip_friction(...)`.
    """
    values = np.asarray(values, dtype=np.float64)
    h, um, alpha, u_s, u_b = reconstruct_boundary_velocities(order, values, eps_div)
    n = order + 2
    P = np.zeros(n, dtype=np.float64)

    slip_term = viscosity / slip_length
    P[1] = slip_term * u_b

    if order >= 1:
        h_reg = h if h > eps_div else eps_div
        c = coefficients.get_coefficients(order)
        i_idx = np.arange(1, order + 1)
        two_i_plus_1 = 2 * i_idx + 1
        phi0 = c.phi_at_0[1:]
        C_block = c.C[1:, 1:]

        mixing = slip_term * phi0 * u_b
        visc = (viscosity / h_reg**2) * (C_block @ values[2:])
        P[2:] = two_i_plus_1 * mixing + two_i_plus_1 * visc

    return P


def compute_friction_operator_matrix(
    order: int, h: float, viscosity: float, slip_length: float
) -> np.ndarray:
    """S(h): the linear operator such that d(w)/dt = S(h) @ w under pure
    Navier-slip friction, treating h as frozen at the given value (valid since
    dh/dt = 0 under friction alone) - the same reduced linear ODE that
    `SWME1D._compute_source_matrix_inverse` solves in closed form per
    hardcoded order via `(I - dt*S(h))^{-1}`. Row/column 0 (h) are zero:
    friction does not change h and does not depend on it as a *state* (h
    enters here only as the frozen parameter).
    """
    n = order + 2
    S = np.zeros((n, n), dtype=np.float64)
    slip_term = viscosity / (slip_length * h)

    if order == 0:
        S[1, 1] = -slip_term
        return S

    c = coefficients.get_coefficients(order)
    phi0 = c.phi_at_0[1:]  # phi_i(0), i=1..N
    i_idx = np.arange(1, order + 1)
    two_i_plus_1 = 2 * i_idx + 1
    C_block = c.C[1:, 1:]

    S[1, 1] = -slip_term
    S[1, 2:] = -slip_term * phi0
    S[2:, 1] = -two_i_plus_1 * slip_term * phi0
    S[2:, 2:] = (
        -two_i_plus_1[:, None] * slip_term * np.outer(phi0, phi0)
        - two_i_plus_1[:, None] * (viscosity / h**2) * C_block
    )
    return S
