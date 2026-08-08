"""Rainfall/infiltration mass source and mixing friction, generic in the
moment order N.

These are the terms the recharge extension *adds* to the base SWME/HSWME
model (thesis eq. 3.30 and the f_R/f_I part of eq. 3.33). The base model's own
Navier-slip bed friction is not here - it lives in `swme.source_terms`, since
it exists with or without recharge. `compute_total_friction` combines the two
into the full P(U) = P_slip(U) + P_mix(U) that the solver subtracts.

Replaces the former hand-written per-order functions
(`compute_recharge_source_n0/n1/n2`, `compute_friction_matrix_n0/n1/n2`) and
their dispatchers, which capped the extension at N = 0, 1, 2. Behavior on
those orders is pinned by tests/test_recharge_regression.py against reference
values captured from that deleted code (RESTRUCTURE_PLAN.md Step 2/3).
"""

# Packages
import numpy as np

from swme import coefficients
from swme.source_terms import reconstruct_boundary_velocities, compute_navier_slip_friction


def compute_recharge_mass_source(
    order: int, values: np.ndarray, R: float, I: float, eps_div: float = 1e-14
) -> np.ndarray:
    """S_{R,I}(U): direct rainfall/infiltration mass and momentum production
    (thesis eq. 3.30), arbitrary N. R = I = 0 reproduces a zero source.

        S[0] = R - I
        S[1] = R*u_s - I*u_b
        S[i+2] = (2i+1)*R*(phi_i(1)*u_s - u_m*r_i - sum_j E[i,j]*alpha_j)
                 + (2i+1)*I*(-phi_i(0)*u_b + u_m*s_i + sum_j F[i,j]*alpha_j)
    """
    values = np.asarray(values, dtype=np.float64)
    h = values[0]
    if h <= eps_div:
        return np.zeros(order + 2, dtype=np.float64)

    _, um, alpha, u_s, u_b = reconstruct_boundary_velocities(order, values, eps_div)
    n = order + 2
    S = np.zeros(n, dtype=np.float64)

    S[0] = R - I
    S[1] = R * u_s - I * u_b

    if order >= 1:
        c = coefficients.get_coefficients(order)
        rain = c.phi1_m * u_s - um * c.r_m - c.E_m @ alpha
        infil = -c.phi0_m * u_b + um * c.s_m + c.F_m @ alpha
        S[2:] = c.two_i_plus_1 * (R * rain + I * infil)

    return S


def compute_mixing_friction(
    order: int, values: np.ndarray, f_R: float, f_I: float, eps_div: float = 1e-14
) -> np.ndarray:
    """P_mix(U): rainfall/infiltration-induced mixing friction only (thesis
    eq. 3.33, excluding the Navier-slip term - see module docstring),
    arbitrary N. f_R = f_I = 0 reproduces a zero contribution.

        P_mix[0] = 0
        P_mix[1] = f_R*u_s + f_I*u_b
        P_mix[i+2] = (2i+1)*(f_R*phi_i(1)*u_s + f_I*phi_i(0)*u_b),   i = 1..N
    """
    values = np.asarray(values, dtype=np.float64)
    h = values[0]
    if h <= eps_div:
        return np.zeros(order + 2, dtype=np.float64)

    _, um, alpha, u_s, u_b = reconstruct_boundary_velocities(order, values, eps_div)
    n = order + 2
    P = np.zeros(n, dtype=np.float64)

    P[1] = f_R * u_s + f_I * u_b

    if order >= 1:
        c = coefficients.get_coefficients(order)
        P[2:] = c.two_i_plus_1 * (
            (f_R * u_s) * c.phi1_m + (f_I * u_b) * c.phi0_m
        )

    return P


def compute_total_friction(
    order: int,
    values: np.ndarray,
    f_R: float,
    f_I: float,
    viscosity: float,
    slip_length: float,
    eps_div: float = 1e-14,
) -> np.ndarray:
    """P(U) = P_slip(U) + P_mix(U), the full generalized friction block used
    by RechargeSWME1D (direct generic analog of the old
    `compute_friction_matrix_n0/n1/n2`, which returned this same combined
    quantity). The solver uses S_total(U) = S_{R,I}(U) - P(U).
    """
    return compute_navier_slip_friction(
        order, values, viscosity, slip_length, eps_div
    ) + compute_mixing_friction(order, values, f_R, f_I, eps_div)