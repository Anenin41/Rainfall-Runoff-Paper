"""The implementation against the written equations of its references.

The regression suites pin the code to values captured from the legacy code. They
cannot tell whether the legacy code implemented the right equations. These tests
compare the code, term by term, with formulas typed in directly from

* the thesis: eqs (3.34) (coefficient definitions), (3.50)-(3.54) (the explicit
  N = 2 system), (4.12) (constant infiltration), and
* Koellermeier and Rominger, "Analysis and numerical simulation of hyperbolic
  shallow water moment equations": Theorem 3.1 (the HSWME matrix) and
  Theorem 3.3 (its eigenvalues).

Two printed formulas of the thesis are known to carry a sign typo and are entered
here in their corrected form, with the evidence:

* (3.50), first-moment row of Q: printed `-(1/5)(a2 d(h a1) + a1 d(h a2))`, correct
  `-(1/5)(a2 d(h a1) - a1 d(h a2))`. The corrected form follows from the thesis'
  own definition of B_ijk in (3.34) (B_112 = 1/5, B_121 = -1/5, checked below by
  independent quadrature) and agrees with the 3/5 alpha_1 entry of the
  Koellermeier-Rominger matrix.
* (C.12), infiltration bracket: printed `- sum_j F_ij a_j`, correct `+ sum_j F_ij
  a_j`, as in (3.30) and as required to reproduce (3.40) and (3.51).
"""

import numpy as np
import pytest
from numpy.polynomial import legendre

from recharge import source_terms as recharge_source_terms
from recharge.laws import ConstantInfiltration
from swme import coefficients, pde

G = 1.0


def _random_state(rng, order):
    h = rng.uniform(0.5, 2.0)
    return h * np.concatenate([[1.0, rng.normal()], rng.normal(size=order)])


# ---------------------------------------------------------------------------
# thesis (3.34): the coefficient tensors, by an independent quadrature
# ---------------------------------------------------------------------------

class TestCoefficientDefinitions:
    """`coefficients.get_coefficients` computes A, B, C exactly with sympy and
    r, s, E, F from closed forms. Here every one of them is recomputed from its
    integral definition in thesis eq. (3.34) with 40-point Gauss-Legendre
    quadrature in numpy, which shares no code with either route."""

    ORDER = 5

    @staticmethod
    @pytest.fixture(scope='module')
    def reference():
        n = TestCoefficientDefinitions.ORDER
        x, w = legendre.leggauss(40)
        z, w = (x + 1.0) / 2.0, w / 2.0

        def phi(i, zeta):
            return legendre.legval(1.0 - 2.0 * zeta, [0.0] * i + [1.0])

        def dphi(i, zeta):
            return -2.0 * legendre.legval(
                1.0 - 2.0 * zeta, legendre.legder([0.0] * i + [1.0]))

        def primitive(j):
            # int_0^z phi_j, on the quadrature nodes
            return np.array([np.sum(w * zq * phi(j, zq * z)) for zq in z])

        primitives = [primitive(j) for j in range(n + 1)]
        idx = range(n + 1)
        return {
            'A': np.array([[[(2 * i + 1) * np.sum(w * phi(i, z) * phi(j, z) * phi(k, z))
                             for k in idx] for j in idx] for i in idx]),
            'B': np.array([[[(2 * i + 1) * np.sum(w * dphi(i, z) * primitives[j] * phi(k, z))
                             for k in idx] for j in idx] for i in idx]),
            'C': np.array([[np.sum(w * dphi(i, z) * dphi(j, z)) for j in idx] for i in idx]),
            'E': np.array([[np.sum(w * z * dphi(i, z) * phi(j, z)) for j in idx] for i in idx]),
            'F': np.array([[np.sum(w * (z - 1.0) * dphi(i, z) * phi(j, z))
                            for j in idx] for i in idx]),
            'r': np.array([np.sum(w * z * dphi(i, z)) for i in idx]),
            's': np.array([np.sum(w * (z - 1.0) * dphi(i, z)) for i in idx]),
        }

    @pytest.mark.parametrize('name', ['A', 'B', 'C', 'E', 'F', 'r', 's'])
    def test_tensor_matches_its_integral_definition(self, reference, name):
        computed = getattr(coefficients.get_coefficients(self.ORDER), name)
        np.testing.assert_allclose(computed, reference[name], atol=1e-10)

    def test_the_b_entries_behind_the_corrected_sign_of_eq_3_50(self):
        c = coefficients.get_coefficients(2)
        assert c.B[1, 1, 2] == pytest.approx(1.0 / 5.0)
        assert c.B[1, 2, 1] == pytest.approx(-1.0 / 5.0)


# ---------------------------------------------------------------------------
# thesis (3.50)-(3.54): the explicit N = 2 system
# ---------------------------------------------------------------------------

def _flux_n2(U):
    """F(U) of thesis eq. (3.50)."""
    h, q, m1, m2 = U
    u, a1, a2 = q / h, m1 / h, m2 / h
    return np.array([
        q,
        h * u * u + h * a1 * a1 / 3.0 + h * a2 * a2 / 5.0 + G * h * h / 2.0,
        2.0 * h * u * a1 + 4.0 / 5.0 * h * a1 * a2,
        2.0 * h * u * a2 + 2.0 / 3.0 * h * a1 * a1 + 2.0 / 7.0 * h * a2 * a2,
    ])


def _nonconservative_matrix_n2(U, sign_a1_dm2):
    """Q(U, dU/dx) = Q_matrix(U) dU/dx of thesis eq. (3.50).

    `sign_a1_dm2 = +1` is the corrected form, `-1` the form as printed.
    """
    h, q, m1, m2 = U
    u, a1, a2 = q / h, m1 / h, m2 / h
    Q = np.zeros((4, 4))
    Q[2, 2] = u - a2 / 5.0
    Q[2, 3] = sign_a1_dm2 * a1 / 5.0
    Q[3, 2] = a1
    Q[3, 3] = u + a2 / 7.0
    return Q


def _jacobian(function, U):
    """Complex-step Jacobian, exact to round-off for these rational fluxes."""
    step = 1e-30
    return np.array([np.imag(function(U + 1j * step * e)) / step
                     for e in np.eye(len(U))]).T


class TestSecondOrderSystem:

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(20261116)

    def test_system_matrix_is_dF_dU_minus_Q(self, rng):
        model = pde.SWME1D('unused', 0.0, 1.0, False, False)
        for _ in range(20):
            U = _random_state(rng, 2)
            expected = _jacobian(_flux_n2, U) - _nonconservative_matrix_n2(U, +1)
            np.testing.assert_allclose(
                model.compute_system_matrix(2, U), expected, atol=1e-12)

    def test_the_printed_sign_of_eq_3_50_does_not_match(self, rng):
        """Negative control for the correction above: with the sign as printed
        the matrices differ by exactly (2/5) a_1 in one entry."""
        model = pde.SWME1D('unused', 0.0, 1.0, False, False)
        U = _random_state(rng, 2)
        printed = _jacobian(_flux_n2, U) - _nonconservative_matrix_n2(U, -1)
        difference = model.compute_system_matrix(2, U) - printed
        a1 = U[2] / U[0]
        assert difference[2, 3] == pytest.approx(-2.0 / 5.0 * a1)
        difference[2, 3] = 0.0
        np.testing.assert_allclose(difference, 0.0, atol=1e-12)

    def test_mass_source_is_eq_3_51(self, rng):
        for _ in range(20):
            U = _random_state(rng, 2)
            u, a1, a2 = U[1:] / U[0]
            R, I = rng.uniform(0.0, 1.0), rng.normal()
            expected = np.array([
                R - I,
                (R - I) * u - (R + I) * a1 + (R - I) * a2,
                2.0 * (R - I) * a1 - 3.0 * (R + I) * a2,
                3.0 * (R - I) * a2,
            ])
            np.testing.assert_allclose(
                recharge_source_terms.compute_recharge_mass_source(2, U, R, I),
                expected, atol=1e-12)

    def test_friction_is_eqs_3_53_and_3_54(self, rng):
        for _ in range(20):
            U = _random_state(rng, 2)
            h = U[0]
            u, a1, a2 = U[1:] / h
            f_R, f_I, nu = rng.uniform(size=3)
            lam = rng.uniform(0.5, 2.0)
            u_b, u_s = u + a1 + a2, u - a1 + a2
            p_std = np.array([
                0.0,
                nu / lam * u_b,
                3.0 * nu / lam * u_b + 12.0 * nu / h * a1,
                5.0 * nu / lam * u_b + 60.0 * nu / h * a2,
            ])
            p_mix = (f_R * u_s * np.array([0.0, 1.0, -3.0, 5.0])
                     + f_I * u_b * np.array([0.0, 1.0, 3.0, 5.0]))
            np.testing.assert_allclose(
                recharge_source_terms.compute_total_friction(
                    2, U, f_R, f_I, nu, lam),
                p_std + p_mix, atol=1e-12)


# ---------------------------------------------------------------------------
# limiting case: no recharge
# ---------------------------------------------------------------------------

class TestRechargeReducesToTheBaseModel:
    """R = I = f_R = f_I = 0 must give back SWME exactly (thesis eqs (3.43),
    (3.55)): the recharge source vanishes identically and the total friction
    is the Navier-slip friction alone."""

    @pytest.mark.parametrize('order', [0, 1, 2, 3, 5])
    def test_source_terms_reduce(self, order):
        from swme import source_terms
        rng = np.random.default_rng(order)
        U = _random_state(rng, order)
        assert np.all(recharge_source_terms.compute_recharge_mass_source(
            order, U, 0.0, 0.0) == 0.0)
        np.testing.assert_array_equal(
            recharge_source_terms.compute_total_friction(
                order, U, 0.0, 0.0, 0.3, 0.7),
            source_terms.compute_navier_slip_friction(order, U, 0.3, 0.7))

    @pytest.mark.parametrize('order', [0, 1, 2])
    def test_a_run_without_recharge_equals_the_swme_run_bit_for_bit(self, order):
        from recharge.laws import AdmissibleMixingFriction
        from recharge.recharge_pde import RechargeSWME1D
        from swme import mesh, simulation, spatialDiscretization, timeIntegration

        def run(model):
            sim = simulation.ClassicalSimulation1D(
                order, model, mesh.UniformRectangularMesh1D([-1.0, 1.0], 40),
                'PERIODIC', 'smooth_wave', spatialDiscretization.Roe(),
                timeIntegration.ExplicitEuler())
            return sim.run_simulation(0.05)

        base = run(pde.SWME1D('smooth_wave', 1e-2, 1.0, True, False))
        recharge = run(RechargeSWME1D(
            'smooth_wave', 1e-2, 1.0, True, False, 0.0,
            ConstantInfiltration(I0=0.0),
            AdmissibleMixingFriction(alpha_R=1.0, alpha_I=1.0)))
        np.testing.assert_array_equal(recharge, base)


# ---------------------------------------------------------------------------
# Koellermeier-Rominger Theorems 3.1 and 3.3: HSWME
# ---------------------------------------------------------------------------

def _hswme_matrix(order, h, u, a1):
    """A_H of Koellermeier-Rominger eq. (3.1)."""
    n = order + 2
    A = np.zeros((n, n))
    A[0, 1] = 1.0
    A[1, 0] = G * h - u * u - a1 * a1 / 3.0
    A[1, 1] = 2.0 * u
    A[1, 2] = 2.0 * a1 / 3.0
    A[2, 0] = -2.0 * u * a1
    A[2, 1] = 2.0 * a1
    if order >= 2:
        A[3, 0] = -2.0 / 3.0 * a1 * a1
    for i in range(1, order + 1):          # row of alpha_i is row i + 1
        A[i + 1, i + 1] = u
        if i < order:
            A[i + 1, i + 2] = (i + 2.0) / (2.0 * i + 3.0) * a1
        if i > 1:
            A[i + 1, i] = (i - 1.0) / (2.0 * i - 1.0) * a1
    return A


class TestHyperbolicRegularisation:

    @pytest.mark.parametrize('order', [1, 2, 3, 4, 5, 6])
    def test_matrix_is_theorem_3_1(self, order):
        rng = np.random.default_rng(order)
        model = pde.SWME1D('unused', 0.0, 1.0, True, False)
        for _ in range(5):
            U = _random_state(rng, order)
            h, u, a1 = U[0], U[1] / U[0], U[2] / U[0]
            np.testing.assert_allclose(
                model.compute_system_matrix(order, U),
                _hswme_matrix(order, h, u, a1), atol=1e-12)

    @pytest.mark.parametrize('order', [1, 2, 3, 4, 5, 6])
    def test_spectrum_is_theorem_3_3(self, order):
        """Real spectrum whose extreme values are u_m +- sqrt(g h + a_1^2)."""
        rng = np.random.default_rng(100 + order)
        model = pde.SWME1D('unused', 0.0, 1.0, True, False)
        for _ in range(5):
            U = _random_state(rng, order)
            h, u, a1 = U[0], U[1] / U[0], U[2] / U[0]
            eigenvalues = np.linalg.eigvals(model.compute_system_matrix(order, U))
            assert np.max(np.abs(eigenvalues.imag)) < 1e-10
            speed = np.sqrt(G * h + a1 * a1)
            assert eigenvalues.real.max() == pytest.approx(u + speed)
            assert eigenvalues.real.min() == pytest.approx(u - speed)

    @pytest.mark.parametrize('order', [1, 2, 3, 5])
    def test_the_cfl_wave_speed_bounds_the_hswme_spectrum(self, order):
        """`compute_max_wavespeed` uses u_m +- sqrt(g h + sum_i a_i^2), which
        is the exact HSWME bound only when a_2..a_N vanish and otherwise
        over-estimates it. It must never under-estimate."""
        rng = np.random.default_rng(200 + order)
        model = pde.SWME1D('unused', 0.0, 1.0, True, False)
        for _ in range(20):
            U = _random_state(rng, order)
            eigenvalues = np.linalg.eigvals(model.compute_system_matrix(order, U))
            bound = model.compute_max_wavespeed(order, U[None, :])
            assert bound >= np.max(np.abs(eigenvalues)) - 1e-12


class TestWaveSpeedHonoursGravity:

    @pytest.mark.parametrize('h', [1.0, 1e-5])     # wet path, wet-dry path
    def test_non_integer_gravity_is_not_truncated(self, h):
        """The shallow-water speed is sqrt(g h). The code used `int(g)`, which
        turned g = 9.81 into 9."""
        model = pde.SWME1D('unused', 0.0, 1.0, False, False)
        g = 9.81
        speed = model.compute_max_wavespeed(0, np.array([[h, 0.0]]), g=g)
        assert speed == pytest.approx(np.sqrt(g * h))


# ---------------------------------------------------------------------------
# thesis (4.12): constant infiltration
# ---------------------------------------------------------------------------

class TestConstantInfiltrationAvailability:
    """Thesis eq. (4.12): for I0 > 0 the realised rate is
    `min(I0, R + h/dt)`, the water in the cell plus the rain of the step."""

    def test_rain_on_a_dry_cell_can_infiltrate(self):
        law = ConstantInfiltration(I0=0.1)
        assert law.rate(t=0.0, rainfall=0.3, h=0.0, dt=1e-3) == pytest.approx(0.1)

    def test_the_bound_is_rain_plus_standing_water(self):
        law = ConstantInfiltration(I0=1.0)
        assert law.rate(t=0.0, rainfall=0.3, h=2e-4, dt=1e-3) == pytest.approx(0.5)

    def test_an_explicit_step_cannot_go_negative(self):
        law = ConstantInfiltration(I0=5.0)
        h, R, dt = 3e-4, 0.2, 1e-3
        assert h + dt * (R - law.rate(t=0.0, rainfall=R, h=h, dt=dt)) >= 0.0

    def test_no_water_and_no_rain_gives_no_infiltration(self):
        law = ConstantInfiltration(I0=1.0)
        assert law.rate(t=0.0, rainfall=0.0, h=0.0, dt=1e-3) == 0.0

    def test_the_rainfall_limited_variant_is_unchanged(self):
        law = ConstantInfiltration(I0=1.0, limit_by_rainfall=True)
        assert law.rate(t=0.0, rainfall=0.3, h=2e-4, dt=1e-3) == pytest.approx(0.2)

    def test_exfiltration_is_never_limited(self):
        law = ConstantInfiltration(I0=-0.4)
        assert law.rate(t=0.0, rainfall=0.0, h=0.0, dt=1e-3) == pytest.approx(-0.4)
