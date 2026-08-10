"""Hyperbolicity of SWME vs. HSWME.

The whole point of the hyperbolic variant (`hyperbolic=True`, HSWME) is that
the plain SWME transport matrix A(U) is *not* globally hyperbolic: for moment
order N >= 2 its eigenvalues leave the real axis once the moments get large
enough relative to sqrt(g*h), and the initial-value problem stops being well
posed there. These tests pin that contrast down, because it is easy to break
silently - a solver happily computes a |lambda| viscosity from a complex
spectrum and produces plausible-looking garbage.

Established here, and relied on by the rest of the suite:

* N=0 and N=1 SWME are UNCONDITIONALLY hyperbolic. The N=1 spectrum is
  exactly {u_m, u_m +- sqrt(g*h + alpha_1^2)}, real for every state, so at
  those orders HSWME and SWME are the same model and the thesis's N=0/N=1
  cases cannot lose hyperbolicity no matter what the forcing does.
* From N=2 up, SWME does lose it, on a boundary that depends only on
  alpha_i/sqrt(g*h) and not at all on u_m.
* HSWME never loses it, at any order tested.
* Augmenting the state with the bed elevation (Step 5) does not change the
  spectrum's realness - it adds one zero eigenvalue and nothing else. This
  matters because otherwise the topography feature would manufacture false
  hyperbolicity-loss reports.
"""

from __future__ import annotations

import numpy as np
import pytest

from swme import mesh as mesh_module
from swme import spatialDiscretization as sd
from swme import timeIntegration
from swme.pde import SWME1D
from swme.simulation import ClassicalSimulation1D

TOL = 1e-10


def make(hyperbolic):
    return SWME1D('x', 0.0, 1.0, hyperbolic, False)


def spectrum(order, h, u_m, alpha, hyperbolic=False):
    state = np.concatenate(([h], [h * u_m], h * np.asarray(alpha, float)))
    return np.linalg.eigvals(make(hyperbolic).compute_system_matrix(order, state))


def max_abs_imag(order, h, u_m, alpha, hyperbolic=False):
    return float(np.max(np.abs(np.imag(
        spectrum(order, h, u_m, alpha, hyperbolic)))))


# ---------------------------------------------------------------------------
# low orders: unconditionally hyperbolic
# ---------------------------------------------------------------------------

class TestLowOrdersAreAlwaysHyperbolic:

    def test_n0_is_the_shallow_water_spectrum(self):
        for h in (0.05, 1.0, 7.0):
            for u in (-3.0, 0.0, 2.5):
                got = np.sort(np.real(spectrum(0, h, u, [])))
                assert max_abs_imag(0, h, u, []) == 0.0
                np.testing.assert_allclose(
                    got, np.sort([u - np.sqrt(h), u + np.sqrt(h)]))

    @pytest.mark.parametrize("alpha_1", [0.0, 0.3, 1.0, 5.0, 20.0, -37.5])
    @pytest.mark.parametrize("u_m", [0.0, 1.7, -4.2])
    @pytest.mark.parametrize("h", [0.05, 1.0, 6.0])
    def test_n1_spectrum_is_exactly_the_closed_form(self, alpha_1, u_m, h):
        """N=1 SWME eigenvalues are u_m and u_m +- sqrt(g*h + alpha_1^2),
        real for every state -- so N=1 can never lose hyperbolicity."""
        c = np.sqrt(h + alpha_1 ** 2)          # g = 1
        expected = np.sort([u_m - c, u_m, u_m + c])
        got = spectrum(1, h, u_m, [alpha_1])
        assert np.max(np.abs(np.imag(got))) == 0.0
        np.testing.assert_allclose(np.sort(np.real(got)), expected, atol=1e-11)

    def test_n1_survives_absurdly_large_moments(self):
        assert max_abs_imag(1, 1.0, 0.0, [1e3]) == 0.0


# ---------------------------------------------------------------------------
# N >= 2: SWME does lose it
# ---------------------------------------------------------------------------

class TestSwmeLosesHyperbolicity:

    def test_a_concrete_non_hyperbolic_n2_state_exists(self):
        # Inside the unstable region; guards against the suite silently
        # becoming vacuous if the transport matrix is ever changed.
        assert max_abs_imag(2, 1.0, 0.0, [1.5, 1.8]) > 1e-6

    def test_the_unstable_set_is_a_band_not_a_threshold(self):
        """Worth pinning explicitly, because the intuitive mental model
        ('big moments are bad') is wrong and would lead to a wrong safety
        criterion: alpha = (1.5, 1.8) is NON-hyperbolic while the strictly
        larger alpha = (2.0, 3.0) is hyperbolic again."""
        assert max_abs_imag(2, 1.0, 0.0, [1.5, 1.8]) > 1e-6
        assert max_abs_imag(2, 1.0, 0.0, [2.0, 3.0]) <= TOL

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_loss_occurs_at_every_order_from_two_up(self, order):
        rng = np.random.default_rng(11 + order)
        found = any(
            max_abs_imag(order, 1.0, 0.0, rng.uniform(-4, 4, size=order)) > TOL
            for _ in range(2000)
        )
        assert found, f"expected some non-hyperbolic N={order} states"

    def test_loss_needs_more_than_the_first_moment(self):
        """alpha_1 alone never breaks N=2 -- it is the interaction with
        alpha_2 that does. Sanity check on the shape of the boundary."""
        for alpha_1 in np.linspace(0.0, 8.0, 40):
            assert max_abs_imag(2, 1.0, 0.0, [alpha_1, 0.0]) <= TOL

    def test_the_boundary_is_galilean_invariant(self):
        """Hyperbolicity cannot depend on the frame: shifting u_m must leave
        the imaginary parts untouched."""
        reference = max_abs_imag(2, 1.0, 0.0, [1.5, 1.8])
        assert reference > 1e-6
        for u_m in (-11.0, -1.0, 0.5, 25.0):
            assert max_abs_imag(2, 1.0, u_m, [1.5, 1.8]) == pytest.approx(
                reference, rel=1e-9)

    def test_the_boundary_depends_only_on_alpha_over_sqrt_gh(self):
        """Scaling h by c^2 and alpha by c leaves alpha/sqrt(g*h) fixed, so
        the spectrum's imaginary part must scale by exactly c."""
        reference = max_abs_imag(2, 1.0, 0.0, [1.5, 1.8])
        for c in (0.5, 2.0, 7.0):
            scaled = max_abs_imag(2, c * c, 0.0, [c * 1.5, c * 1.8])
            assert scaled == pytest.approx(c * reference, rel=1e-9)


# ---------------------------------------------------------------------------
# HSWME: never
# ---------------------------------------------------------------------------

class TestHswmeIsAlwaysHyperbolic:

    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6])
    def test_random_states_never_leave_the_real_axis(self, order):
        rng = np.random.default_rng(500 + order)
        worst = 0.0
        for _ in range(3000):
            worst = max(worst, max_abs_imag(
                order,
                rng.uniform(0.05, 5.0),
                rng.uniform(-4, 4),
                rng.uniform(-4, 4, size=order),
                hyperbolic=True,
            ))
        assert worst <= TOL, f"HSWME went complex at N={order}: {worst}"

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_hswme_fixes_states_that_break_swme(self, order):
        """The states that matter: ones plain SWME cannot handle."""
        rng = np.random.default_rng(900 + order)
        checked = 0
        for _ in range(4000):
            alpha = rng.uniform(-4, 4, size=order)
            if max_abs_imag(order, 1.0, 0.0, alpha) > TOL:
                checked += 1
                assert max_abs_imag(order, 1.0, 0.0, alpha,
                                    hyperbolic=True) <= TOL
        assert checked > 0, "no broken SWME states sampled; test was vacuous"

    @pytest.mark.parametrize("order", [1, 2, 3, 4])
    def test_hswme_keeps_the_outer_wave_speeds(self, order):
        """The regularization must not destroy the physics it is regularizing:
        the fastest waves stay u_m +- sqrt(g*h + alpha_1^2), as at N=1."""
        h, u_m, alpha_1 = 1.3, 0.6, 0.7
        alpha = [alpha_1] + [1.9, -2.4, 3.1][:order - 1]
        got = np.sort(np.real(spectrum(order, h, u_m, alpha, hyperbolic=True)))
        c = np.sqrt(h + alpha_1 ** 2)
        assert got[0] == pytest.approx(u_m - c, rel=1e-9)
        assert got[-1] == pytest.approx(u_m + c, rel=1e-9)

    @pytest.mark.parametrize("order", [2, 3, 4])
    def test_hswme_spectrum_ignores_the_higher_moments(self, order):
        """HSWME zeroes alpha_2..alpha_N in the transport matrix, so the
        spectrum must not move when they change. This is the mechanism behind
        the guarantee, stated as a test so it cannot regress quietly."""
        base = np.sort(np.real(spectrum(
            order, 1.0, 0.4, [0.8] + [0.0] * (order - 1), hyperbolic=True)))
        rng = np.random.default_rng(77)
        for _ in range(20):
            alpha = [0.8] + list(rng.uniform(-5, 5, size=order - 1))
            other = np.sort(np.real(spectrum(order, 1.0, 0.4, alpha,
                                             hyperbolic=True)))
            np.testing.assert_allclose(other, base, atol=1e-11)


# ---------------------------------------------------------------------------
# topography must not manufacture false reports
# ---------------------------------------------------------------------------

class TestAugmentationDoesNotAffectHyperbolicity:

    @pytest.mark.parametrize("order", [0, 1, 2, 3, 4])
    @pytest.mark.parametrize("hyperbolic", [False, True])
    def test_augmented_spectrum_is_the_plain_one_plus_a_zero(
            self, order, hyperbolic):
        rng = np.random.default_rng(1300 + order)
        pde = make(hyperbolic)
        for _ in range(300):
            h = rng.uniform(0.2, 3.0)
            state = np.concatenate((
                [h], [h * rng.uniform(-2, 2)], h * rng.uniform(-2, 2, size=order)))
            bed = rng.uniform(-2, 2)

            plain = np.sort_complex(np.linalg.eigvals(
                pde.compute_system_matrix(order, state)))
            augmented = np.sort_complex(np.linalg.eigvals(
                pde.compute_augmented_system_matrix(order, np.append(state, bed))))

            # one extra eigenvalue, and it is zero
            assert len(augmented) == len(plain) + 1
            assert np.min(np.abs(augmented)) < 1e-9
            # the imaginary parts - the thing hyperbolicity is judged on -
            # must agree exactly, or topography would fake a loss
            assert np.max(np.abs(np.imag(augmented))) == pytest.approx(
                np.max(np.abs(np.imag(plain))), abs=1e-11)

    def test_augmenting_a_non_hyperbolic_state_does_not_worsen_it(self):
        pde = make(False)
        h, alpha = 1.0, [1.5, 1.8]
        state = np.concatenate(([h], [0.0], h * np.array(alpha)))
        plain = np.max(np.abs(np.imag(np.linalg.eigvals(
            pde.compute_system_matrix(2, state)))))
        augmented = np.max(np.abs(np.imag(np.linalg.eigvals(
            pde.compute_augmented_system_matrix(2, np.append(state, 0.9))))))
        assert plain > 1e-6
        assert augmented == pytest.approx(plain, rel=1e-9)


# ---------------------------------------------------------------------------
# the runtime detector
# ---------------------------------------------------------------------------

class TestRuntimeDetection:

    def test_roe_counts_a_complex_spectrum(self):
        scheme = sd.Roe()
        pde = make(False)
        state = np.array([1.0, 0.0, 1.5, 1.8])       # known non-hyperbolic
        scheme.compute_viscosity(pde.compute_system_matrix(2, state), 0.01, 0.1)
        assert scheme.nonhyperbolic_count == 1
        assert scheme.max_abs_imaginary_eigenvalue > 1e-6

    def test_roe_does_not_cry_wolf_on_a_real_spectrum(self):
        scheme = sd.Roe()
        pde = make(False)
        state = np.array([1.0, 0.5, 0.05, -0.02])    # thesis-scale moments
        scheme.compute_viscosity(pde.compute_system_matrix(2, state), 0.01, 0.1)
        assert scheme.spectra_examined == 1
        assert scheme.nonhyperbolic_count == 0

    def test_the_viscosity_is_real_even_for_a_complex_spectrum(self):
        """A real matrix's complex eigenvalues come in conjugate pairs with
        equal moduli, so R|D|R^-1 is real. The scheme must return a real
        matrix rather than something that gets silently truncated later."""
        scheme = sd.Roe()
        pde = make(False)
        matrix = pde.compute_system_matrix(2, np.array([1.0, 0.0, 1.5, 1.8]))
        viscosity = scheme.compute_viscosity(matrix, 0.01, 0.1)

        assert not np.iscomplexobj(viscosity)
        # Bit-identical to the implicit cast this replaced.
        eigenvalues, R = np.linalg.eig(matrix)
        naive = R @ np.diag(np.abs(eigenvalues)) @ np.linalg.inv(R)
        np.testing.assert_array_equal(viscosity, naive.real)
        # The discarded part really is round-off, not signal.
        assert np.max(np.abs(naive.imag)) < 1e-12

    def test_absolute_value_matrix_squares_back_to_A_squared_when_real(self):
        """|A|^2 = A^2 characterizes |A| -- but ONLY for a real spectrum:
        for lambda = a+bi, |lambda|^2 = a^2+b^2 while lambda^2 = a^2-b^2+2abi.
        So this identity is the right check on the hyperbolic branch and is
        genuinely false on the non-hyperbolic one, which is worth stating
        rather than quietly testing only the easy case."""
        scheme = sd.Roe()
        pde = make(False)

        real_spectrum = pde.compute_system_matrix(
            2, np.array([1.0, 0.5, 0.05, -0.02]))
        viscosity = scheme.compute_viscosity(real_spectrum, 0.01, 0.1)
        np.testing.assert_allclose(viscosity @ viscosity,
                                   real_spectrum @ real_spectrum, atol=1e-10)

        complex_spectrum = pde.compute_system_matrix(
            2, np.array([1.0, 0.0, 1.5, 1.8]))
        viscosity = scheme.compute_viscosity(complex_spectrum, 0.01, 0.1)
        assert not np.allclose(viscosity @ viscosity,
                               complex_spectrum @ complex_spectrum, atol=1e-6)

    def test_counters_reset_between_runs(self):
        scheme = sd.Roe()
        pde = make(False)
        scheme.compute_viscosity(
            pde.compute_system_matrix(2, np.array([1.0, 0.0, 1.5, 1.8])),
            0.01, 0.1)
        assert scheme.nonhyperbolic_count == 1
        scheme.reset_hyperbolicity_diagnostics()
        assert scheme.nonhyperbolic_count == 0
        assert scheme.spectra_examined == 0
        assert scheme.max_abs_imaginary_eigenvalue == 0.0


# ---------------------------------------------------------------------------
# end to end: a run that really does lose it
# ---------------------------------------------------------------------------

class _WedgeIC(SWME1D):
    """Uniform state parked inside the N=2 non-hyperbolic wedge
    (alpha_2/alpha_1 ~ 1.25), with a small ripple so there is some dynamics."""

    def get_initial_values(self, order, initial_condition, position):
        h = 1.0 + 0.01 * np.sin(2 * np.pi * position)
        return np.array([h, 0.0, h * 3.0, h * 3.75])[:order + 2]


def _run(hyperbolic, t_end=0.05):
    pde = _WedgeIC('wedge', 0.0, 1.0, hyperbolic, False)
    mesh = mesh_module.UniformRectangularMesh1D([0.0, 1.0], 20)
    scheme = sd.Roe()
    sim = ClassicalSimulation1D(2, pde, mesh, 'PERIODIC', 'wedge', scheme,
                                timeIntegration.ExplicitEuler())
    return sim, scheme


class TestEndToEndDetection:

    def test_a_non_hyperbolic_swme_run_reports_itself(self):
        sim, scheme = _run(hyperbolic=False)
        with pytest.warns(RuntimeWarning, match="Complex spectra at"):
            sim.run_simulation(0.05)
        assert scheme.nonhyperbolic_count > 0
        assert scheme.max_abs_imaginary_eigenvalue > 1e-6

    def test_the_hswme_variant_of_the_same_run_stays_clean(self):
        """Same initial state, same grid, same scheme -- only the closure
        differs. This is the whole justification for HSWME existing."""
        sim, scheme = _run(hyperbolic=True)
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim.run_simulation(0.05)
        assert not [w for w in caught
                    if "Complex spectra" in str(w.message)]
        assert scheme.nonhyperbolic_count == 0
        assert scheme.spectra_examined > 0
        assert scheme.max_abs_imaginary_eigenvalue <= TOL

    def test_an_ordinary_hyperbolic_run_does_not_warn(self):
        """Guard against the detector crying wolf on every run, which is what
        the ComplexWarning it replaced did."""
        pde = SWME1D('damBreak_noVelocity', 0.01, 0.05, False, False)
        mesh = mesh_module.UniformRectangularMesh1D([-1.0, 1.0], 20)
        scheme = sd.Roe()
        sim = ClassicalSimulation1D(2, pde, mesh, 'INFLOW_OUTFLOW',
                                    'damBreak_noVelocity', scheme,
                                    timeIntegration.ExplicitEuler())
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim.run_simulation(0.2)
        assert not [w for w in caught
                    if "Complex spectra" in str(w.message)]
        assert scheme.spectra_examined > 0
        assert scheme.nonhyperbolic_count == 0
