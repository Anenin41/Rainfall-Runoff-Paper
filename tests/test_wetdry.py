"""Wet-dry treatment (RESTRUCTURE_PLAN.md Step 6).

Before this step the solver raised `RuntimeError` the instant any cell height
reached zero, so a dam break onto a dry bed - the single most standard
shallow-water benchmark there is - could not be run at all. Three layers make
it possible: a two-tier threshold convention, desingularized primitive
reconstruction with a moment ramp, and a positivity-preserving timestep
limiter.

The tests are grouped by what they protect:

1. The threshold object and the primitive-extraction rule, including the
   property everything else leans on - that a fully wet cell goes through
   *exactly* the pre-Step-6 arithmetic.
2. Every place that divides by h: transport matrix, wave speeds, friction,
   the implicit friction operator, the recharge sources.
3. The mandatory end-to-end regression: a dam break onto a dry bed, checked
   for positivity, mass conservation and against the exact Ritter solution.
4. The physics that would be silently wrong if the dry-cell handling were
   naive - above all, that rain can still wet dry ground.
"""

from __future__ import annotations

import numpy as np
import pytest

from swme import mesh as mesh_module
from swme import source_terms
from swme import spatialDiscretization as sd
from swme import timeIntegration
from swme import wetdry
from swme.pde import SWME1D
from swme.simulation import ClassicalSimulation1D
from swme.wetdry import (
    WetDryThresholds,
    desingularized_primitives,
    desingularized_primitives_array,
)

DEFAULTS = WetDryThresholds()


def run(pde, order, initial_condition, resolution=200, boundaries=(-1.0, 1.0),
        t_end=0.2, boundary='INFLOW_OUTFLOW', scheme=None):
    mesh = mesh_module.UniformRectangularMesh1D(list(boundaries), resolution)
    simulation = ClassicalSimulation1D(
        order, pde, mesh, boundary, initial_condition,
        scheme if scheme is not None else sd.Roe(),
        timeIntegration.ExplicitEuler(),
    )
    return simulation.run_simulation(t_end), mesh, simulation


# ---------------------------------------------------------------------------
# 1. thresholds and the primitive-extraction rule
# ---------------------------------------------------------------------------

class TestThresholds:

    def test_defaults_are_ordered(self):
        assert 0 < DEFAULTS.eps_div <= DEFAULTS.h_dry < DEFAULTS.h_wet

    @pytest.mark.parametrize("kwargs,match", [
        (dict(eps_div=0.0), "eps_div"),
        (dict(eps_div=-1e-14), "eps_div"),
        (dict(h_dry=1e-16), "h_dry"),          # below eps_div
        (dict(h_wet=1e-5), "h_wet"),           # below h_dry
        (dict(h_dry=1e-3, h_wet=1e-3), "h_wet"),   # zero-width ramp
    ])
    def test_inconsistent_thresholds_are_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            WetDryThresholds(**kwargs)

    def test_moment_ramp_spans_the_transition_band(self):
        t = DEFAULTS
        assert t.moment_ramp(t.h_dry) == 0.0
        assert t.moment_ramp(t.h_wet) == 1.0
        assert t.moment_ramp(0.5 * (t.h_dry + t.h_wet)) == pytest.approx(0.5)
        assert t.moment_ramp(0.0) == 0.0
        assert t.moment_ramp(1.0) == 1.0

    def test_safe_height_floors_at_h_dry_not_eps_div(self):
        """nu/h^2 at eps_div = 1e-14 would be 1e28 - finite, so no isfinite
        guard catches it, and completely meaningless."""
        assert wetdry.safe_height(1e-20, DEFAULTS) == DEFAULTS.h_dry
        assert wetdry.safe_height(0.5, DEFAULTS) == 0.5


class TestDesingularizedPrimitives:

    def test_a_wet_cell_uses_exactly_the_original_arithmetic(self):
        """The load-bearing property of the whole step: everything above
        h_wet must be bit-for-bit what it was before wet-dry existed, or every
        previously validated result moves."""
        rng = np.random.default_rng(3)
        for _ in range(500):
            h = rng.uniform(DEFAULTS.h_wet, 5.0)
            state = np.concatenate(([h], rng.normal(size=3)))
            _, u_m, alpha = desingularized_primitives(2, state, DEFAULTS)
            assert u_m == state[1] / h
            np.testing.assert_array_equal(alpha, state[2:] / h)

    def test_a_dry_cell_has_no_velocity_and_no_moments(self):
        state = np.array([0.0, 0.0, 0.0, 0.0])
        h, u_m, alpha = desingularized_primitives(2, state, DEFAULTS)
        assert h == 0.0 and u_m == 0.0
        assert np.all(alpha == 0.0)

    def test_velocity_stays_bounded_as_the_cell_empties(self):
        """Plain q/h diverges; the desingularized form must not."""
        speeds = []
        for h in np.geomspace(1e-18, DEFAULTS.h_wet, 200):
            q = h * 1.0                       # a physical u_m of 1
            _, u_m, _ = desingularized_primitives(0, np.array([h, q]), DEFAULTS)
            speeds.append(abs(u_m))
        assert max(speeds) <= 1.0 + 1e-12
        assert speeds[0] < 1e-6              # driven to zero at the empty end

    def test_velocity_is_exactly_q_over_h_down_to_h_dry(self):
        """The Kurganov-Petrova denominator is exactly 2h^2 for h >= h_dry, so
        no accuracy is given up anywhere above the dry threshold."""
        for h in np.geomspace(DEFAULTS.h_dry, 1.0, 50):
            q = 0.37 * h
            _, u_m, _ = desingularized_primitives(0, np.array([h, q]), DEFAULTS)
            assert u_m == pytest.approx(q / h, rel=1e-14)

    def test_moments_ramp_to_zero_across_the_transition_band(self):
        for h, expected in [(DEFAULTS.h_wet, 1.0),
                            (0.5 * (DEFAULTS.h_dry + DEFAULTS.h_wet), 0.5),
                            (DEFAULTS.h_dry, 0.0)]:
            state = np.array([h, 0.0, h * 0.4])
            _, _, alpha = desingularized_primitives(1, state, DEFAULTS)
            assert alpha[0] == pytest.approx(0.4 * expected, rel=1e-9)

    def test_primitives_are_continuous_across_both_thresholds(self):
        """A jump at either threshold would be a spurious wave source right at
        the wetting front."""
        for threshold in (DEFAULTS.h_dry, DEFAULTS.h_wet):
            for eps in (1e-12, 1e-14):
                below = np.array([threshold - eps, (threshold - eps) * 0.5,
                                  (threshold - eps) * 0.3])
                above = np.array([threshold + eps, (threshold + eps) * 0.5,
                                  (threshold + eps) * 0.3])
                _, u_lo, a_lo = desingularized_primitives(1, below, DEFAULTS)
                _, u_hi, a_hi = desingularized_primitives(1, above, DEFAULTS)
                assert u_lo == pytest.approx(u_hi, abs=1e-6)
                assert a_lo[0] == pytest.approx(a_hi[0], abs=1e-6)

    @pytest.mark.parametrize("order", [0, 1, 2, 3])
    def test_array_and_scalar_paths_agree(self, order):
        rng = np.random.default_rng(20 + order)
        heights = np.concatenate((
            [0.0, DEFAULTS.eps_div * 0.5, DEFAULTS.h_dry, DEFAULTS.h_wet],
            np.geomspace(1e-12, 3.0, 40),
        ))
        states = np.column_stack(
            [heights] + [rng.normal(size=len(heights)) for _ in range(order + 1)])
        h_a, u_a, al_a = desingularized_primitives_array(order, states, DEFAULTS)
        for i, state in enumerate(states):
            h_s, u_s, al_s = desingularized_primitives(order, state, DEFAULTS)
            assert h_a[i] == h_s
            assert u_a[i] == pytest.approx(u_s, rel=1e-15, abs=1e-300)
            np.testing.assert_allclose(al_a[i], al_s, rtol=1e-15, atol=1e-300)


# ---------------------------------------------------------------------------
# 2. everywhere that divides by h
# ---------------------------------------------------------------------------

class TestClosuresSurviveDryCells:

    @pytest.mark.parametrize("order", [0, 1, 2, 4])
    def test_system_matrix_is_finite_at_and_near_zero_depth(self, order):
        pde = SWME1D('x', 0.0, 1.0, False, False)
        for h in (0.0, 1e-18, DEFAULTS.eps_div, DEFAULTS.h_dry, DEFAULTS.h_wet):
            state = np.concatenate(([h], np.full(order + 1, h * 0.5)))
            assert np.isfinite(pde.compute_system_matrix(order, state)).all()

    def test_system_matrix_at_zero_depth_is_the_trivial_one(self):
        pde = SWME1D('x', 0.0, 1.0, False, False)
        matrix = pde.compute_system_matrix(1, np.zeros(3))
        expected = np.zeros((3, 3))
        expected[0, 1] = 1.0
        np.testing.assert_array_equal(matrix, expected)

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_wave_speed_is_finite_on_a_partly_dry_grid(self, order):
        pde = SWME1D('x', 0.0, 1.0, False, False)
        grid = np.zeros((5, order + 2))
        grid[0, 0] = 1.0
        grid[0, 1] = 1.0
        grid[1, 0] = DEFAULTS.h_dry * 0.5
        # cells 2..4 exactly dry
        speed = pde.compute_max_wavespeed(order, grid)
        assert np.isfinite(speed) and speed > 0.0

    def test_wave_speed_on_a_completely_dry_grid_is_finite_and_positive(self):
        """The caller divides by this to get delta_t, so it must never be
        zero, NaN or inf even when there is no water at all."""
        pde = SWME1D('x', 0.0, 1.0, False, False)
        speed = pde.compute_max_wavespeed(1, np.zeros((4, 3)))
        assert np.isfinite(speed) and speed > 0.0

    def test_wave_speed_matches_the_old_expression_when_fully_wet(self):
        pde = SWME1D('x', 0.0, 1.0, False, False)
        rng = np.random.default_rng(5)
        grid = np.column_stack([rng.uniform(0.5, 2.0, 30),
                                rng.normal(size=30), rng.normal(size=30)])
        legacy = grid[:, 0] * 1
        legacy = legacy + np.divide(grid[:, 2] * grid[:, 2],
                                    grid[:, 0] * grid[:, 0])
        expected = max(
            np.max(np.abs(np.divide(grid[:, 1], grid[:, 0]) + np.sqrt(legacy))),
            np.max(np.abs(np.divide(grid[:, 1], grid[:, 0]) - np.sqrt(legacy))),
        )
        assert pde.compute_max_wavespeed(1, grid) == expected

    @pytest.mark.parametrize("order", [0, 1, 3])
    def test_friction_vanishes_on_a_dry_cell(self, order):
        state = np.zeros(order + 2)
        friction = source_terms.compute_navier_slip_friction(
            order, state, viscosity=0.1, slip_length=0.05, thresholds=DEFAULTS)
        assert np.all(friction == 0.0)

    def test_friction_stays_bounded_through_the_transition_band(self):
        """nu/h^2 is the sharpest divergence in the model. Without the floor
        and the ramp this grows without bound as the cell empties."""
        magnitudes = []
        for h in np.geomspace(1e-16, 1.0, 300):
            state = np.array([h, h * 0.5, h * 0.2])
            magnitudes.append(np.max(np.abs(
                source_terms.compute_navier_slip_friction(
                    1, state, 0.1, 0.05, DEFAULTS))))
        assert np.isfinite(magnitudes).all()
        # bounded by the value at the wet end, i.e. no blow-up on the way down
        assert max(magnitudes) < 100 * max(magnitudes[-10:])

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_implicit_friction_operator_is_identity_on_a_dry_cell(self, order):
        n = order + 2
        S = source_terms.compute_friction_operator_matrix(
            order, 0.0, viscosity=0.1, slip_length=0.05, thresholds=DEFAULTS)
        np.testing.assert_array_equal(S, np.zeros((n, n)))
        # so the implicit step leaves the empty cell alone
        np.testing.assert_allclose(np.linalg.inv(np.eye(n) - 0.01 * S), np.eye(n))

    def test_implicit_friction_operator_is_finite_near_dry(self):
        for h in np.geomspace(1e-18, 1.0, 100):
            S = source_terms.compute_friction_operator_matrix(
                1, h, 0.1, 0.05, DEFAULTS)
            assert np.isfinite(S).all()
            assert np.isfinite(np.linalg.inv(np.eye(3) - 0.001 * S)).all()


# ---------------------------------------------------------------------------
# 3. the mandatory end-to-end regression
# ---------------------------------------------------------------------------

class TestDamBreakOntoDryBed:
    """RESTRUCTURE_PLAN.md Step 6 names this the required validation. It is
    also the case the solver simply could not run before: any dry cell raised
    RuntimeError on the first step."""

    H0 = 1.0
    T_END = 0.2

    # Tight thresholds, because a vacuum front is exactly the case where the
    # exact solution contains arbitrarily small depths and a coarse h_dry
    # truncates the leading edge - see WetDryThresholds' docstring and
    # test_the_front_speed_degrades_with_a_coarse_h_dry below.
    THRESHOLDS = WetDryThresholds(h_dry=1e-10, h_wet=1e-9)

    @pytest.fixture(scope="class")
    def solved(self):
        pde = SWME1D('damBreak_dryBed', 0.0, 1.0, False, False,
                     wet_dry=self.THRESHOLDS)
        data, mesh, _ = run(pde, 1, 'damBreak_dryBed', resolution=400,
                            t_end=self.T_END)
        return data, mesh

    def test_it_runs_at_all(self, solved):
        data, _ = solved
        assert np.isfinite(data).all()

    def test_height_never_goes_negative(self, solved):
        data, _ = solved
        assert data[:, 1].min() >= 0.0

    def test_mass_is_conserved(self, solved):
        data, mesh = solved
        delta_x = (mesh.boundaries[1] - mesh.boundaries[0]) / mesh.resolution
        # the rarefaction has not reached either boundary by T_END
        initial = np.sum(
            np.where(mesh.cell_center_positions < 0.0, self.H0, 0.0)) * delta_x
        assert np.sum(data[:, 1]) * delta_x == pytest.approx(initial, rel=2e-3)

    def test_the_wet_front_travels_at_about_the_ritter_speed(self, solved):
        """Exact solution for a frictionless dam break onto a dry bed: the
        front advances at 2*sqrt(g*h0), the rarefaction tail at -sqrt(g*h0).

        The tolerance on the front is 10%, not machine precision, and that is
        a real limitation rather than slack: see
        test_the_front_speed_degrades_with_a_coarse_h_dry.
        """
        data, mesh = solved
        c0 = np.sqrt(self.H0)                       # g = 1
        wet = data[:, 1] > 1e-10
        front = data[wet, 0].max()
        tail = data[wet, 0].min()
        delta_x = (mesh.boundaries[1] - mesh.boundaries[0]) / mesh.resolution

        assert front == pytest.approx(2.0 * c0 * self.T_END, rel=0.10)
        assert tail <= -c0 * self.T_END + 15 * delta_x

    def test_the_profile_matches_the_ritter_solution(self, solved):
        """h(x,t) = (2*c0 - x/t)^2 / (9g) inside the rarefaction fan.

        Checked in L1 rather than pointwise: a first-order scheme smears the
        two corners of the fan over a few cells, so the pointwise maximum is
        dominated by those corners and does not converge, while the integral
        error does.
        """
        data, _ = solved
        c0 = np.sqrt(self.H0)
        x, h = data[:, 0], data[:, 1]
        exact = np.where(x < -c0 * self.T_END, self.H0,
                         np.where(x > 2.0 * c0 * self.T_END, 0.0,
                                  (2.0 * c0 - x / self.T_END) ** 2 / 9.0))
        delta_x = x[1] - x[0]
        assert np.sum(np.abs(h - exact)) * delta_x < 0.02

    def test_the_l1_error_converges_under_refinement(self):
        errors = []
        for resolution in (100, 400):
            pde = SWME1D('damBreak_dryBed', 0.0, 1.0, False, False,
                         wet_dry=self.THRESHOLDS)
            data, _, _ = run(pde, 0, 'damBreak_dryBed', resolution=resolution,
                             t_end=self.T_END)
            x, h = data[:, 0], data[:, 1]
            c0 = np.sqrt(self.H0)
            exact = np.where(x < -c0 * self.T_END, self.H0,
                             np.where(x > 2.0 * c0 * self.T_END, 0.0,
                                      (2.0 * c0 - x / self.T_END) ** 2 / 9.0))
            errors.append(np.sum(np.abs(h - exact)) * (x[1] - x[0]))
        assert errors[1] < errors[0]

    def test_the_front_speed_degrades_with_a_coarse_h_dry(self):
        """Documents a genuine limitation, so nobody rediscovers it as a bug.

        At a vacuum front the exact solution's depth vanishes quadratically,
        so the leading edge lies below any fixed h_dry and is damped: the
        coarser h_dry is, the slower the computed front. Refining the mesh does
        not help - it only resolves more of the truncated region. Picking
        h_dry is therefore a modelling decision, not a formality.
        """
        speeds = {}
        for h_dry in (1e-4, 1e-10):
            thresholds = WetDryThresholds(h_dry=h_dry, h_wet=h_dry * 10)
            pde = SWME1D('damBreak_dryBed', 0.0, 1.0, False, False,
                         wet_dry=thresholds)
            data, _, _ = run(pde, 0, 'damBreak_dryBed', resolution=400,
                             t_end=self.T_END)
            speeds[h_dry] = data[data[:, 1] > 1e-12, 0].max() / self.T_END

        exact = 2.0 * np.sqrt(self.H0)
        assert speeds[1e-10] > speeds[1e-4]                 # tighter is faster
        assert abs(speeds[1e-10] - exact) < abs(speeds[1e-4] - exact)
        assert speeds[1e-10] == pytest.approx(exact, rel=0.10)

    def test_the_dry_region_stays_exactly_dry_ahead_of_the_front(self, solved):
        data, _ = solved
        far_ahead = data[data[:, 0] > 0.7, 1]
        assert np.all(far_ahead <= DEFAULTS.h_dry)

    def test_no_velocity_is_reported_where_there_is_no_water(self, solved):
        data, _ = solved
        dry = data[:, 1] <= 0.0
        assert np.all(data[dry, 2] == 0.0)
        assert np.all(np.isfinite(data[:, 2]))

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_it_runs_at_every_moment_order(self, order):
        pde = SWME1D('damBreak_dryBed', 0.0, 1.0, False, False)
        data, _, _ = run(pde, order, 'damBreak_dryBed', resolution=120,
                         t_end=0.15)
        assert np.isfinite(data).all()
        assert data[:, 1].min() >= 0.0

    def test_it_runs_with_friction_on_the_implicit_source_path(self):
        """Viscous drying needs implicit source integration - see the next
        test for why, and for the failure it is avoiding."""
        pde = SWME1D('damBreak_dryBed', 0.01, 0.05, False, True,
                     wet_dry=self.THRESHOLDS)
        mesh = mesh_module.UniformRectangularMesh1D([-1.0, 1.0], 120)
        simulation = ClassicalSimulation1D(
            1, pde, mesh, 'INFLOW_OUTFLOW', 'damBreak_dryBed', sd.Roe(),
            timeIntegration.ImplicitEuler(True))
        data = simulation.run_simulation(0.15)
        assert np.isfinite(data).all()
        assert data[:, 1].min() >= 0.0
        # and the moment stays physical rather than blowing up
        assert np.abs(data[:, 3]).max() < 10.0

    def test_explicit_friction_near_a_dry_front_is_stiff_and_says_so(self):
        """A real limitation, pinned so it stays documented rather than being
        rediscovered as a mystery crash.

        The Navier-slip friction carries nu/h^2. Near a drying front that is
        arbitrarily large however the dry threshold is chosen (flooring h at
        h_dry only caps it at nu/h_dry^2, which is still ~1e6 at the default),
        so explicit integration of the source goes unstable: the moment blows
        up first and drags the height negative. Measured: this fails for
        viscosities from 1e-4 up, and for h_dry from 1e-4 to 1e-2 alike, so it
        is genuinely about stiffness rather than about threshold tuning.

        The error must name the cause, because the symptom (negative height)
        points at the positivity limiter, which is not at fault.
        """
        pde = SWME1D('damBreak_dryBed', 0.01, 0.05, False, False)
        with pytest.raises(RuntimeError, match="stiff near"):
            run(pde, 1, 'damBreak_dryBed', resolution=120, t_end=0.15)

    def test_partial_dam_break_onto_a_shallow_layer_also_works(self):
        pde = SWME1D('damBreak_dryBed_partial', 0.0, 1.0, False, False)
        data, _, _ = run(pde, 1, 'damBreak_dryBed_partial', resolution=200,
                         t_end=0.2)
        assert np.isfinite(data).all()
        assert data[:, 1].min() >= 0.0


# ---------------------------------------------------------------------------
# 4. the positivity limiter, and what it must NOT do
# ---------------------------------------------------------------------------

class TestPositivityLimiter:

    def test_a_wet_run_is_bit_identical_to_one_with_wet_dry_disabled(self):
        """The limiter must not bind, and the desingularization must not
        activate, for a flow that stays well above h_wet - this is what keeps
        every previously validated result intact. Compared against thresholds
        pushed so low they cannot possibly engage."""
        disabled = WetDryThresholds(eps_div=1e-300, h_dry=1e-290,
                                    h_wet=1e-280)
        a, _, _ = run(SWME1D('damBreak_noVelocity', 0.01, 0.05, False, False),
                      1, 'damBreak_noVelocity', resolution=60, t_end=0.25)
        b, _, _ = run(SWME1D('damBreak_noVelocity', 0.01, 0.05, False, False,
                             wet_dry=disabled),
                      1, 'damBreak_noVelocity', resolution=60, t_end=0.25)
        np.testing.assert_array_equal(a, b)

    def test_a_draining_cell_reaches_dry_without_going_negative(self):
        """Constant exfiltration-free drainage: a lone puddle spreading out."""
        class Puddle(SWME1D):
            def get_initial_values(self, order, initial_condition, position):
                values = np.zeros(order + 2)
                values[0] = 0.5 if abs(position) < 0.1 else 0.0
                return values

        pde = Puddle('puddle', 0.0, 1.0, False, False)
        data, _, _ = run(pde, 1, 'puddle', resolution=200, t_end=0.4)
        assert data[:, 1].min() >= 0.0
        assert np.isfinite(data).all()

    def test_the_limiter_reports_rather_than_silently_stalling(self):
        """If a state really cannot be advanced, that must surface as an
        error, not as a zero timestep that hangs the run."""
        import inspect
        from swme import simulation as simulation_module
        source = inspect.getsource(simulation_module.ClassicalSimulation1D)
        assert "Positivity limiter drove the timestep to zero" in source


# ---------------------------------------------------------------------------
# 5. the physics that would be quietly wrong
# ---------------------------------------------------------------------------

class TestRainCanWetDryGround:
    """The single most important behavioural check in this step. Before it,
    RechargeSWME1D.compute_source_term returned an all-zero source for any
    cell at or below its dry tolerance, which made it impossible for rainfall
    to ever wet dry ground - in a rainfall-runoff model."""

    @staticmethod
    def _pde(rainfall, infiltration_rate=0.0, **kwargs):
        from recharge.initial_conditions import RechargeSWME1D_CustomIC
        from recharge.laws import AdmissibleMixingFriction, ConstantInfiltration
        return RechargeSWME1D_CustomIC(
            'constantHeight_noVelocity', 0.0, 1.0, False, False, rainfall,
            ConstantInfiltration(I0=infiltration_rate),
            AdmissibleMixingFriction(alpha_R=0.0, alpha_I=0.0),
            **kwargs,
        )

    def test_the_mass_source_is_active_on_a_completely_dry_cell(self):
        pde = self._pde(rainfall=1e-2)
        pde.set_source_context(time=0.0, dt=1e-3, cell_index=0, x=0.0)
        source = pde.compute_source_term(1, np.zeros(3), 1e-3)
        assert source[0] == pytest.approx(1e-2)

    def test_rain_carries_no_horizontal_momentum_onto_dry_ground(self):
        pde = self._pde(rainfall=1e-2)
        pde.set_source_context(time=0.0, dt=1e-3, cell_index=0, x=0.0)
        source = pde.compute_source_term(1, np.zeros(3), 1e-3)
        assert np.all(source[1:] == 0.0)

    def test_infiltration_cannot_drain_a_cell_that_has_no_water(self):
        pde = self._pde(rainfall=0.0, infiltration_rate=1.0)
        pde.set_source_context(time=0.0, dt=1e-3, cell_index=0, x=0.0)
        source = pde.compute_source_term(1, np.zeros(3), 1e-3)
        assert source[0] == pytest.approx(0.0)

    def test_a_dry_domain_actually_fills_up_over_a_real_run(self):
        """End to end: start from a bone-dry bed and rain on it.

        Depth must end up R * t. Note this compares against the *actual*
        simulated time, not the configured t_end: `run_simulation`'s loop is
        `while t < t_end` with no final partial step, so a run overshoots its
        requested end time by up to one timestep. That is pre-existing
        behaviour, unrelated to wet-dry, and deliberately not changed here -
        see RESTRUCTURE_PLAN.md Step 6's notes.
        """
        from recharge.initial_conditions import RechargeSWME1D_CustomIC
        from recharge.laws import AdmissibleMixingFriction, ConstantInfiltration

        class DryStart(RechargeSWME1D_CustomIC):
            def get_initial_values(self, order, initial_condition, position):
                # A film, not a vacuum: on a *completely* dry grid the CFL
                # condition has nothing to bite on and the timestep is set by
                # the dry fallback, which is far too coarse for a source term.
                values = np.zeros(order + 2)
                values[0] = 1e-3
                return values

        pde = DryStart('dry', 0.0, 1.0, False, False, 0.5,
                       ConstantInfiltration(I0=0.0),
                       AdmissibleMixingFriction(alpha_R=0.0, alpha_I=0.0))
        mesh = mesh_module.UniformRectangularMesh1D([0.0, 1.0], 20)
        simulation = ClassicalSimulation1D(
            1, pde, mesh, 'PERIODIC', 'dry', sd.Roe(),
            timeIntegration.ExplicitEuler())
        simulation.store_history = True
        simulation.history_stride = 1
        data = simulation.run_simulation(0.2)

        t_final = simulation.history[-1]["time"]
        assert data[:, 1].min() > 0.0
        assert np.mean(data[:, 1]) == pytest.approx(1e-3 + 0.5 * t_final,
                                                    rel=1e-3)

    def test_rain_wets_a_genuinely_empty_cell_in_one_step(self):
        """The vacuum case, isolated from timestep-control concerns."""
        from recharge.laws import AdmissibleMixingFriction, ConstantInfiltration
        from recharge.initial_conditions import RechargeSWME1D_CustomIC
        pde = RechargeSWME1D_CustomIC(
            'x', 0.0, 1.0, False, False, 0.5,
            ConstantInfiltration(I0=0.0),
            AdmissibleMixingFriction(alpha_R=0.0, alpha_I=0.0))
        pde.set_source_context(time=0.0, dt=0.01, cell_index=0, x=0.0)
        state = np.zeros(3)
        updated = timeIntegration.ExplicitEuler().integrate(
            state, lambda v, dt: pde.compute_source_term(1, v, dt), 0.01)
        assert updated[0] == pytest.approx(0.5 * 0.01)


class TestHyperbolicityUnderDrying:
    """Step 5.5 established that hyperbolicity depends on alpha/sqrt(g*h), so
    a drying cell drives the *scaled* moments up even while the raw ones stay
    small. The moment ramp exists partly to suppress exactly that; this checks
    it actually does."""

    def test_the_ramp_bounds_the_scaled_moments_as_the_cell_empties(self):
        raw, ramped = [], []
        for h in np.geomspace(1e-10, 1.0, 400):
            state = np.array([h, 0.0, h * 0.3])       # physical alpha_1 = 0.3
            _, _, alpha = desingularized_primitives(1, state, DEFAULTS)
            raw.append(0.3 / np.sqrt(h))              # what it would have been
            ramped.append(abs(alpha[0]) / np.sqrt(h))
        assert max(raw) > 1e4                          # unbounded without it
        assert max(ramped) < max(raw)
        # and the scaled moment collapses in the dry limit rather than growing
        assert ramped[0] == pytest.approx(0.0, abs=1e-9)

    def test_the_model_stays_hyperbolic_through_a_dry_dam_break(self):
        """A(U) itself, evaluated at every state the run visits, stays real.

        This is the model-level question, and it is NOT the same as the
        interface-level counter on the scheme - see the next test.
        """
        from swme import pde as pde_module
        original = pde_module._compute_system_matrix_generic
        worst = {'max_imag': 0.0, 'calls': 0}

        def traced(order, values, g=1, hyperbolic=False, thresholds=None):
            kwargs = {} if thresholds is None else {'thresholds': thresholds}
            matrix = original(order, values, g=g, hyperbolic=hyperbolic, **kwargs)
            worst['calls'] += 1
            worst['max_imag'] = max(worst['max_imag'], float(np.max(np.abs(
                np.imag(np.linalg.eigvals(matrix))))))
            return matrix

        pde_module._compute_system_matrix_generic = traced
        try:
            pde = SWME1D('damBreak_dryBed', 0.0, 1.0, False, False,
                         wet_dry=WetDryThresholds(h_dry=1e-10, h_wet=1e-9))
            run(pde, 2, 'damBreak_dryBed', resolution=120, t_end=0.15)
        finally:
            pde_module._compute_system_matrix_generic = original

        assert worst['calls'] > 0
        assert worst['max_imag'] <= 1e-10

    def test_the_interface_average_does_go_complex_at_a_wet_dry_front(self):
        """A distinction that is easy to get backwards, so it is pinned here.

        The always-on counter on the scheme eigendecomposes the *path-averaged*
        matrix sum_k w_k A(psi(s_k)), not A at any state. A(U) is nonlinear in
        U, so that average is not A(anything) and need not be hyperbolic even
        when every matrix being averaged is. A jump from h=1 to h=0 across one
        interface is violent enough to do it.

        The proof that this says nothing about the model: it happens at N=0
        too, i.e. for plain shallow water, which is unconditionally
        hyperbolic and has no moments to destabilize.
        """
        counts = {}
        for order in (0, 1, 2):
            scheme = sd.Roe()
            pde = SWME1D('damBreak_dryBed', 0.0, 1.0, False, False)
            run(pde, order, 'damBreak_dryBed', resolution=120, t_end=0.15,
                scheme=scheme)
            counts[order] = scheme.nonhyperbolic_count

        assert counts[0] > 0, (
            "expected the interface average to go complex even at N=0; if it "
            "no longer does, the distinction this test documents may have "
            "changed"
        )
        # Identical across orders, because it is not about the moments at all.
        assert counts[0] == counts[1] == counts[2]
