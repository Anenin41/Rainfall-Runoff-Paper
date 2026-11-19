"""Shoreline treatment: wet-dry interfaces across a bed step.

The treatment (`swme.wetdry.shoreline_fluctuation`, wired into
`ClassicalSimulation1D.run_simulation`) adapts the modified Roe scheme of
Castro, Gonzalez-Vida and Pares, "Numerical treatment of wet/dry fronts in
shallow flows with a modified Roe scheme", M3AS 16 (2006) 897-931, to the
path-conservative PVM schemes and to the moment variables.

What is checked, and against what:

* the pieces: where the treatment acts, Castro's case analysis (Cases 2,
  3.1, 3.2, 3.3) and his exact interface states (eqs 5.8, 5.14, 5.23-5.29),
  the momentum flux against the system matrix, the fluctuation's
  well-balance, mass conservation and positivity, for every scheme and
  several orders;
* a lake at rest with dry shores, held to round-off over a long run (the
  extended C-property of Castro Section 5.4), with the negative control that
  the same run fails on its first step with the treatment switched off;
* Thacker's planar sloshing in a parabolic bowl, which has an exact solution
  with two moving shorelines;
* Castro Test 1 (shock reflection at an emerged step) and Castro Test 2 (dam
  break over an inclined plane, rising and falling).

Everything runs in the code's units, g = 1.
"""

import warnings

import numpy as np
import pytest
from numpy.polynomial import legendre

from swme import cli, mesh, pde, simulation, spatialDiscretization, timeIntegration
from swme import wetdry
from swme.wetdry import WetDryThresholds

SCHEMES = {
    'Roe': spatialDiscretization.Roe,
    'Osher': spatialDiscretization.Osher,
    'LF': spatialDiscretization.LF,
    'PRICE': spatialDiscretization.PRICE,
}


class _RoeGaussLegendre(spatialDiscretization.Roe):
    """Roe with the correct five-point Gauss-Legendre path quadrature.

    The shipped schemes carry mis-transcribed nodes (HANDOFF.md, audit item
    A3; docs/limitations.md), which makes them non-conservative in momentum
    across a strong shock. Castro Test 1 is such a shock, so it is run with
    both: this class shows the shoreline treatment reproduces the exact
    solution, the shipped one pins the size of the known error.
    """

    _NODES, _WEIGHTS = legendre.leggauss(5)

    def compute_fluctuation(self, value_left, value_right, system_matrix,
                            delta_t, delta_x):
        jump = value_right - value_left
        matrix = sum(
            0.5 * w * system_matrix(value_left + 0.5 * (s + 1.0) * jump)
            for s, w in zip(self._NODES, self._WEIGHTS))
        viscosity = self.compute_viscosity(matrix, delta_t, delta_x) @ jump
        central = matrix @ jump
        return (central - viscosity) / 2, (central + viscosity) / 2


class _CustomInitialCondition(pde.SWME1D):
    """SWME/HSWME started from an arbitrary state function of x."""

    def __init__(self, state_of_x, **kwargs):
        super().__init__('custom', 0.0, 1.0, True, False, **kwargs)
        self._state_of_x = state_of_x

    def get_initial_values(self, order, initial_condition, position):
        values = np.zeros(order + 2)
        state = self._state_of_x(position)
        values[:len(state)] = state
        return values


def _build(state_of_x, bed, domain, cells, *, order=0, scheme='Roe',
           treatment=True, h_dry=1e-4, h_wet=1e-3):
    thresholds = WetDryThresholds(h_dry=h_dry, h_wet=h_wet,
                                  shoreline_treatment=treatment)
    model = _CustomInitialCondition(state_of_x, wet_dry=thresholds)
    grid = mesh.UniformRectangularMesh1D(list(domain), cells)
    grid.set_bed_elevation(bed, 'INFLOW_OUTFLOW')
    scheme_object = scheme() if isinstance(scheme, type) else SCHEMES[scheme]()
    sim = simulation.ClassicalSimulation1D(
        order, model, grid, 'INFLOW_OUTFLOW', 'custom', scheme_object,
        timeIntegration.ExplicitEuler())
    sim.store_history = True
    sim.history_stride = 1
    return sim


def _run(sim, t_end):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return sim.run_simulation(t_end)


def _cell_width(sim):
    return (sim.mesh.boundaries[1] - sim.mesh.boundaries[0]) / sim.mesh.resolution


def _masses(sim):
    return np.array([np.sum(entry['data'][:, 1]) for entry in sim.history]) \
        * _cell_width(sim)


def _system_matrix(order, hyperbolic=True):
    model = pde.SWME1D('unused', 0.0, 1.0, hyperbolic, False)
    return lambda values: model.compute_system_matrix(order, values)


# ---------------------------------------------------------------------------
# 1. which interfaces are treated
# ---------------------------------------------------------------------------

class TestWhereTheTreatmentActs:
    """`shoreline_fluctuation` returns None, i.e. the ordinary fluctuation is
    kept, unless a cell is dry and the bed is not flat, and also in Castro's
    Case 3.3."""

    T = WetDryThresholds(h_dry=1e-4, h_wet=1e-3)

    def _treated(self, left, right, bed_left, bed_right, **kwargs):
        return wetdry.shoreline_fluctuation(
            SCHEMES['Roe'](), _system_matrix(0), np.array(left, dtype=float),
            np.array(right, dtype=float), bed_left, bed_right, 0.01, 0.05,
            self.T, **kwargs) is not None

    def test_dry_cell_above_the_water_line(self):
        assert self._treated([0.3, 0.0], [0.0, 0.0], 0.0, 0.5)
        assert self._treated([0.0, 0.0], [0.3, 0.0], 0.5, 0.0)

    def test_water_on_a_ledge_above_a_dry_lower_cell(self):
        assert self._treated([0.3, 0.0], [0.0, 0.0], 0.5, 0.0)

    def test_a_thin_film_counts_as_dry(self):
        assert self._treated([0.3, 0.0], [5e-5, 0.0], 0.0, 0.5)

    def test_two_wet_cells_are_never_treated(self):
        assert not self._treated([0.3, 0.0], [0.01, 0.0], 0.0, 0.5)

    def test_a_flat_interface_is_never_treated(self):
        assert not self._treated([0.3, 0.1], [0.0, 0.0], 0.2, 0.2)
        assert not self._treated([0.0, 0.0], [0.0, 0.0], 0.2, 0.2)

    def test_still_water_over_a_submerged_dry_bed_is_case_3_3(self):
        # the dry cell's bed lies below the wet cell's free surface
        assert not self._treated([0.3, 0.0], [0.0, 0.0], 0.0, 0.2)

    def test_case_3_3_takes_the_wall_when_forced(self):
        assert self._treated([0.3, 0.0], [0.0, 0.0], 0.0, 0.2, force_wall=True)


# ---------------------------------------------------------------------------
# 1b. Castro's nonlinear Riemann problems, piece by piece
# ---------------------------------------------------------------------------

class TestMomentumFlux:
    """The conservative rows of the fluctuation use F_mom; its gradient has
    to be the momentum row of the system matrix the scheme integrates."""

    @pytest.mark.parametrize('hyperbolic', [False, True])
    @pytest.mark.parametrize('order', [0, 1, 2, 3])
    def test_its_gradient_is_the_momentum_row_of_the_system_matrix(
            self, hyperbolic, order):
        rng = np.random.default_rng(10 * order + hyperbolic)
        matrix = _system_matrix(order, hyperbolic)
        for _ in range(10):
            state = np.concatenate([[rng.uniform(0.2, 2.0)],
                                    rng.normal(size=order + 1)])
            gradient = np.array([
                (wetdry.momentum_flux(state + 1e-6 * e, 1.0, hyperbolic)
                 - wetdry.momentum_flux(state - 1e-6 * e, 1.0, hyperbolic))
                / 2e-6 for e in np.eye(order + 2)])
            np.testing.assert_allclose(gradient, matrix(state)[1], atol=1e-8)

    def test_negative_control_hswme_drops_the_higher_moments(self):
        """With all moments in the flux (what the wall used before), the
        gradient misses the HSWME momentum row at N = 2."""
        state = np.array([0.7, 0.2, 0.1, 0.3])
        row = _system_matrix(2, True)(state)[1]
        gradient = np.array([
            (wetdry.momentum_flux(state + 1e-6 * e, 1.0, False)
             - wetdry.momentum_flux(state - 1e-6 * e, 1.0, False)) / 2e-6
            for e in np.eye(4)])
        assert np.max(np.abs(gradient - row)) > 0.1

    def test_an_empty_state_has_no_flux(self):
        assert wetdry.momentum_flux(np.zeros(3)) == 0.0


class TestVacuumEdgeState:
    """Castro eq. (5.8): the rarefaction into a dry bed, sampled at x = 0."""

    h = 0.8

    def test_supercritical_flow_keeps_the_cell_state(self):
        u = 1.2 * np.sqrt(self.h)
        assert wetdry.vacuum_edge_state(self.h, u) == (self.h, u)

    def test_flow_away_faster_than_2c_leaves_nothing(self):
        assert wetdry.vacuum_edge_state(self.h, -2.0 * np.sqrt(self.h)) == (0.0, 0.0)

    @pytest.mark.parametrize('froude', [-1.9, -1.0, 0.0, 0.5, 0.999])
    def test_in_between_the_state_is_critical_with_the_same_invariant(
            self, froude):
        g = 1.3
        c = np.sqrt(g * self.h)
        h0, u0 = wetdry.vacuum_edge_state(self.h, froude * c, g)
        assert u0 == pytest.approx(np.sqrt(g * h0), rel=1e-14)
        assert u0 + 2 * np.sqrt(g * h0) == pytest.approx(froude * c + 2 * c,
                                                         rel=1e-14)
        assert h0 * u0 == pytest.approx((froude * c + 2 * c) ** 3 / (27 * g),
                                        rel=1e-13)

    def test_it_is_continuous_at_both_ends(self):
        c = np.sqrt(self.h)
        h0, u0 = wetdry.vacuum_edge_state(self.h, c * (1 - 1e-12))
        assert (h0, u0) == pytest.approx((self.h, c), rel=1e-11)
        h0, _ = wetdry.vacuum_edge_state(self.h, -2 * c * (1 - 1e-9))
        assert h0 < 1e-17


class TestBernoulliDepth:
    """h + q^2/(2 g h^2) = E, the steady contact at a bed step."""

    @pytest.mark.parametrize('q,energy', [(0.3, 1.0), (1e-6, 0.01), (2.0, 3.0)])
    def test_both_roots_satisfy_the_relation_either_side_of_critical(
            self, q, energy):
        g = 1.3
        critical = (q * q / g) ** (1 / 3)
        for supercritical in (True, False):
            h = wetdry.bernoulli_depth(q, energy, g, supercritical)
            assert h + q * q / (2 * g * h * h) == pytest.approx(energy, rel=1e-13)
            assert (h < critical) == supercritical

    def test_the_supercritical_root_is_the_least_positive_root_of_eq_5_15(self):
        g, h_l, q_l, rise = 9.81, 0.3, 0.9, 0.5        # Castro's H_R - H_L
        roots = np.roots([1.0, -rise - q_l ** 2 / (2 * g * h_l ** 2) - h_l,
                          0.0, q_l ** 2 / (2 * g)])
        roots = np.sort(roots[np.abs(roots.imag) < 1e-12].real)
        energy = h_l + q_l ** 2 / (2 * g * h_l ** 2) + rise
        assert wetdry.bernoulli_depth(q_l, energy, g, True) == pytest.approx(
            roots[roots > 0].min(), rel=1e-12)
        assert wetdry.bernoulli_depth(q_l, energy, g, False) == pytest.approx(
            roots.max(), rel=1e-12)

    def test_below_the_critical_energy_it_returns_the_critical_depth(self):
        assert wetdry.bernoulli_depth(0.3, 0.1) == pytest.approx(0.09 ** (1 / 3))


class TestOneSidedProblem:
    """Castro's case analysis for a cell facing an empty neighbour; `u` is
    the velocity towards the neighbour, g = 1."""

    @staticmethod
    def _solve(h, u, step, direction=1, moment=None):
        state = np.array([h, direction * h * u] + ([] if moment is None
                                                   else [h * moment]))
        return wetdry.one_sided_problem(state, direction, step)

    @pytest.mark.parametrize('direction', [1, -1])
    def test_still_or_receding_water_below_an_emerged_bed_meets_a_wall(
            self, direction):
        for u in (0.0, -0.3):
            kind, near, far = self._solve(0.4, u, 0.5, direction)
            assert kind == 'wall'
            assert near[1] == 0.0 and np.all(far == 0.0)
            assert near[0] == pytest.approx(
                0.4 + wetdry.wall_depth_change(0.4, u))

    def test_a_submerged_dry_bed_keeps_the_ordinary_scheme(self):
        assert self._solve(0.4, -0.3, 0.3)[0] == 'ordinary'
        assert self._solve(0.4, 0.0, 0.3)[0] == 'ordinary'

    def test_flow_without_the_energy_to_climb_an_emerged_step_is_reflected(self):
        # Fr = 0.5 against a step of 0.5 > h: eq. (5.20) holds
        kind, near, _ = self._solve(0.4, 0.5 * np.sqrt(0.4), 0.5)
        assert kind == 'wall'
        assert near[0] > 0.4                    # the reflected shock, (5.27)

    def test_subcritical_flow_with_the_energy_to_climb_is_case_3_3(self):
        """An emerged step can only be climbable at Fr >= 3^(3/4) (from
        u^2/(2g) >= (3/2) h_c), so subcritical flow with the energy to climb
        always faces a submerged step."""
        h, u = 0.4, 0.9 * np.sqrt(0.4)
        energy = h + u * u / 2
        critical = (h * u) ** (2 / 3)
        step = 0.5 * (energy - 1.5 * critical)
        assert self._solve(h, u, step)[0] == 'ordinary'

    def test_supercritical_flow_with_the_energy_climbs_eq_5_29(self):
        h, u, step, moment = 0.2, 1.5, 0.3, 0.1
        kind, near, far = self._solve(h, u, step, -1, moment)
        assert kind == 'climb'
        np.testing.assert_array_equal(near, [h, -h * u, h * moment])
        assert far[1] == -h * u                 # discharge unchanged
        energy = far[0] + (h * u) ** 2 / (2 * far[0] ** 2)
        assert energy == pytest.approx(h + u * u / 2 - step, rel=1e-13)
        assert far[0] < (h * u) ** (2 / 3)      # supercritical on the step
        assert far[2] == pytest.approx(far[0] * moment)

    def test_supercritical_flow_without_the_energy_is_reflected(self):
        assert self._solve(0.2, 1.5, 1.4)[0] == 'wall'

    def test_the_wall_needs_the_bed_to_emerge(self):
        """The deliberate reading of Castro's Case 3.1 (docs/numerics.md):
        flow towards a dry cell whose bed is below the free surface is not
        walled off, even when it could not pass the step in steady flow."""
        h, u, step = 0.1, 0.2, 0.09
        energy, critical = h + u * u / 2, (h * u) ** (2 / 3)
        assert step > energy - 1.5 * critical    # eq. (5.20) holds
        assert self._solve(h, u, step)[0] == 'ordinary'

    @pytest.mark.parametrize('froude', [-1.5, 0.0, 0.6])
    @pytest.mark.parametrize('direction', [1, -1])
    def test_overfall_from_the_critical_state(self, froude, direction):
        h, drop = 0.3, 0.2
        u = froude * np.sqrt(h)
        kind, near, far = self._solve(h, u, -drop, direction, 0.2)
        h0, u0 = wetdry.vacuum_edge_state(h, u)
        assert kind == 'overfall'
        np.testing.assert_allclose(near, [h0, direction * h0 * u0, 0.2 * h0],
                                   rtol=1e-14)
        assert far[1] == near[1]
        energy = far[0] + (h0 * u0) ** 2 / (2 * far[0] ** 2)
        assert energy == pytest.approx(h0 + u0 * u0 / 2 + drop, rel=1e-13)
        assert far[0] < h0                       # accelerates down the step
        assert far[2] == pytest.approx(0.2 * far[0])

    def test_overfall_of_supercritical_flow_starts_from_the_cell(self):
        kind, near, far = self._solve(0.3, 1.2, -0.2)
        assert kind == 'overfall'
        np.testing.assert_array_equal(near, [0.3, 0.36])
        assert far[1] == 0.36

    def test_water_running_away_from_the_drop_leaves_nothing(self):
        kind, near, far = self._solve(0.3, -2.1 * np.sqrt(0.3), -0.2)
        assert kind == 'vacuum'
        assert np.all(near == 0.0) and np.all(far == 0.0)

    def test_an_empty_cell_has_no_problem(self):
        assert wetdry.one_sided_problem(np.zeros(2), 1, 0.5)[0] == 'empty'

    def test_a_flat_interface_is_refused(self):
        with pytest.raises(ValueError, match='bed step'):
            wetdry.one_sided_problem(np.array([0.3, 0.0]), 1, 0.0)


class TestDrainsADryCell:

    T = WetDryThresholds(h_dry=1e-4, h_wet=1e-3)

    def test_any_loss_from_an_empty_cell_is_rejected(self):
        # flux in +x leaves the left cell, which is empty
        assert wetdry.drains_a_dry_cell(0.0, 0.5, 1e-9, 0.1, 0.1, self.T)
        # flux in -x leaves the right cell, which is empty
        assert wetdry.drains_a_dry_cell(0.5, 0.0, -1e-9, 0.1, 0.1, self.T)

    def test_a_flux_into_the_dry_cell_is_fine(self):
        assert not wetdry.drains_a_dry_cell(0.0, 0.5, -1e-3, 0.1, 0.1, self.T)
        assert not wetdry.drains_a_dry_cell(0.5, 0.0, 1e-3, 0.1, 0.1, self.T)

    def test_a_film_may_lose_up_to_half_of_itself(self):
        h = 8e-5
        assert not wetdry.drains_a_dry_cell(h, 0.5, 0.4 * h, 0.1, 0.1, self.T)
        assert wetdry.drains_a_dry_cell(h, 0.5, 0.6 * h, 0.1, 0.1, self.T)

    def test_wet_cells_are_left_to_the_timestep_limiter(self):
        assert not wetdry.drains_a_dry_cell(0.01, 0.5, 1.0, 0.1, 0.1, self.T)


# ---------------------------------------------------------------------------
# 2. Castro's exact wall state
# ---------------------------------------------------------------------------

class TestWallDepth:

    def test_at_rest_the_wall_depth_is_the_depth(self):
        assert wetdry.wall_depth_change(0.7, 0.0) == 0.0

    @pytest.mark.parametrize('u', [-1.5, -0.5, -1e-6])
    def test_receding_flow_is_castro_eq_5_25(self, u):
        h, g = 0.8, 1.0
        expected = (u / 2 + np.sqrt(g * h)) ** 2 / g
        assert h + wetdry.wall_depth_change(h, u, g) == pytest.approx(expected)

    def test_a_dry_zone_opens_when_the_water_outruns_its_expansion(self):
        h = 0.8
        assert wetdry.wall_depth_change(h, -2.0 * np.sqrt(h) - 0.1) == -h

    @pytest.mark.parametrize('u', [1e-6, 0.05, 0.5, 1.0, 3.2])
    def test_approaching_flow_is_the_greatest_root_of_castro_eq_5_27(self, u):
        h, g = 0.8, 1.3
        q = h * u
        roots = np.roots([1.0, -h, -h * h - 2.0 * q * q / (g * h), h ** 3])
        expected = roots.real.max()
        assert h + wetdry.wall_depth_change(h, u, g) == pytest.approx(
            expected, rel=1e-7)

    @pytest.mark.parametrize('u', [0.05, 0.5, 3.2])
    def test_the_reflected_shock_satisfies_rankine_hugoniot(self, u):
        """Independent of eq. (5.27): mass and momentum jump conditions
        between (h, u) and (h_w, 0) with one shock speed."""
        h, g = 0.8, 1.0
        h_w = h + wetdry.wall_depth_change(h, u, g)
        speed = -h * u / (h_w - h)
        momentum_jump = (0.5 * g * h_w ** 2) - (h * u * u + 0.5 * g * h ** 2)
        assert speed * (0.0 - h * u) == pytest.approx(momentum_jump, rel=1e-12)

    def test_small_velocities_keep_full_precision(self):
        """The cubic has a double root at rest; a generic root finder loses
        half the digits there. The linearised answer is d = h u / c."""
        h, u = 0.5, 1e-9
        assert wetdry.wall_depth_change(h, u) == pytest.approx(
            h * u / np.sqrt(h), rel=1e-8)

    def test_the_two_branches_join_continuously(self):
        h = 0.5
        below = wetdry.wall_depth_change(h, -1e-12)
        above = wetdry.wall_depth_change(h, +1e-12)
        assert abs(above - below) < 1e-11


# ---------------------------------------------------------------------------
# 3. the fluctuation
# ---------------------------------------------------------------------------

def _state(rng, order, h=None):
    h = rng.uniform(0.2, 1.5) if h is None else h
    return h * np.concatenate([[1.0, rng.normal(scale=0.4)],
                               rng.normal(scale=0.3, size=order)])


def _beds(lower_side, step=5.0):
    """Bed elevations (left, right) with the lower cell on `lower_side`."""
    return (0.0, step) if lower_side == 1 else (step, 0.0)


def _fluctuation(scheme, order, left, right, lower_side, step=5.0, **kwargs):
    return wetdry.shoreline_fluctuation(
        SCHEMES[scheme](), _system_matrix(order), left, right,
        *_beds(lower_side, step), 0.01, 0.05, hyperbolic=True, **kwargs)


class TestShorelineFluctuation:

    @pytest.mark.parametrize('scheme', sorted(SCHEMES))
    @pytest.mark.parametrize('order', [0, 1, 3])
    @pytest.mark.parametrize('lower_side', [1, -1])
    def test_water_at_rest_below_a_dry_cell_gives_exact_zeros(
            self, scheme, order, lower_side):
        wet = np.zeros(order + 2)
        wet[0] = 0.37
        dry = np.zeros(order + 2)
        left, right = (wet, dry) if lower_side == 1 else (dry, wet)
        fluct_min, fluct_plus = _fluctuation(scheme, order, left, right,
                                             lower_side, step=1.0)
        assert np.all(fluct_min == 0.0)
        assert np.all(fluct_plus == 0.0)

    @pytest.mark.parametrize('scheme', sorted(SCHEMES))
    @pytest.mark.parametrize('order', [0, 1, 2])
    @pytest.mark.parametrize('lower_side', [1, -1])
    @pytest.mark.parametrize('step', [0.05, 5.0])
    def test_one_mass_flux_for_both_cells(self, scheme, order, lower_side,
                                          step):
        """In flux form F = q_left + D-[0] = q_right - D+[0]: the mass that
        leaves one cell is the mass that enters the other, exactly. Every
        combination of a wet cell and a film, either way up, small and large
        steps (wall, climb, overfall and their superposition)."""
        rng = np.random.default_rng(100 + 7 * order + lower_side)
        for _ in range(20):
            wet, film = _state(rng, order), _state(rng, order, h=5e-5)
            for left, right in ((wet, film), (film, wet), (film, 0.4 * film)):
                result = _fluctuation(scheme, order, left, right, lower_side,
                                      step, force_wall=True)
                fluct_min, fluct_plus = result
                assert left[1] + fluct_min[0] == pytest.approx(
                    right[1] - fluct_plus[0], abs=1e-14)

    @pytest.mark.parametrize('scheme', ['Roe', 'Osher'])
    @pytest.mark.parametrize('order', [0, 1, 2])
    @pytest.mark.parametrize('lower_side', [1, -1])
    def test_no_mass_crosses_the_wall(self, scheme, order, lower_side):
        """Moving water below a high dry step: the interface mass flux is
        zero and the dry cell receives nothing at all."""
        rng = np.random.default_rng(200 + 11 * order + lower_side)
        wet, dry = _state(rng, order), np.zeros(order + 2)
        left, right = (wet, dry) if lower_side == 1 else (dry, wet)
        fluct_min, fluct_plus = _fluctuation(scheme, order, left, right,
                                             lower_side)
        to_wet, to_dry = ((fluct_min, fluct_plus) if lower_side == 1
                          else (fluct_plus, fluct_min))
        assert np.all(to_dry == 0.0)
        assert to_wet[0] == -lower_side * wet[1]

    @pytest.mark.parametrize('froude', [-1.0, -0.2, 0.3, 1.2])
    @pytest.mark.parametrize('lower_side', [1, -1])
    def test_wall_momentum_flux_is_castros_wall_pressure(self, froude,
                                                         lower_side):
        """N = 0: the flux at the wall is F(W_wall) = (0, g h_w^2 / 2),
        Castro eq. (5.1) with (5.24)-(5.27)."""
        h = 0.6
        u_normal = froude * np.sqrt(h)
        wet = np.array([h, lower_side * h * u_normal])
        dry = np.zeros(2)
        left, right = (wet, dry) if lower_side == 1 else (dry, wet)
        fluct_min, fluct_plus = _fluctuation('Roe', 0, left, right, lower_side)
        to_wet = fluct_min if lower_side == 1 else fluct_plus
        momentum_flux_in_cell = wet[1] ** 2 / h + 0.5 * h * h
        # D- = F_wall - F(U) for a left cell, D+ = F(U) - F_wall for a right one
        wall_flux = momentum_flux_in_cell + lower_side * to_wet[1]
        h_wall = h + wetdry.wall_depth_change(h, u_normal)
        assert wall_flux == pytest.approx(0.5 * h_wall ** 2, rel=1e-12)

    def test_hswme_wall_pressure_has_the_first_moment_only(self):
        """HSWME at N = 2: the wall's momentum flux is the model's,
        g h_w^2/2 + h_w a_1^2/3, without the a_2 term the system matrix
        does not carry."""
        h, u, a1, a2 = 0.6, 0.3, 0.2, 0.4
        wet = h * np.array([1.0, u, a1, a2])
        fluct_min, _ = _fluctuation('Roe', 2, wet, np.zeros(4), 1)
        cell_flux = h * u * u + 0.5 * h * h + h * a1 * a1 / 3
        h_wall = h + wetdry.wall_depth_change(h, u)
        assert cell_flux + fluct_min[1] == pytest.approx(
            0.5 * h_wall ** 2 + h_wall * a1 * a1 / 3, rel=1e-12)

    @pytest.mark.parametrize('scheme', ['Roe', 'Osher'])
    @pytest.mark.parametrize('order', [0, 1, 2])
    @pytest.mark.parametrize('lower_side', [1, -1])
    def test_the_dry_lower_cell_only_ever_gains_water(self, scheme, order,
                                                      lower_side):
        """Water on a ledge above a dry cell: whatever its velocity, the
        overfall flux points from the ledge to the dry cell or is zero."""
        rng = np.random.default_rng(300 + 13 * order + lower_side)
        for _ in range(20):
            ledge, dry = _state(rng, order), np.zeros(order + 2)
            left, right = (dry, ledge) if lower_side == 1 else (ledge, dry)
            fluct_min, fluct_plus = _fluctuation(scheme, order, left, right,
                                                 lower_side, step=0.3)
            gain_of_dry_cell = -(fluct_min[0] if lower_side == 1
                                 else fluct_plus[0])
            assert gain_of_dry_cell >= 0.0

    @pytest.mark.parametrize('lower_side', [1, -1])
    def test_overfall_fluxes_are_castros_case_2(self, lower_side):
        """N = 0, water at rest on a ledge: the ledge loses the flux of the
        critical state W0 of eq. (5.8), the dry cell below receives the flux
        of the supercritical state W1 on the same contact. The difference of
        the two momentum fluxes is the force of the step."""
        h, drop = 0.3, 0.2
        ledge, dry = np.array([h, 0.0]), np.zeros(2)
        left, right = (dry, ledge) if lower_side == 1 else (ledge, dry)
        fluct_min, fluct_plus = _fluctuation('Roe', 0, left, right, lower_side,
                                             step=drop)
        h0, u0 = wetdry.vacuum_edge_state(h, 0.0)
        q0 = h0 * u0
        h1 = wetdry.bernoulli_depth(q0, h0 + u0 * u0 / 2 + drop)
        to_ledge, to_dry = ((fluct_plus, fluct_min) if lower_side == 1
                            else (fluct_min, fluct_plus))
        # D on the ledge: F(U) - F(W0) for a right cell, F(W0) - F(U) for a left one
        flux_w0 = 0.5 * h * h - lower_side * to_ledge[1]
        assert flux_w0 == pytest.approx(q0 * u0 + 0.5 * h0 * h0, rel=1e-13)
        # D on the dry cell: F(W1) for a left cell, -F(W1) for a right one
        assert lower_side * to_dry[1] == pytest.approx(
            q0 * q0 / h1 + 0.5 * h1 * h1, rel=1e-13)
        assert -to_dry[0] == pytest.approx(q0, rel=1e-14)     # its gain
        assert q0 * q0 / h1 + 0.5 * h1 * h1 > flux_w0

    def test_supercritical_flow_climbs_onto_a_dry_step_case_3_2(self):
        """Castro Test 2 uphill and Test 1 at a low step: mass flows onto
        the dry step with the cell's discharge, and the step receives the
        flux of the supercritical state on it."""
        h, u, step = 0.2, 1.5, 0.3
        wet = np.array([h, h * u])
        fluct_min, fluct_plus = _fluctuation('Roe', 0, wet, np.zeros(2), 1,
                                             step=step)
        assert np.all(fluct_min == 0.0)             # D- = F(W_L) - F(W_L)
        h1 = wetdry.bernoulli_depth(h * u, h + u * u / 2 - step)
        np.testing.assert_allclose(
            fluct_plus, [-h * u, -((h * u) ** 2 / h1 + 0.5 * h1 * h1)],
            rtol=1e-13)

    def test_momentum_without_water_is_not_read_as_a_discharge(self):
        """A drained cell can keep a residue of momentum. The wall and the
        overfall are posed on the regularised state, where an empty cell
        carries none, so nothing flows out of it."""
        empty_with_residue = np.array([0.0, -6e-4])
        dry = np.zeros(2)
        fluct_min, fluct_plus = _fluctuation('Roe', 0, empty_with_residue, dry,
                                             1, step=1.0)
        mass_flux = empty_with_residue[1] + fluct_min[0]
        assert mass_flux == 0.0
        assert fluct_plus[0] == 0.0

    def test_regularised_state_leaves_a_wet_cell_untouched(self):
        values = np.array([0.5, 0.2, -0.1])
        assert wetdry.regularised_state(values) is values

    def test_regularised_state_empties_an_empty_cell(self):
        np.testing.assert_array_equal(
            wetdry.regularised_state(np.array([0.0, 3e-4, -2e-4])), 0.0)


# ---------------------------------------------------------------------------
# 4. lake at rest with dry shores (extended C-property)
# ---------------------------------------------------------------------------

def _bowl(x):
    return 0.5 * np.asarray(x, dtype=np.float64) ** 2


def _lake_in_bowl(level):
    return lambda x: [max(level - 0.5 * x * x, 0.0), 0.0]


class TestLakeAtRestWithDryShores:

    @pytest.mark.parametrize('scheme', ['Roe', 'Osher'])
    @pytest.mark.parametrize('order', [0, 1, 2])
    def test_the_lake_stays_at_rest_over_a_long_run(self, scheme, order):
        sim = _build(_lake_in_bowl(0.6), _bowl, (-2.0, 2.0), 60,
                     order=order, scheme=scheme)
        final = _run(sim, 10.0)
        initial = sim.history[0]['data']

        assert len(sim.history) > 200          # a long run, not a few steps
        assert np.sum(initial[:, 1] == 0.0) >= 20       # genuinely dry shores
        assert np.max(np.abs(final[:, 1] - initial[:, 1])) < 1e-14
        assert np.max(np.abs(final[:, 2:])) < 1e-13
        assert np.min(final[:, 1]) == 0.0
        assert sim.mass_created_by_clamping == 0.0
        masses = _masses(sim)
        assert np.max(np.abs(masses - masses[0])) < 1e-14

    def test_the_dry_cells_are_bit_for_bit_untouched(self):
        sim = _build(_lake_in_bowl(0.6), _bowl, (-2.0, 2.0), 60)
        final = _run(sim, 2.0)
        dry = sim.history[0]['data'][:, 1] == 0.0
        assert np.all(final[dry, 1:] == 0.0)

    @pytest.mark.parametrize('scheme', ['Roe', 'Osher'])
    @pytest.mark.parametrize('order', [0, 1])
    def test_negative_control_without_the_treatment_it_fails_at_once(
            self, scheme, order):
        sim = _build(_lake_in_bowl(0.6), _bowl, (-2.0, 2.0), 60,
                     order=order, scheme=scheme, treatment=False)
        with pytest.raises(RuntimeError, match='Negative height') as error:
            _run(sim, 10.0)
        assert 'step=0,' in str(error.value)

    def test_the_error_message_no_longer_blames_the_wrong_thing(self):
        sim = _build(_lake_in_bowl(0.6), _bowl, (-2.0, 2.0), 60,
                     treatment=False)
        with pytest.raises(RuntimeError) as error:
            _run(sim, 1.0)
        message = str(error.value)
        assert 'cannot come from the flux update' not in message
        assert 'at or below h_dry' in message
        assert 'depth before the flux update was 0.0' in message

    def test_a_fully_wet_lake_is_unchanged_by_the_switch(self):
        """Where no cell is dry the treatment must not act at all."""
        results = []
        for treatment in (True, False):
            sim = _build(lambda x: [1.5 - 0.5 * x * x + 0.05 * np.exp(-x * x), 0.0],
                         _bowl, (-1.0, 1.0), 40, treatment=treatment)
            results.append(_run(sim, 0.5))
        np.testing.assert_array_equal(results[0], results[1])


# ---------------------------------------------------------------------------
# 5. moving shorelines: Thacker's planar solution
# ---------------------------------------------------------------------------

class TestSloshingInAParabolicBowl:
    """Over Z = x^2/2 with g = 1 the free surface stays a straight line,

        eta(x, t) = c(t) + B cos(t) x,   u(t) = -B sin(t),
        c(t) = 1/2 + B^2/2 - B^2 cos(t)^2 / 2,

    so the two shorelines are at B cos(t) -+ sqrt(1 + B^2): they move by
    2B over half a period, one drying while the other floods (Thacker 1981).
    """

    B = 0.3

    def _initial(self, moment=0.0, order=0):
        def state(x):
            h = max(0.5 + self.B * x - 0.5 * x * x, 0.0)
            return [h, 0.0] + [moment * h] * min(order, 1)
        return state

    def _exact_depth(self, x, t):
        slope = self.B * np.cos(t)
        level = 0.5 + 0.5 * self.B ** 2 - 0.5 * slope ** 2
        return np.maximum(level + slope * x - 0.5 * x * x, 0.0)

    @pytest.mark.parametrize('scheme,order,moment', [
        ('Roe', 0, 0.0), ('Osher', 0, 0.0), ('Roe', 1, 0.2), ('Roe', 2, 0.2),
    ])
    def test_half_a_period_conserves_mass_and_stays_non_negative(
            self, scheme, order, moment):
        sim = _build(self._initial(moment, order), _bowl, (-2.5, 2.5), 60,
                     order=order, scheme=scheme, h_dry=1e-5, h_wet=1e-4)
        _run(sim, np.pi)
        masses = _masses(sim)
        assert np.max(np.abs(masses - masses[0])) < 1e-13
        assert min(entry['data'][:, 1].min() for entry in sim.history) >= 0.0
        assert sim.mass_created_by_clamping < 1e-15

    @pytest.mark.parametrize('scheme', ['Roe', 'Osher'])
    def test_the_solution_follows_thacker(self, scheme):
        sim = _build(self._initial(), _bowl, (-2.5, 2.5), 60, scheme=scheme,
                     h_dry=1e-5, h_wet=1e-4)
        _run(sim, np.pi)
        x = sim.history[0]['data'][:, 0]
        width = _cell_width(sim)

        errors = [np.sum(np.abs(entry['data'][:, 1]
                                - self._exact_depth(x, entry['time']))) * width
                  for entry in sim.history]
        volume = _masses(sim)[0]
        assert max(errors) < 0.06 * volume

        quarter = min(sim.history, key=lambda e: abs(e['time'] - np.pi / 2))
        core = np.abs(x) < 0.5
        assert np.mean(quarter['data'][core, 2]) == pytest.approx(
            -self.B * np.sin(quarter['time']), rel=0.1)

        # the right shore dries, the left one floods, by about 2B each
        def shores(entry):
            wet = x[entry['data'][:, 1] > 1e-3]
            return wet.min(), wet.max()
        left_start, right_start = shores(sim.history[0])
        left_end, right_end = shores(sim.history[-1])
        assert right_start - right_end == pytest.approx(2 * self.B, abs=2.5 * width)
        assert left_start - left_end == pytest.approx(2 * self.B, abs=2.5 * width)

    def test_negative_control_without_the_treatment_it_fails(self):
        sim = _build(self._initial(), _bowl, (-2.5, 2.5), 60, treatment=False,
                     h_dry=1e-5, h_wet=1e-4)
        with pytest.raises(RuntimeError, match='Negative height'):
            _run(sim, np.pi)


# ---------------------------------------------------------------------------
# 6. Castro Test 1: shock reflection at an emerged step
# ---------------------------------------------------------------------------

class TestCastroShockReflection:
    """Castro et al. Section 6.1. Water of depth 1 moving at 10 m/s
    (g = 9.81) meets a step of height 10 at x = 10; in g = 1 units the
    velocity is 10/sqrt(9.81). The step is never overtopped, and a shock
    travels back upstream leaving water at rest with the depth of eq. (5.27).
    """

    H, U = 1.0, 10.0 / np.sqrt(9.81)

    @staticmethod
    def _step(x):
        return np.where(np.asarray(x, dtype=np.float64) < 10.0, 0.0, 10.0)

    def _exact(self):
        h_wall = self.H + wetdry.wall_depth_change(self.H, self.U)
        return h_wall, -self.H * self.U / (h_wall - self.H)

    def _run_case(self, scheme, order=0, moment=0.0):
        def state(x):
            if x >= 10.0:
                return [0.0, 0.0]
            return [self.H, self.H * self.U] + [moment * self.H] * min(order, 1)
        sim = _build(state, self._step, (0.0, 12.0), 120, order=order,
                     scheme=scheme)
        final = _run(sim, 1.0)
        x, h = final[:, 0], final[:, 1]
        width = _cell_width(sim)
        upstream = x < 10.0
        near_wall = upstream & (x > 9.6)
        h_wall = np.median(h[near_wall])
        stored = np.sum(h[upstream] - self.H) * width
        shock_position = 10.0 - stored / (h_wall - self.H)
        return sim, final, h_wall, shock_position

    def test_exact_values_quoted_in_the_docs(self):
        h_wall, speed = self._exact()
        assert h_wall == pytest.approx(5.13060, abs=1e-5)
        assert speed == pytest.approx(-0.77295, abs=1e-5)

    def test_with_gauss_legendre_quadrature_the_exact_solution_is_recovered(self):
        sim, final, h_wall, shock_position = self._run_case(_RoeGaussLegendre)
        exact_depth, exact_speed = self._exact()
        time = sim.history[-1]['time']
        assert h_wall == pytest.approx(exact_depth, rel=2e-3)
        assert shock_position == pytest.approx(10.0 + exact_speed * time,
                                               abs=_cell_width(sim))

    @pytest.mark.parametrize('scheme', ['Roe', 'Osher'])
    def test_the_shipped_schemes_are_within_their_known_quadrature_error(
            self, scheme):
        """Measured: reflected depth 4.845 (Roe) and 4.869 (Osher) against
        5.131, i.e. 5.6 % and 5.1 % low. That deficit is the mis-transcribed
        path quadrature (audit item A3), not the shoreline treatment: see
        the test above. The bound is deliberately not tighter than that."""
        _, _, h_wall, _ = self._run_case(scheme)
        exact_depth, _ = self._exact()
        assert h_wall < exact_depth
        assert h_wall == pytest.approx(exact_depth, rel=0.07)

    @pytest.mark.parametrize('scheme,order,moment', [
        ('Roe', 0, 0.0), ('Osher', 0, 0.0), ('Roe', 1, 0.3), ('Roe', 2, 0.3),
    ])
    def test_the_step_stays_dry_and_the_wall_is_tight(self, scheme, order,
                                                      moment):
        sim, final, _, _ = self._run_case(scheme, order, moment)
        x = final[:, 0]
        assert np.all(final[x >= 10.0, 1:] == 0.0)
        assert np.min(final[:, 1]) >= 0.0
        # the only mass entering is the inflow q = H U through the left edge
        masses = _masses(sim)
        time = sim.history[-1]['time']
        assert masses[-1] - masses[0] == pytest.approx(self.H * self.U * time,
                                                       rel=1e-12)
        # water at the wall has been brought to rest
        near_wall = (x < 10.0) & (x > 9.6)
        assert np.max(np.abs(final[near_wall, 2])) < 1e-2


# ---------------------------------------------------------------------------
# 7. Castro Test 2: dam break over an inclined plane
# ---------------------------------------------------------------------------

class TestCastroDamBreakOnAPlane:
    """Castro et al. Section 6.2: bed Z = x tan(alpha), a horizontal lake
    surface for x < 0 with depth 1 at the dam, dry for x > 0. The front
    position is x_f(t) = 2 t sqrt(g h0 cos(alpha)) - g t^2 tan(alpha) / 2
    (eq. 6.6). alpha = +pi/60 is a front climbing an emerging bed,
    alpha = -pi/60 a front running down a dry slope.
    """

    @staticmethod
    def _case(alpha, **kwargs):
        tangent = np.tan(alpha)

        def bed(x):
            return np.asarray(x, dtype=np.float64) * tangent

        def state(x):
            return [max(1.0 - x * tangent, 0.0) if x < 0.0 else 0.0, 0.0]

        return _build(state, bed, (-4.0, 5.0), 180, h_dry=1e-6, h_wet=1e-5,
                      **kwargs)

    @staticmethod
    def _front(entry):
        x, h = entry['data'][:, 0], entry['data'][:, 1]
        return x[np.flatnonzero(h > 1e-3).max()]

    @pytest.mark.parametrize('alpha', [np.pi / 60, -np.pi / 60])
    @pytest.mark.parametrize('scheme,order', [('Roe', 0), ('Osher', 0),
                                              ('Roe', 2)])
    def test_the_front_runs_and_mass_is_conserved(self, alpha, scheme, order):
        sim = self._case(alpha, scheme=scheme, order=order)
        final = _run(sim, 1.5)
        masses = _masses(sim)
        # zero-gradient ends: the rarefaction has not reached x = -4, and the
        # front has not reached x = 5, so the domain is closed in effect
        assert np.max(np.abs(masses - masses[0])) < 1e-13
        assert np.min(final[:, 1]) >= 0.0
        assert sim.mass_created_by_clamping < 1e-15
        assert self._front(sim.history[-1]) > 1.5

    @pytest.mark.parametrize('alpha', [np.pi / 60, -np.pi / 60])
    def test_negative_control_without_the_treatment_it_fails(self, alpha):
        sim = self._case(alpha, treatment=False)
        with pytest.raises(RuntimeError, match='Negative height'):
            _run(sim, 1.5)

    @staticmethod
    def _front_speed(sim):
        early = min(sim.history, key=lambda e: abs(e['time'] - 0.75))
        late = sim.history[-1]
        speed = ((TestCastroDamBreakOnAPlane._front(late)
                  - TestCastroDamBreakOnAPlane._front(early))
                 / (late['time'] - early['time']))
        return speed, 0.5 * (late['time'] + early['time'])

    @pytest.mark.parametrize('alpha,tolerance', [(np.pi / 60, 0.20),
                                                 (-np.pi / 60, 0.13)])
    def test_front_speed_against_castro_eq_6_7(self, alpha, tolerance):
        """Front speed over t in [0.75, 1.5] against the exact
        u_f = 2 sqrt(cos(alpha)) - t tan(alpha).

        Measured at this resolution (dx = 0.05, h_dry = 1e-6, Roe): flat bed
        1.711 against 2.000 (-14.4 %), which is the lag of the first-order
        scheme at a vacuum front, with the front read off at h = 1e-3, and
        has nothing to do with the bed; uphill 1.601 against 1.940
        (-17.4 %); downhill 1.845 against 2.058 (-10.3 %). Before Castro's
        Cases 2 and 3.2 were implemented (the session-start code of
        2026-11-19) the uphill front ran at 1.466 (-24.4 %) and the downhill
        one at 1.724 (-16.2 %). The tolerances state the measurements; they
        are not targets.
        """
        sim = self._case(alpha)
        _run(sim, 1.5)
        speed, mid_time = self._front_speed(sim)
        exact = 2.0 * np.sqrt(np.cos(alpha)) - mid_time * np.tan(alpha)
        assert speed == pytest.approx(exact, rel=tolerance)
        assert speed < exact

    def test_the_slope_effect_relative_to_the_flat_bed(self):
        """The bed slope changes the exact front speed by -3.0 % uphill and
        +2.9 % downhill relative to a flat bed. Measured here, dx = 0.05:
        -6.4 % and +7.8 % (at dx = 0.025: -4.7 % and +5.7 %). The uphill
        front of the session-start code, which walled off every emerged
        step, was -14.3 %; this test fails for it. The bound is 6
        percentage points either way."""
        speeds = {}
        for name, alpha in (('up', np.pi / 60), ('flat', 0.0),
                            ('down', -np.pi / 60)):
            sim = self._case(alpha) if alpha else self._flat_case()
            _run(sim, 1.5)
            speed, mid_time = self._front_speed(sim)
            exact = 2.0 * np.sqrt(np.cos(alpha)) - mid_time * np.tan(alpha)
            speeds[name] = (speed, exact)
        flat, flat_exact = speeds['flat']
        for name in ('up', 'down'):
            speed, exact = speeds[name]
            assert speed / flat == pytest.approx(exact / flat_exact, abs=0.06)

    @staticmethod
    def _flat_case():
        return _build(lambda x: [1.0 if x < 0.0 else 0.0, 0.0],
                      lambda x: np.zeros_like(np.asarray(x, dtype=np.float64)) + 1.0,
                      (-4.0, 5.0), 180, h_dry=1e-6, h_wet=1e-5)

    def test_the_slope_changes_the_front_the_way_the_exact_solution_says(self):
        """Uphill the front is slower than on a flat bed, downhill faster."""
        fronts = {}
        for name, alpha in (('up', np.pi / 60), ('flat', 0.0),
                            ('down', -np.pi / 60)):
            sim = self._case(alpha) if alpha else self._flat_case()
            _run(sim, 1.5)
            fronts[name] = self._front(sim.history[-1])
        assert fronts['up'] < fronts['flat'] < fronts['down']


# ---------------------------------------------------------------------------
# 8. configuration
# ---------------------------------------------------------------------------

class TestConfiguration:

    def test_the_treatment_is_on_by_default(self):
        assert WetDryThresholds().shoreline_treatment is True
        assert cli.build_wet_dry({}).shoreline_treatment is True

    def test_the_key_is_read_from_the_config(self):
        thresholds = cli.build_wet_dry({'wet_dry': {'shoreline_treatment': False}})
        assert thresholds.shoreline_treatment is False

    def test_a_non_boolean_value_is_rejected(self):
        with pytest.raises(ValueError, match='shoreline_treatment'):
            cli.build_wet_dry({'wet_dry': {'shoreline_treatment': 'yes'}})

    def test_a_misspelt_key_is_still_rejected(self):
        with pytest.raises(ValueError, match='shoreline_treatmnt'):
            cli.build_wet_dry({'wet_dry': {'shoreline_treatmnt': True}})
