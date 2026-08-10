"""Tests for bottom topography and well-balancing (RESTRUCTURE_PLAN.md Step 5).

Three groups, in increasing order of how much of the solver they exercise:

1. Unit tests of the pieces - the bed-profile library, the mesh's sampled bed
   with its ghost cells, and the structure of the augmented system matrix.
2. The algebraic C-property: A~(W).(W_R - W_L) = 0 for a lake-at-rest jump,
   at every point of the linear path, for arbitrary moment order.
3. The C-property through the actual scheme: a full `ClassicalSimulation1D`
   run over a non-trivial bed must hold the lake at rest to machine precision.
   This is the validation the plan flagged as mandatory for this step - do not
   weaken its tolerances to make an unrelated change pass.

Plus the strictly-additive guarantee: a config that never mentions topography
must produce bit-identical results to the pre-topography solver.
"""

from __future__ import annotations

import numpy as np
import pytest

from swme import mesh as mesh_module
from swme import pde as pde_module
from swme import spatialDiscretization as sd
from swme import timeIntegration
from swme import topography as topo
from swme.simulation import ClassicalSimulation1D

GRAVITY = 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_pde(hyperbolic=False, topography=None):
    """A frictionless SWME1D, so a lake at rest has an exactly zero source."""
    return pde_module.SWME1D(
        initial_condition='lakeAtRest',
        viscosity=0.0,
        slip_length=1.0,
        hyperbolic=hyperbolic,
        linear_source=False,
        topography=topography,
    )


def lake_at_rest_state(order, height, bed):
    """Augmented state W = (h, 0, 0, ..., 0, Z)."""
    w = np.zeros(order + 3, dtype=np.float64)
    w[0] = height
    w[order + 2] = bed
    return w


def build_lake_simulation(order, bed_profile, reference_level, resolution,
                          boundaries, boundary_condition, scheme=None,
                          initial_condition='lakeAtRest', **perturbation):
    """Wire up a complete lake-at-rest simulation over a given bed."""
    settings = topo.TopographySettings(
        bed_elevation=bed_profile,
        reference_water_level=reference_level,
        **perturbation,
    )
    _pde = make_pde(topography=settings)
    _pde.initial_condition = initial_condition

    _mesh = mesh_module.UniformRectangularMesh1D(boundaries, resolution)
    _mesh.set_bed_elevation(bed_profile, boundary_condition)

    simulation = ClassicalSimulation1D(
        order,
        _pde,
        _mesh,
        boundary_condition,
        initial_condition,
        scheme if scheme is not None else sd.Roe(),
        timeIntegration.ExplicitEuler(),
    )
    return simulation, _mesh


# ---------------------------------------------------------------------------
# 1. the bed-profile library
# ---------------------------------------------------------------------------

class TestBedProfiles:

    def test_every_advertised_profile_builds_and_evaluates(self):
        x = np.linspace(-1.0, 2.0, 17)
        for name in topo.available_bed_profiles():
            z = topo.get_bed_profile(name)(x)
            assert np.shape(z) == np.shape(x), name
            assert np.isfinite(z).all(), name

    def test_profiles_accept_scalars(self):
        for name in topo.available_bed_profiles():
            z = topo.get_bed_profile(name)(0.37)
            assert np.isscalar(z) or np.ndim(z) == 0, name

    def test_flat_is_identically_zero_by_default(self):
        x = np.linspace(0.0, 1.0, 11)
        assert np.all(topo.get_bed_profile('flat')(x) == 0.0)

    def test_flat_honours_a_nonzero_elevation(self):
        x = np.linspace(0.0, 1.0, 11)
        assert np.all(topo.get_bed_profile('flat', elevation=2.5)(x) == 2.5)

    def test_gaussian_bump_peaks_at_its_centre(self):
        z_of_x = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.3, center=0.4, width=0.1)
        assert z_of_x(0.4) == pytest.approx(0.3)
        assert z_of_x(10.0) == pytest.approx(0.0, abs=1e-12)

    def test_parabolic_bump_is_clipped_at_zero_outside_its_support(self):
        z_of_x = topo.get_bed_profile(
            'parabolic_bump', amplitude=0.2, center=10.0, half_width=2.0)
        assert z_of_x(10.0) == pytest.approx(0.2)
        assert z_of_x(8.0) == pytest.approx(0.0)
        assert z_of_x(0.0) == pytest.approx(0.0)   # clipped, not negative

    def test_linear_slope(self):
        z_of_x = topo.get_bed_profile('linear_slope', slope=0.5, x_ref=1.0)
        assert z_of_x(3.0) == pytest.approx(1.0)
        assert z_of_x(1.0) == pytest.approx(0.0)

    def test_sinusoidal_is_periodic_over_its_wavelength(self):
        z_of_x = topo.get_bed_profile(
            'sinusoidal', amplitude=0.1, wavelength=0.5)
        assert z_of_x(0.2) == pytest.approx(z_of_x(0.7))

    def test_step_is_discontinuous(self):
        z_of_x = topo.get_bed_profile('step', amplitude=0.4, position=0.5)
        assert z_of_x(0.499) == pytest.approx(0.0)
        assert z_of_x(0.501) == pytest.approx(0.4)

    def test_tanh_step_interpolates_between_its_two_levels(self):
        z_of_x = topo.get_bed_profile(
            'tanh_step', amplitude=0.4, position=0.5, width=0.01)
        assert z_of_x(0.5) == pytest.approx(0.2)
        assert z_of_x(0.0) == pytest.approx(0.0, abs=1e-12)
        assert z_of_x(1.0) == pytest.approx(0.4, abs=1e-12)

    def test_unknown_profile_name_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown bed profile"):
            topo.get_bed_profile('mount_doom')

    def test_unknown_profile_parameter_is_rejected_not_ignored(self):
        with pytest.raises(TypeError, match="Bad parameters"):
            topo.get_bed_profile('gaussian_bump', amplitud=0.2)

    @pytest.mark.parametrize("name,params", [
        ('gaussian_bump', {'width': 0.0}),
        ('parabolic_bump', {'half_width': -1.0}),
        ('sinusoidal', {'wavelength': 0.0}),
        ('tanh_step', {'width': -0.1}),
    ])
    def test_degenerate_length_scales_are_rejected(self, name, params):
        with pytest.raises(ValueError):
            topo.get_bed_profile(name, **params)


class TestTopographySettings:

    def test_defaults_are_a_flat_bed_at_zero(self):
        settings = topo.TopographySettings()
        assert settings.bed_elevation is None
        assert settings.elevation_at(1.234) == 0.0

    def test_elevation_at_returns_a_plain_float(self):
        settings = topo.TopographySettings(
            bed_elevation=topo.get_bed_profile('gaussian_bump'))
        value = settings.elevation_at(0.5)
        assert isinstance(value, float)


# ---------------------------------------------------------------------------
# 2. the mesh's sampled bed
# ---------------------------------------------------------------------------

class TestMeshBedElevation:

    def test_default_bed_is_flat_and_topography_is_off(self):
        m = mesh_module.UniformRectangularMesh1D([0.0, 1.0], 8)
        assert m.bed_elevation.shape == (10,)
        assert np.all(m.bed_elevation == 0.0)
        assert m.has_topography is False
        assert m.bed_elevation_function is None

    def test_sampling_fills_physical_cells_from_the_profile(self):
        m = mesh_module.UniformRectangularMesh1D([0.0, 1.0], 8)
        z_of_x = topo.get_bed_profile('linear_slope', slope=2.0)
        m.set_bed_elevation(z_of_x, 'INFLOW_OUTFLOW')
        np.testing.assert_allclose(
            m.bed_elevation[1:9], z_of_x(m.cell_center_positions))
        assert m.has_topography is True
        assert m.bed_elevation_function is z_of_x

    def test_inflow_outflow_ghost_cells_are_zero_gradient(self):
        m = mesh_module.UniformRectangularMesh1D([0.0, 1.0], 8)
        m.set_bed_elevation(
            topo.get_bed_profile('linear_slope', slope=2.0), 'INFLOW_OUTFLOW')
        assert m.bed_elevation[0] == m.bed_elevation[1]
        assert m.bed_elevation[9] == m.bed_elevation[8]

    def test_periodic_ghost_cells_wrap_like_the_state_array_does(self):
        # ClassicalSimulation1D._update_boundary_conditions uses
        # values[-2] for the left ghost and values[1] for the right ghost;
        # the bed has to be filled identically or the edge interfaces see an
        # inconsistent (U, Z) pair.
        m = mesh_module.UniformRectangularMesh1D([0.0, 1.0], 8)
        m.set_bed_elevation(
            topo.get_bed_profile('linear_slope', slope=2.0), 'PERIODIC')
        assert m.bed_elevation[0] == m.bed_elevation[8]
        assert m.bed_elevation[9] == m.bed_elevation[1]

    def test_an_everywhere_zero_profile_leaves_topography_off(self):
        # This is what keeps `bed_profile = flat` a genuine no-op rather than
        # a numerically-equivalent-but-different code path.
        m = mesh_module.UniformRectangularMesh1D([0.0, 1.0], 8)
        m.set_bed_elevation(topo.get_bed_profile('flat'), 'PERIODIC')
        assert m.has_topography is False

    def test_a_constant_nonzero_bed_still_counts_as_topography(self):
        m = mesh_module.UniformRectangularMesh1D([0.0, 1.0], 8)
        m.set_bed_elevation(topo.get_bed_profile('flat', elevation=3.0),
                            'PERIODIC')
        assert m.has_topography is True

    def test_non_finite_bed_is_rejected(self):
        m = mesh_module.UniformRectangularMesh1D([0.0, 1.0], 8)
        with pytest.raises(ValueError, match="non-finite"):
            m.set_bed_elevation(lambda x: np.full_like(x, np.nan), 'PERIODIC')


# ---------------------------------------------------------------------------
# 3. structure of the augmented system matrix
# ---------------------------------------------------------------------------

class TestAugmentedSystemMatrix:

    @pytest.mark.parametrize("order", range(0, 6))
    @pytest.mark.parametrize("hyperbolic", [False, True])
    def test_top_left_block_is_exactly_the_ordinary_system_matrix(
            self, order, hyperbolic):
        rng = np.random.default_rng(1000 + order)
        state = np.concatenate(([1.0 + rng.uniform()], rng.normal(size=order + 1)))
        bed = rng.normal()
        _pde = make_pde(hyperbolic=hyperbolic)

        n = order + 2
        plain = _pde.compute_system_matrix(order, state)
        augmented = _pde.compute_augmented_system_matrix(
            order, np.append(state, bed))

        np.testing.assert_array_equal(augmented[:n, :n], plain)

    @pytest.mark.parametrize("order", range(0, 6))
    def test_bed_slope_entry_is_plus_g_h_in_the_momentum_row(self, order):
        # Sign matters and is easy to get backwards: the bed-slope term is
        # -g*h*dZ/dx as a right-hand-side source, hence +g*h*dZ/dx once moved
        # into the transport matrix on the left.
        rng = np.random.default_rng(2000 + order)
        state = np.concatenate(([1.0 + rng.uniform()], rng.normal(size=order + 1)))
        _pde = make_pde()
        n = order + 2
        augmented = _pde.compute_augmented_system_matrix(
            order, np.append(state, 0.7))

        assert augmented[1, n] == pytest.approx(GRAVITY * state[0])
        # nothing else in the bed column
        column = np.delete(augmented[:, n], 1)
        assert np.all(column == 0.0)

    @pytest.mark.parametrize("order", range(0, 6))
    def test_bed_row_is_zero_because_the_bed_does_not_evolve(self, order):
        _pde = make_pde()
        state = np.zeros(order + 3)
        state[0] = 1.5
        state[order + 2] = 0.3
        augmented = _pde.compute_augmented_system_matrix(order, state)
        assert np.all(augmented[order + 2, :] == 0.0)
        assert augmented.shape == (order + 3, order + 3)

    def test_gravity_is_honoured(self):
        _pde = make_pde()
        state = np.array([2.0, 0.0, 0.0, 0.5])   # h, h*u_m, h*a_1, Z
        augmented = _pde.compute_augmented_system_matrix(1, state, g=9.81)
        assert augmented[1, 3] == pytest.approx(9.81 * 2.0)

    @pytest.mark.parametrize("bad,match", [
        (np.array([1.0, 0.0, 0.0]), "length 4"),          # too short for N=1
        (np.array([1.0, 0.0, np.nan, 0.0]), "Non-finite"),
        (np.array([-1.0, 0.0, 0.0, 0.0]), "Negative height"),
    ])
    def test_input_validation(self, bad, match):
        with pytest.raises(ValueError, match=match):
            make_pde().compute_augmented_system_matrix(1, bad)

    def test_a_dry_cell_is_accepted_not_rejected(self):
        """h == 0 stopped being an error in Step 6 (wet-dry): a dry cell over
        topography is an ordinary state, and must produce a finite matrix."""
        augmented = make_pde().compute_augmented_system_matrix(
            1, np.array([0.0, 0.0, 0.0, 0.7]))
        assert np.isfinite(augmented).all()


# ---------------------------------------------------------------------------
# 4. the algebraic C-property
# ---------------------------------------------------------------------------

class TestAlgebraicCProperty:
    """A~(W(s)).(W_R - W_L) must vanish identically along the whole path when
    both cells are at a common lake-at-rest free-surface level."""

    PATH_POINTS = (0.0, 0.11270166537925831, 0.5, 0.8872983346207417, 1.0)

    @pytest.mark.parametrize("order", range(0, 7))
    @pytest.mark.parametrize("hyperbolic", [False, True])
    def test_equilibrium_jump_is_annihilated_along_the_path(
            self, order, hyperbolic):
        rng = np.random.default_rng(3000 + order)
        _pde = make_pde(hyperbolic=hyperbolic)

        for _ in range(20):
            level = 2.0 + rng.uniform()
            bed_left, bed_right = rng.uniform(0.0, 1.0, size=2)
            w_left = lake_at_rest_state(order, level - bed_left, bed_left)
            w_right = lake_at_rest_state(order, level - bed_right, bed_right)
            jump = w_right - w_left

            for s in self.PATH_POINTS:
                w = (1.0 - s) * w_left + s * w_right
                residual = _pde.compute_augmented_system_matrix(order, w) @ jump
                assert np.max(np.abs(residual)) < 1e-13

    @pytest.mark.parametrize("order", range(0, 5))
    def test_a_non_equilibrium_jump_is_not_annihilated(self, order):
        """Guards against the test above passing for a trivial reason (e.g. an
        accidentally all-zero augmented matrix)."""
        _pde = make_pde()
        w_left = lake_at_rest_state(order, 1.0, 0.5)
        w_right = lake_at_rest_state(order, 1.3, 0.5)   # surface not level
        residual = _pde.compute_augmented_system_matrix(
            order, 0.5 * (w_left + w_right)) @ (w_right - w_left)
        assert np.max(np.abs(residual)) > 1e-3

    def test_the_sign_of_the_bed_column_is_what_makes_this_work(self):
        """Flipping the bed-slope entry to -g*h breaks the C-property. Pins the
        sign choice documented in
        pde._compute_augmented_system_matrix_generic."""
        order = 1
        _pde = make_pde()
        w_left = lake_at_rest_state(order, 1.6, 0.4)
        w_right = lake_at_rest_state(order, 1.3, 0.7)
        jump = w_right - w_left
        w_mid = 0.5 * (w_left + w_right)

        correct = _pde.compute_augmented_system_matrix(order, w_mid)
        flipped = correct.copy()
        flipped[1, order + 2] *= -1.0

        assert np.max(np.abs(correct @ jump)) < 1e-14
        assert np.max(np.abs(flipped @ jump)) > 1e-2


class TestSchemeLevelCProperty:
    """Both fluctuations of a single interface must vanish at a lake at rest.
    The central part does so for every scheme; whether the numerical viscosity
    does is exactly what `SpatialDiscretization.well_balanced` records."""

    @staticmethod
    def _interface_fluctuations(scheme, order, level, bed_left, bed_right):
        _pde = make_pde()
        w_left = lake_at_rest_state(order, level - bed_left, bed_left)
        w_right = lake_at_rest_state(order, level - bed_right, bed_right)
        return scheme.compute_fluctuation(
            w_left,
            w_right,
            lambda w: _pde.compute_augmented_system_matrix(order, w),
            0.01,
            0.05,
        )

    @pytest.mark.parametrize("scheme_class", [sd.Roe, sd.Osher])
    @pytest.mark.parametrize("order", range(0, 5))
    def test_well_balanced_schemes_produce_zero_fluctuations(
            self, scheme_class, order):
        assert scheme_class.well_balanced is True
        minus, plus = self._interface_fluctuations(
            scheme_class(), order, 2.0, 0.3, 0.8)
        assert np.max(np.abs(minus)) < 1e-13
        assert np.max(np.abs(plus)) < 1e-13

    @pytest.mark.parametrize("scheme_class", [sd.LF, sd.PRICE])
    def test_non_well_balanced_schemes_are_honestly_flagged(self, scheme_class):
        # Documents a real limitation rather than hiding it: the constant term
        # in these schemes' viscosity leaves an O(dx/dt * delta h) residual at
        # rest, so they must not be used with topography.
        assert scheme_class.well_balanced is False
        minus, plus = self._interface_fluctuations(
            scheme_class(), 1, 2.0, 0.3, 0.8)
        assert max(np.max(np.abs(minus)), np.max(np.abs(plus))) > 1e-3


# ---------------------------------------------------------------------------
# 5. the mandatory end-to-end lake-at-rest regression
# ---------------------------------------------------------------------------

BED_CASES = [
    ('gaussian_bump', {'amplitude': 0.4, 'center': 0.5, 'width': 0.1},
     'INFLOW_OUTFLOW'),
    ('parabolic_bump', {'amplitude': 0.5, 'center': 0.5, 'half_width': 0.2},
     'INFLOW_OUTFLOW'),
    ('step', {'amplitude': 0.6, 'position': 0.5}, 'INFLOW_OUTFLOW'),
    ('tanh_step', {'amplitude': 0.6, 'position': 0.5, 'width': 0.05},
     'INFLOW_OUTFLOW'),
    ('linear_slope', {'slope': 0.5}, 'INFLOW_OUTFLOW'),
    ('sinusoidal', {'amplitude': 0.3, 'wavelength': 1.0}, 'PERIODIC'),
]


class TestLakeAtRest:
    """RESTRUCTURE_PLAN.md Step 5 calls this validation mandatory: a flat free
    surface over a non-trivial bed, at rest, must stay at machine-precision
    rest when run forward in time. Do not relax these tolerances."""

    REFERENCE_LEVEL = 2.0
    RESOLUTION = 50
    BOUNDARIES = [0.0, 1.0]
    T_END = 0.5

    @pytest.mark.parametrize("profile_name,params,boundary", BED_CASES)
    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_lake_at_rest_is_preserved(self, profile_name, params, boundary,
                                       order):
        bed_profile = topo.get_bed_profile(profile_name, **params)
        simulation, m = build_lake_simulation(
            order, bed_profile, self.REFERENCE_LEVEL,
            self.RESOLUTION, self.BOUNDARIES, boundary,
        )
        assert m.has_topography is True

        result = simulation.run_simulation(self.T_END)

        # result columns: [x, h, u_m, a_1, ..., a_N]
        height = result[:, 1]
        bed = m.bed_elevation[1:self.RESOLUTION + 1]
        free_surface = height + bed

        np.testing.assert_allclose(
            free_surface, self.REFERENCE_LEVEL, atol=1e-12,
            err_msg=f"free surface drifted for {profile_name}, N={order}",
        )
        assert np.max(np.abs(result[:, 2:])) < 1e-12, (
            f"spurious velocity/moments for {profile_name}, N={order}"
        )

    @pytest.mark.parametrize("order", [0, 1, 2])
    def test_lake_at_rest_holds_for_the_hyperbolic_variant_too(self, order):
        bed_profile = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.4, center=0.5, width=0.1)
        settings = topo.TopographySettings(
            bed_elevation=bed_profile,
            reference_water_level=self.REFERENCE_LEVEL,
        )
        _pde = make_pde(hyperbolic=True, topography=settings)
        m = mesh_module.UniformRectangularMesh1D(
            self.BOUNDARIES, self.RESOLUTION)
        m.set_bed_elevation(bed_profile, 'INFLOW_OUTFLOW')

        simulation = ClassicalSimulation1D(
            order, _pde, m, 'INFLOW_OUTFLOW', 'lakeAtRest',
            sd.Roe(), timeIntegration.ExplicitEuler(),
        )
        result = simulation.run_simulation(self.T_END)

        bed = m.bed_elevation[1:self.RESOLUTION + 1]
        np.testing.assert_allclose(
            result[:, 1] + bed, self.REFERENCE_LEVEL, atol=1e-12)
        assert np.max(np.abs(result[:, 2:])) < 1e-12

    def test_lake_at_rest_holds_end_to_end_under_osher_too(self):
        # The interface-level test above is in principle sufficient (zero
        # fluctuations everywhere means a zero update), but Osher is the other
        # scheme this repo advertises as well balanced, so confirm it whole.
        # Small grid: Osher does one eigendecomposition per quadrature node.
        bed_profile = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.4, center=0.5, width=0.1)
        simulation, m = build_lake_simulation(
            1, bed_profile, self.REFERENCE_LEVEL, 20, self.BOUNDARIES,
            'INFLOW_OUTFLOW', scheme=sd.Osher(),
        )
        result = simulation.run_simulation(0.2)
        np.testing.assert_allclose(
            result[:, 1] + m.bed_elevation[1:21], self.REFERENCE_LEVEL,
            atol=1e-12)
        assert np.max(np.abs(result[:, 2:])) < 1e-12

    def test_lake_at_rest_holds_under_friction(self):
        """Friction acts on the velocity, which is zero, so a lake at rest must
        survive a viscous run as well - this checks the source step does not
        inject anything on its own."""
        bed_profile = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.4, center=0.5, width=0.1)
        settings = topo.TopographySettings(
            bed_elevation=bed_profile,
            reference_water_level=self.REFERENCE_LEVEL,
        )
        _pde = pde_module.SWME1D('lakeAtRest', 0.01, 0.05, False, False,
                                 topography=settings)
        m = mesh_module.UniformRectangularMesh1D(self.BOUNDARIES, 30)
        m.set_bed_elevation(bed_profile, 'INFLOW_OUTFLOW')

        simulation = ClassicalSimulation1D(
            1, _pde, m, 'INFLOW_OUTFLOW', 'lakeAtRest',
            sd.Roe(), timeIntegration.ExplicitEuler(),
        )
        result = simulation.run_simulation(0.2)
        np.testing.assert_allclose(
            result[:, 1] + m.bed_elevation[1:31], self.REFERENCE_LEVEL,
            atol=1e-12)
        assert np.max(np.abs(result[:, 2:])) < 1e-12

    def test_a_perturbation_of_the_lake_actually_moves(self):
        """Complements the tests above: the scheme must be still at rest, but
        it must not be inert. A small free-surface bump has to propagate, and
        it must stay small - no O(1) waves generated by the bed."""
        bed_profile = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.4, center=0.5, width=0.1)
        amplitude = 1e-3
        simulation, m = build_lake_simulation(
            1, bed_profile, self.REFERENCE_LEVEL, self.RESOLUTION,
            self.BOUNDARIES, 'INFLOW_OUTFLOW',
            initial_condition='perturbedLakeAtRest',
            perturbation_amplitude=amplitude,
            perturbation_center=0.15,
            perturbation_width=0.03,
        )
        result = simulation.run_simulation(0.3)

        free_surface = result[:, 1] + m.bed_elevation[1:self.RESOLUTION + 1]
        deviation = free_surface - self.REFERENCE_LEVEL

        # It moved ...
        assert np.max(np.abs(deviation)) > 1e-6
        assert np.max(np.abs(result[:, 2])) > 1e-6
        # ... and it stayed a perturbation: the bed did not generate waves of
        # its own, which would be orders of magnitude larger than this.
        assert np.max(np.abs(deviation)) < 10 * amplitude


# ---------------------------------------------------------------------------
# 6. the strictly-additive guarantee
# ---------------------------------------------------------------------------

def run_dam_break(bed_profile=None, boundary_condition='INFLOW_OUTFLOW',
                  order=1, resolution=24, t_end=0.3):
    """A short ordinary run, optionally over a bed, for A/B comparisons."""
    settings = (topo.TopographySettings(bed_elevation=bed_profile)
                if bed_profile is not None else None)
    _pde = pde_module.SWME1D('damBreak_noVelocity', 0.01, 0.05, False, False,
                             topography=settings)
    m = mesh_module.UniformRectangularMesh1D([-1.0, 1.0], resolution)
    if bed_profile is not None:
        m.set_bed_elevation(bed_profile, boundary_condition)
    simulation = ClassicalSimulation1D(
        order, _pde, m, boundary_condition, 'damBreak_noVelocity',
        sd.Roe(), timeIntegration.ExplicitEuler(),
    )
    return simulation.run_simulation(t_end), m


class TestFlatBedIsUnchanged:

    def test_no_topography_matches_an_explicitly_flat_bed_bit_for_bit(self):
        """`bed_profile = flat` must be a genuine no-op, not merely a
        numerically-equivalent second code path - that is what makes this whole
        step strictly additive for every pre-existing config."""
        without, _ = run_dam_break(bed_profile=None)
        with_flat, m = run_dam_break(bed_profile=topo.get_bed_profile('flat'))
        assert m.has_topography is False
        np.testing.assert_array_equal(without, with_flat)

    def test_a_constant_bed_takes_the_augmented_path_and_agrees(self):
        """A constant nonzero bed has zero slope, so it is physically the same
        problem - but it *does* switch the solver onto the augmented path.
        Agreement here is the cross-check that the augmented path reproduces
        the plain one."""
        without, _ = run_dam_break(bed_profile=None)
        with_constant, m = run_dam_break(
            bed_profile=topo.get_bed_profile('flat', elevation=3.0))
        assert m.has_topography is True
        np.testing.assert_allclose(without, with_constant, rtol=1e-10,
                                   atol=1e-12)


# ---------------------------------------------------------------------------
# 7. initial conditions and wiring
# ---------------------------------------------------------------------------

class TestTopographyInitialConditions:

    def test_lake_at_rest_sets_h_to_the_reference_level_minus_the_bed(self):
        bed_profile = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.4, center=0.5, width=0.1)
        _pde = make_pde(topography=topo.TopographySettings(
            bed_elevation=bed_profile, reference_water_level=2.0))
        for x in (0.0, 0.25, 0.5, 0.9):
            values = _pde.get_initial_values(2, 'lakeAtRest', x)
            assert values[0] == pytest.approx(2.0 - bed_profile(x))
            assert np.all(values[1:] == 0.0)

    def test_lake_at_rest_over_the_default_flat_bed_is_just_a_constant_depth(self):
        _pde = make_pde()
        values = _pde.get_initial_values(1, 'lakeAtRest', 0.42)
        assert values[0] == pytest.approx(1.0)

    def test_perturbed_lake_adds_a_gaussian_to_the_free_surface(self):
        _pde = make_pde(topography=topo.TopographySettings(
            reference_water_level=2.0,
            perturbation_amplitude=0.1,
            perturbation_center=0.5,
            perturbation_width=0.05,
        ))
        assert _pde.get_initial_values(1, 'perturbedLakeAtRest', 0.5)[0] == \
            pytest.approx(2.1)
        assert _pde.get_initial_values(1, 'perturbedLakeAtRest', 5.0)[0] == \
            pytest.approx(2.0)

    def test_a_bed_poking_out_of_the_water_is_rejected_with_a_clear_message(self):
        # Dry cells arrive in Step 6; until then this must fail loudly at
        # setup rather than cryptically mid-run.
        _pde = make_pde(topography=topo.TopographySettings(
            bed_elevation=topo.get_bed_profile(
                'gaussian_bump', amplitude=2.0, center=0.5, width=0.1),
            reference_water_level=1.0,
        ))
        with pytest.raises(ValueError, match="non-positive water height"):
            _pde.get_initial_values(1, 'lakeAtRest', 0.5)


class TestSimulationWiring:

    def test_a_non_well_balanced_scheme_warns_when_topography_is_active(self):
        bed_profile = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.2, center=0.0, width=0.3)
        simulation, _ = build_lake_simulation(
            1, bed_profile, 2.0, 12, [-1.0, 1.0], 'INFLOW_OUTFLOW',
            scheme=sd.LF(),
        )
        with pytest.warns(RuntimeWarning, match="not well balanced"):
            simulation.run_simulation(0.02)

    def test_no_well_balancing_warning_without_topography(self):
        # Scoped to the well-balancing warning on purpose. A blanket
        # "no warnings at all" assertion trips on a pre-existing ComplexWarning
        # from Roe's eigendecomposition (np.linalg.eig returns complex arrays;
        # the tiny imaginary part is discarded on assignment), which fires on
        # the flat-bed path too and is unrelated to topography.
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_dam_break(bed_profile=None)
        assert not [w for w in caught if "well balanced" in str(w.message)]

    def test_a_pde_without_the_augmented_matrix_is_rejected(self, monkeypatch):
        bed_profile = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.2, center=0.0, width=0.3)
        simulation, _ = build_lake_simulation(
            1, bed_profile, 2.0, 12, [-1.0, 1.0], 'INFLOW_OUTFLOW')
        monkeypatch.delattr(pde_module.SWME1D,
                            'compute_augmented_system_matrix')
        with pytest.raises(NotImplementedError,
                           match="compute_augmented_system_matrix"):
            simulation.run_simulation(0.02)

    def test_a_bed_sampled_with_the_wrong_boundary_condition_is_rejected(self):
        # Otherwise this is a silent wrong answer confined to the two edge
        # interfaces - the kind that looks like a plausible physical result.
        bed_profile = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.2, center=0.0, width=0.3)
        simulation, m = build_lake_simulation(
            1, bed_profile, 2.0, 12, [-1.0, 1.0], 'INFLOW_OUTFLOW')
        m.set_bed_elevation(bed_profile, 'PERIODIC')     # mismatched
        with pytest.raises(ValueError, match="boundary condition"):
            simulation.run_simulation(0.02)

    def test_a_mis_sized_bed_array_is_rejected(self):
        bed_profile = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.2, center=0.0, width=0.3)
        simulation, m = build_lake_simulation(
            1, bed_profile, 2.0, 12, [-1.0, 1.0], 'INFLOW_OUTFLOW')
        m.bed_elevation = m.bed_elevation[:-1]     # forgot the ghost cells
        with pytest.raises(ValueError, match="one entry per cell"):
            simulation.run_simulation(0.02)


class TestRechargeInheritsTopography:

    @staticmethod
    def _recharge_pde(settings):
        from recharge.initial_conditions import RechargeSWME1D_CustomIC
        from recharge.laws import AdmissibleMixingFriction, ConstantInfiltration
        return RechargeSWME1D_CustomIC(
            'lakeAtRest', 0.0, 1.0, False, False, 0.0,
            ConstantInfiltration(I0=0.0),
            AdmissibleMixingFriction(alpha_R=0.0, alpha_I=0.0),
            topography=settings,
        )

    def test_recharge_pde_accepts_and_uses_topography(self):
        bed_profile = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.4, center=0.5, width=0.1)
        settings = topo.TopographySettings(
            bed_elevation=bed_profile, reference_water_level=2.0)
        _pde = self._recharge_pde(settings)

        # inherited IC ...
        assert _pde.get_initial_values(1, 'lakeAtRest', 0.5)[0] == \
            pytest.approx(2.0 - bed_profile(0.5))
        # ... and inherited augmented matrix
        state = lake_at_rest_state(1, 1.6, 0.4)
        assert _pde.compute_augmented_system_matrix(1, state)[1, 3] == \
            pytest.approx(1.6)

    def test_lake_at_rest_holds_for_the_recharge_model_with_zero_forcing(self):
        bed_profile = topo.get_bed_profile(
            'gaussian_bump', amplitude=0.4, center=0.5, width=0.1)
        settings = topo.TopographySettings(
            bed_elevation=bed_profile, reference_water_level=2.0)
        _pde = self._recharge_pde(settings)

        m = mesh_module.UniformRectangularMesh1D([0.0, 1.0], 30)
        m.set_bed_elevation(bed_profile, 'INFLOW_OUTFLOW')
        simulation = ClassicalSimulation1D(
            1, _pde, m, 'INFLOW_OUTFLOW', 'lakeAtRest',
            sd.Roe(), timeIntegration.ExplicitEuler(),
        )
        result = simulation.run_simulation(0.2)
        np.testing.assert_allclose(
            result[:, 1] + m.bed_elevation[1:31], 2.0, atol=1e-12)
        assert np.max(np.abs(result[:, 2:])) < 1e-12
