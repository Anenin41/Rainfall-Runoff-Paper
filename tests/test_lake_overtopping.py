"""The rain-fills-a-lake-until-it-overtops-a-dam case and what it needs:
the `basin_dam` bed profile, the `lakeWithDryShores` initial condition, the
opt-in mass budget, and a short version of the shipped config
`rain_fills_lake_overtops_dam`.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from swme import cli
from swme import topography as topo
from swme import pde


def _run(sim, t_end):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return sim.run_simulation(t_end)


# ---------------------------------------------------------------------------
# bed profile
# ---------------------------------------------------------------------------

class TestBasinDamProfile:

    def test_it_is_advertised(self):
        assert 'basin_dam' in topo.available_bed_profiles()

    def test_formula(self):
        params = dict(bank_height=0.9, bank_position=1.5, bank_width=0.5,
                      dam_height=0.5, dam_position=6.0, dam_width=0.4,
                      downstream_slope=0.05)
        z = topo.get_bed_profile('basin_dam', **params)
        x = np.linspace(0.0, 10.0, 41)
        expected = (0.9 * np.exp(-((x - 1.5) / 0.5) ** 2)
                    + 0.5 * np.exp(-((x - 6.0) / 0.4) ** 2)
                    - 0.05 * np.maximum(x - 6.0, 0.0))
        np.testing.assert_allclose(z(x), expected, rtol=0, atol=1e-15)

    def test_the_three_pieces(self):
        z = topo.get_bed_profile('basin_dam')       # defaults
        # the bank is a ridge: a crest of the given height, falling to both
        # sides, so it is a divide
        assert z(1.0) == pytest.approx(0.6, abs=1e-9)
        assert z(1.0) > z(0.9) and z(1.0) > z(1.1)
        assert z(0.0) < z(0.5) < z(1.0)
        # the basin floor is at zero between bank and dam
        assert abs(z(3.5)) < 1e-6
        # the crest stands dam_height above the floor
        assert z(6.0) == pytest.approx(0.5, abs=1e-6)
        assert z(6.0) > z(5.9) and z(6.0) > z(6.1)
        # below the dam the bed falls away at the given slope
        assert z(9.0) - z(8.0) == pytest.approx(-0.05, abs=1e-9)

    def test_accepts_a_scalar(self):
        assert np.ndim(topo.get_bed_profile('basin_dam')(2.0)) == 0

    @pytest.mark.parametrize('params,match', [
        ({'bank_width': 0.0}, 'bank_width'),
        ({'dam_width': -1.0}, 'dam_width'),
        ({'bank_position': 6.0, 'dam_position': 6.0}, 'dam_position'),
    ])
    def test_meaningless_geometry_is_rejected(self, params, match):
        with pytest.raises(ValueError, match=match):
            topo.get_bed_profile('basin_dam', **params)

    def test_an_unknown_parameter_is_rejected(self):
        with pytest.raises(TypeError, match='basin_dam'):
            topo.get_bed_profile('basin_dam', dam_hieght=0.5)


# ---------------------------------------------------------------------------
# initial condition
# ---------------------------------------------------------------------------

def _model(**settings):
    return pde.SWME1D(
        'lakeWithDryShores', 0.0, 1.0, True, False,
        topography=topo.TopographySettings(
            bed_elevation=topo.get_bed_profile('basin_dam'), **settings))


class TestLakeWithDryShores:

    def test_depth_is_level_minus_bed_where_that_is_positive(self):
        model = _model(reference_water_level=0.3)
        bed = topo.get_bed_profile('basin_dam')
        for x in (2.5, 3.0, 4.0, 5.0):
            values = model.get_initial_values(2, 'lakeWithDryShores', x)
            assert values[0] == pytest.approx(0.3 - float(bed(x)))
            assert np.all(values[1:] == 0.0)

    def test_ground_above_the_level_is_exactly_dry(self):
        model = _model(reference_water_level=0.3)
        for x in (0.8, 1.0, 6.0):          # bank, bank crest, dam crest
            values = model.get_initial_values(1, 'lakeWithDryShores', x)
            assert np.all(values == 0.0)

    def test_without_an_extent_the_valley_below_the_dam_floods(self):
        # at x = 10 the bed is at -0.2, below the lake level
        model = _model(reference_water_level=0.3)
        assert model.get_initial_values(0, 'lakeWithDryShores', 10.0)[0] == \
            pytest.approx(0.5, abs=1e-6)

    def test_the_extent_keeps_the_lake_behind_the_dam(self):
        model = _model(reference_water_level=0.3, lake_left=1.0, lake_right=6.0)
        assert np.all(model.get_initial_values(0, 'lakeWithDryShores', 10.0) == 0.0)
        assert model.get_initial_values(0, 'lakeWithDryShores', 4.0)[0] > 0.29

    def test_an_empty_extent_is_rejected(self):
        with pytest.raises(ValueError, match='lake_left'):
            topo.TopographySettings(lake_left=2.0, lake_right=2.0)

    def test_lake_at_rest_still_refuses_dry_cells_and_points_here(self):
        model = _model(reference_water_level=0.3)
        with pytest.raises(ValueError) as error:
            model.get_initial_values(0, 'lakeAtRest', 1.0)
        message = str(error.value)
        assert 'non-positive water height' in message
        assert 'lakeWithDryShores' in message
        assert 'not supported yet' not in message

    def test_the_recharge_model_inherits_it(self):
        from recharge.initial_conditions import RechargeSWME1D_CustomIC
        from recharge.laws import AdmissibleMixingFriction, ConstantInfiltration
        model = RechargeSWME1D_CustomIC(
            'lakeWithDryShores', 0.0, 1.0, True, False, 0.0,
            ConstantInfiltration(I0=0.0),
            AdmissibleMixingFriction(alpha_R=1.0, alpha_I=1.0),
            topography=topo.TopographySettings(
                bed_elevation=topo.get_bed_profile('basin_dam'),
                reference_water_level=0.3, lake_left=1.0, lake_right=6.0))
        assert model.get_initial_values(1, 'lakeWithDryShores', 4.0)[0] > 0.29
        assert np.all(model.get_initial_values(1, 'lakeWithDryShores', 8.0) == 0.0)


# ---------------------------------------------------------------------------
# the shipped case, shortened
# ---------------------------------------------------------------------------

CONFIG = 'rain_fills_lake_overtops_dam'


def _config(**overrides):
    """The shipped config at half resolution, with section.key overrides."""
    config = cli.load_config(cli.resolve_config(CONFIG))
    config['grid']['resolution_x'] = 100
    for dotted, value in overrides.items():
        section, key = dotted.split('__')
        config[section][key] = value
    return config


def _without_infiltration(config):
    for key in ('horton_f0', 'horton_fc', 'horton_k'):
        del config['pde'][key]
    config['pde']['infiltration_type'] = 'constant'
    config['pde']['constant_infiltration_rate'] = 0.0
    return config


def _lake_level(sim, entry, position=4.0):
    x = entry['data'][:, 0]
    i = int(np.argmin(np.abs(x - position)))
    return entry['data'][i, 1] + sim.mesh.bed_elevation[1:-1][i]


def _budget(sim):
    return pd.DataFrame(sim.mass_budget)


class TestShippedConfig:

    def test_it_is_shipped_and_loads(self):
        assert CONFIG in cli.shipped_config_names()
        sim = cli.build_simulation(cli.load_config(cli.resolve_config(CONFIG)))
        assert sim.initial_condition == 'lakeWithDryShores'
        assert sim.mesh.has_topography
        assert sim.store_mass_budget
        assert sim.pde_type.wet_dry.shoreline_treatment
        # the default, ponded, Horton closure
        assert sim.pde_type.infiltration_model.limit_by_rainfall is False

    def test_initial_state_is_a_lake_with_dry_banks_and_a_dry_valley(self):
        sim = cli.build_simulation(_config())
        _run(sim, 1e-9)
        initial = sim.history[0]['data']
        x, h = initial[:, 0], initial[:, 1]
        assert np.all(h[x < 1.5] == 0.0)             # outer slope and ridge
        assert np.all(h[x > 6.0] == 0.0)             # dam face and valley
        assert h[np.argmin(np.abs(x - 4.0))] == pytest.approx(0.4, abs=1e-3)

    def test_without_rain_or_infiltration_the_lake_stays_at_rest(self):
        """The initial state, the bed and the shoreline treatment together:
        with the forcing switched off nothing may move."""
        config = _without_infiltration(_config(pde__rainfall_rate=0.0))
        sim = cli.build_simulation(config)
        final = _run(sim, 2.0)
        initial = sim.history[0]['data']
        assert np.max(np.abs(final[:, 1] - initial[:, 1])) < 1e-14
        assert np.max(np.abs(final[:, 2:])) < 1e-13
        assert np.max(np.abs(_budget(sim)['residual'])) < 1e-14

    def test_negative_control_the_lake_needs_the_shoreline_treatment(self):
        config = _without_infiltration(_config(pde__rainfall_rate=0.0))
        config['wet_dry']['shoreline_treatment'] = False
        sim = cli.build_simulation(config)
        with pytest.raises(RuntimeError, match='Negative height'):
            _run(sim, 2.0)

    def test_short_run_closes_the_mass_budget(self):
        """Mass changes only by rain, infiltration and the two boundary
        fluxes: the residual of that balance stays at round-off."""
        sim = cli.build_simulation(_config())
        final = _run(sim, 4.0)
        budget = _budget(sim)

        assert np.min(final[:, 1]) >= 0.0
        assert sim.mass_created_by_clamping < 1e-15
        assert np.max(np.abs(budget['residual'])) < 1e-12

        # rain is uniform, so its volume is known in closed form; what the
        # sources added on top of it is minus the infiltrated volume
        time = budget['time'].iloc[-1]
        rain = 0.012 * time * 10.0
        infiltration = rain - budget['cumulative_source'].iloc[-1]
        assert 0.0 < infiltration < rain + 0.4 * 10.0
        # nothing has reached the outflow this early
        assert abs(budget['cumulative_outflow_right'].iloc[-1]) < 1e-3

    def test_the_lake_falls_until_the_ponding_time_then_rises(self):
        """Under the ponded Horton closure the lake itself infiltrates at the
        full capacity f_p(t), so its level falls while f_p > R and turns
        round at t_p = ln((f0 - fc)/(R - fc))/k = 2.357. In the middle of the
        lake there is no flow yet, so the depth there follows the explicit
        Euler sum of R - f_p(t_n)."""
        sim = cli.build_simulation(_config())
        _run(sim, 4.0)
        R, f0, fc, k = 0.012, 0.03, 0.004, 0.5
        t_p = np.log((f0 - fc) / (R - fc)) / k
        assert t_p == pytest.approx(2.357, abs=1e-3)

        times = np.array([entry['time'] for entry in sim.history])
        levels = np.array([_lake_level(sim, entry) for entry in sim.history])
        turning = times[np.argmin(levels)]
        assert turning == pytest.approx(t_p, abs=2.5 * np.max(np.diff(times)))
        assert levels[-1] > levels.min()

        capacity = fc + (f0 - fc) * np.exp(-k * times[:-1])
        euler = 0.4 + np.concatenate(
            [[0.0], np.cumsum((R - capacity) * np.diff(times))])
        before = times <= t_p
        np.testing.assert_allclose(levels[before], euler[before], atol=2e-4)

    def test_rain_and_infiltration_decide_how_fast_the_lake_fills(self):
        """Same rain, three soils. Without infiltration the lake only rises;
        the stronger the infiltration the lower it stands at the same time."""
        levels = {}
        for name, config in (
                ('none', _without_infiltration(_config())),
                ('base', _config()),
                ('strong', _config(pde__horton_f0=0.06, pde__horton_fc=0.008))):
            sim = cli.build_simulation(config)
            _run(sim, 4.0)
            levels[name] = _lake_level(sim, sim.history[-1])
        assert levels['none'] > 0.4 > levels['strong']
        assert levels['none'] > levels['base'] > levels['strong']


# ---------------------------------------------------------------------------
# mass budget diagnostic
# ---------------------------------------------------------------------------

class TestMassBudget:

    def test_it_is_off_by_default_and_writes_nothing(self, tmp_path):
        config = cli.load_config(cli.resolve_config('wetdry_dam_break'))
        sim = cli.build_simulation(config)
        assert sim.store_mass_budget is False
        _run(sim, 0.02)
        assert sim.mass_budget == []

    def test_the_key_is_validated(self):
        config = cli.load_config(cli.resolve_config('wetdry_dam_break'))
        config['postprocessing'] = {'store_mass_budget': 'yes'}
        with pytest.raises(ValueError, match='store_mass_budget'):
            cli.build_simulation(config)

    def test_boundary_fluxes_account_for_an_open_dam_break(self):
        """Dam break with zero-gradient ends, run until the waves have left
        through both: the mass lost equals the recorded boundary fluxes."""
        config = cli.load_config(cli.resolve_config('config'))
        config['pde'] = {'type': 'SWME1D',
                         'initial_condition': 'damBreak_noVelocity',
                         'viscosity': 0.0, 'slip_length': 1.0,
                         'linear_source': False, 'hyperbolic': False}
        config['grid'] = {'x1': -1.0, 'x2': 1.0, 'resolution_x': 80}
        config['numerics'].update(order=1, t_end=1.5, pvm='Roe',
                                  time_integrator='ExplicitEuler',
                                  boundary_condition='INFLOW_OUTFLOW')
        config.pop('topography', None)
        config['postprocessing'] = {'store_mass_budget': True}
        sim = cli.build_simulation(config)
        _run(sim, 1.5)
        budget = _budget(sim)
        change = budget['mass'].iloc[-1] - budget['mass'].iloc[0]
        through_ends = (budget['cumulative_inflow_left'].iloc[-1]
                        - budget['cumulative_outflow_right'].iloc[-1])
        assert abs(change) > 1e-2                  # water really did leave
        assert change == pytest.approx(through_ends, abs=1e-13)
        assert np.all(budget['cumulative_source'] == 0.0)
        assert np.max(np.abs(budget['residual'])) < 1e-13

    def test_the_file_is_written_when_asked_for(self, tmp_path):
        config = _config(numerics__t_end=0.2)
        sim = cli.build_simulation(config)
        data = _run(sim, 0.2)
        prefix = str(tmp_path / 'case')
        metadata = cli.build_run_metadata(sim, config, config_name=CONFIG)
        cli.write_outputs(sim, data, str(tmp_path), prefix, 1, metadata)
        written = pd.read_csv(f'{prefix}_mass_budget.csv')
        assert list(written.columns) == [
            'step', 'time', 'mass', 'cumulative_source',
            'cumulative_inflow_left', 'cumulative_outflow_right',
            'cumulative_clamping', 'residual']
        assert len(written) == len(sim.history)
        assert metadata['store_mass_budget'] is True
        assert metadata['initial_condition'] == 'lakeWithDryShores'
        assert metadata['bed_profile'] == 'basin_dam'
        assert (metadata['lake_left'], metadata['lake_right']) == (1.0, 6.0)
        assert metadata['wet_dry']['shoreline_treatment'] is True
