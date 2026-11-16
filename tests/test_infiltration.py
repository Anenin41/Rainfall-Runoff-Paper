"""Horton infiltration: the ponded closure and the legacy rainfall-limited one.

The realised rate is the capacity capped by the water supplied to the bed. The
supply is the rain plus the standing water, so over one source step

    ponded (default)          I = min(f_p(t), R + h/dt)
    limit_by_rainfall=True    I = min(f_p(t), R, h/dt)      (thesis, legacy)

The legacy closure withholds standing water from the ground; it is kept only
so the thesis configs reproduce their published figures. These tests pin both
closures, the config key that selects between them, and that every shipped
thesis Horton config still selects the legacy one.
"""

from __future__ import annotations

import numpy as np
import pytest

from recharge.laws import HortonInfiltration
from swme import cli

F0, FC, K = 0.24, 0.05, 3.0


def capacity(t):
    return FC + (F0 - FC) * np.exp(-K * t)


@pytest.fixture
def ponded():
    return HortonInfiltration(F0, FC, K)


@pytest.fixture
def legacy():
    return HortonInfiltration(F0, FC, K, limit_by_rainfall=True)


class TestPondedClosure:

    def test_is_the_default(self, ponded):
        assert ponded.limit_by_rainfall is False

    def test_standing_water_infiltrates_at_full_capacity_above_the_rain(self, ponded):
        # f_p(0) = 0.24 > R = 0.2: the case the legacy closure gets wrong
        I = ponded.rate(t=0.0, rainfall=0.2, h=1.0, dt=1e-3)
        assert I == pytest.approx(F0, rel=0, abs=0)
        assert I > 0.2

    def test_follows_the_capacity_on_a_ponded_bed(self, ponded):
        t = np.linspace(0.0, 2.0, 41)
        I = ponded.rate(t=t, rainfall=0.2, h=np.ones_like(t), dt=1e-3)
        np.testing.assert_array_equal(I, capacity(t))

    def test_dry_bed_is_capped_by_the_rain(self, ponded):
        # Before any water stands on the bed, the supply is the rain alone
        assert ponded.rate(t=0.0, rainfall=0.2, h=0.0, dt=1e-3) == 0.2
        # ... and once f_p < R the capacity binds instead
        assert ponded.rate(t=1.0, rainfall=0.2, h=0.0, dt=1e-3) == pytest.approx(capacity(1.0))

    def test_shallow_water_is_capped_by_the_supply(self, ponded):
        # R + h/dt = 0.1 + 0.05 = 0.15 < f_p(0) = 0.24
        h, dt = 5e-5, 1e-3
        assert ponded.rate(t=0.0, rainfall=0.1, h=h, dt=dt) == pytest.approx(0.1 + h / dt)

    def test_one_step_never_removes_more_water_than_is_present(self, ponded):
        rng = np.random.default_rng(0)
        h = rng.uniform(0.0, 1e-3, 500)
        R = rng.uniform(0.0, 0.3, 500)
        dt = 1e-2
        I = ponded.rate(t=0.0, rainfall=R, h=h, dt=dt)
        assert np.all(h + dt * (R - I) >= -1e-15)
        assert np.all(I >= 0.0)

    def test_negative_depth_supplies_nothing(self, ponded):
        assert ponded.rate(t=0.0, rainfall=0.1, h=-1.0, dt=1e-3) == 0.1

    def test_without_depth_information_reduces_to_min_of_capacity_and_rain(self, ponded):
        assert ponded.rate(t=0.0, rainfall=0.2) == 0.2
        assert ponded.infiltrated_depth(t=0.0, rainfall=0.2, dt=0.5) == pytest.approx(0.1)

    def test_scalar_in_scalar_out(self, ponded):
        assert isinstance(ponded.rate(t=0.0, rainfall=0.2, h=1.0, dt=1e-3), np.float64)


class TestLegacyClosure:

    def test_matches_the_thesis_formula(self, legacy):
        rng = np.random.default_rng(1)
        t = rng.uniform(0.0, 2.0, 500)
        R = rng.uniform(0.0, 0.3, 500)
        h = rng.uniform(0.0, 1e-3, 500)
        dt = 1e-2
        expected = np.maximum(np.minimum(np.minimum(capacity(t), R), h / dt), 0.0)
        np.testing.assert_array_equal(legacy.rate(t=t, rainfall=R, h=h, dt=dt), expected)

    def test_standing_water_never_infiltrates_above_the_rain(self, legacy):
        assert legacy.rate(t=0.0, rainfall=0.2, h=1.0, dt=1e-3) == 0.2

    def test_closures_agree_once_the_capacity_is_below_the_rain_on_a_deep_bed(self, ponded, legacy):
        t = np.linspace(1.0, 3.0, 11)          # f_p < R = 0.2 throughout
        kw = dict(rainfall=0.2, h=np.ones_like(t), dt=1e-3)
        np.testing.assert_array_equal(ponded.rate(t=t, **kw), legacy.rate(t=t, **kw))

    def test_repr_names_the_closure(self, legacy):
        assert "limit_by_rainfall=True" in repr(legacy)


class TestConfig:

    BASE = {'infiltration_type': 'horton',
            'horton_f0': F0, 'horton_fc': FC, 'horton_k': K}

    def test_key_defaults_to_the_ponded_closure(self):
        model = cli._build_infiltration_model(dict(self.BASE))
        assert model.limit_by_rainfall is False

    @pytest.mark.parametrize("flag", [True, False])
    def test_key_selects_the_closure(self, flag):
        model = cli._build_infiltration_model(
            dict(self.BASE, horton_limit_by_rainfall=flag))
        assert model.limit_by_rainfall is flag

    def test_key_must_be_a_boolean(self):
        with pytest.raises(ValueError, match="horton_limit_by_rainfall"):
            cli._build_infiltration_model(
                dict(self.BASE, horton_limit_by_rainfall="true"))

    def test_thesis_horton_configs_keep_the_legacy_closure(self):
        names = sorted(p.stem for p in cli._CONFIG_DIR.glob('thesis_*horton*.yaml'))
        assert len(names) == 10
        for name in names:
            pde_section = cli.load_config(cli.resolve_config(name))['pde']
            model = cli._build_infiltration_model(pde_section)
            assert model.limit_by_rainfall is True, name

    def test_smoke_test_1_uses_the_ponded_closure(self):
        pde_section = cli.load_config(cli.resolve_config('smoke_test_1'))['pde']
        assert cli._build_infiltration_model(pde_section).limit_by_rainfall is False
