"""The YAML config layer and CLI dispatch (RESTRUCTURE_PLAN.md Step 7).

The point of moving off `configparser` was not the file format — it was that
INI silently absorbed mistakes. A misspelled key fell back to a default, an
unsupported `pvm` printed a line and carried on to a `NameError`, and every
value arrived as a string needing a `getfloat`/`getboolean` at each use site.
So most of what is worth testing here is that bad configs now *fail, loudly,
naming the problem* — and that the shipped configs still build exactly the
objects they used to.
"""

from __future__ import annotations

import textwrap

import pytest

from swme import cli
from swme import pde as pde_module
from swme import spatialDiscretization as sd
from swme import timeIntegration


def write(tmp_path, body, name="case.yaml"):
    """Write a config verbatim.

    Deliberately does NOT dedent: MINIMAL is dedented once at definition, so
    tests can concatenate extra (already flush-left) sections onto it without
    a second dedent flattening the indentation of the first block.
    """
    path = tmp_path / name
    path.write_text(body)
    return path


MINIMAL = textwrap.dedent("""
    pde:
      type: SWME1D
      initial_condition: damBreak_noVelocity
      viscosity: 0.0
      slip_length: 1.0
      linear_source: false
    grid:
      x1: -1.0
      x2: 1.0
      resolution_x: 20
    numerics:
      order: 1
      t_end: 0.05
      method: classical
      fvm_type: PVM
      pvm: Roe
      time_integrator: ExplicitEuler
      boundary_condition: INFLOW_OUTFLOW
""")


# ---------------------------------------------------------------------------
# the shipped configs
# ---------------------------------------------------------------------------

class TestShippedConfigs:

    def test_every_shipped_config_loads_and_builds(self):
        """Guards the whole INI->YAML migration: all 24 must still construct
        a complete solver, not merely parse."""
        names = cli.shipped_config_names()
        assert len(names) >= 24
        for name in names:
            config = cli.load_config(cli.resolve_config(name))
            topography = cli.build_topography(config)
            wet_dry = cli.build_wet_dry(config)
            cli.build_pde(config, topography, wet_dry)
            cli.build_scheme(config)
            cli.build_time_integration(config)
            cli.build_mesh(config, topography,
                           config['numerics']['boundary_condition'])

    def test_configs_resolve_by_bare_name_from_any_directory(self):
        assert cli.resolve_config('thesis_5p2_horton_at_rest').is_file()
        assert cli.resolve_config('thesis_5p2_horton_at_rest.yaml').is_file()

    def test_an_unknown_config_name_lists_what_is_available(self):
        with pytest.raises(FileNotFoundError, match="Available shipped configs"):
            cli.resolve_config('no_such_case')

    def test_thesis_config_values_survived_the_migration(self):
        """Spot-check against the thesis Table 2 parameters directly, so a
        transcription slip in the conversion could not pass silently."""
        config = cli.load_config(cli.resolve_config('thesis_5p2_horton_at_rest'))
        assert config['pde']['type'] == 'RechargeSWME1D'
        assert config['pde']['rainfall_rate'] == pytest.approx(1.0e-3)
        assert config['pde']['horton_f0'] == pytest.approx(1.2e-3)
        assert config['pde']['horton_fc'] == pytest.approx(7.0e-4)
        assert config['pde']['horton_k'] == pytest.approx(1.0e-3)
        assert config['numerics']['t_end'] == pytest.approx(1800.0)
        assert config['grid']['resolution_x'] == 10

    def test_numeric_values_are_parsed_as_numbers_not_strings(self):
        """YAML 1.1 resolves `1.0e3` (unsigned exponent) to a *string*. If any
        config picked that form up, the run would fail deep inside numpy."""
        for name in cli.shipped_config_names():
            config = cli.load_config(cli.resolve_config(name))
            for section in ('pde', 'grid', 'numerics', 'wet_dry'):
                for key, value in (config.get(section) or {}).items():
                    if isinstance(value, str):
                        assert not value.replace('.', '').replace('-', '') \
                            .replace('+', '').replace('e', '').isdigit(), (
                            f"{name}: {section}.{key} = {value!r} parsed as a "
                            "string; YAML needs a signed exponent")


# ---------------------------------------------------------------------------
# validation: the reason for the rewrite
# ---------------------------------------------------------------------------

class TestConfigValidation:

    def test_a_misspelled_key_is_rejected_not_ignored(self, tmp_path):
        """Under INI this silently used the default viscosity."""
        path = write(tmp_path, MINIMAL.replace('viscosity:', 'viscosty:'))
        with pytest.raises(ValueError, match="Unknown key.*viscosty"):
            cli.load_config(path)

    def test_an_unknown_section_is_rejected(self, tmp_path):
        path = write(tmp_path, MINIMAL + "\nnumercs:\n  order: 1\n")
        with pytest.raises(ValueError, match="Unknown section.*numercs"):
            cli.load_config(path)

    @pytest.mark.parametrize("section", ['pde', 'grid', 'numerics'])
    def test_missing_required_sections_are_named(self, tmp_path, section):
        body = MINIMAL.replace(f"{section}:", "unused_removed:")
        path = write(tmp_path, body)
        with pytest.raises(ValueError):
            cli.load_config(path)

    def test_a_missing_required_key_names_itself(self, tmp_path):
        path = write(tmp_path, MINIMAL.replace('  viscosity: 0.0\n', ''))
        config = cli.load_config(path)
        with pytest.raises(ValueError, match="missing required key 'viscosity'"):
            cli.build_pde(config, cli.build_topography(config),
                          cli.build_wet_dry(config))

    def test_an_unsupported_scheme_raises_instead_of_printing(self, tmp_path):
        """main.py printed 'this pvm method is not implemented yet' and then
        continued to a NameError on the undefined object."""
        path = write(tmp_path, MINIMAL.replace('pvm: Roe', 'pvm: Rusanov'))
        config = cli.load_config(path)
        with pytest.raises(ValueError, match="numerics.pvm.*Rusanov"):
            cli.build_scheme(config)

    def test_an_unsupported_model_raises_instead_of_printing(self, tmp_path):
        path = write(tmp_path, MINIMAL.replace('type: SWME1D', 'type: HermiteME'))
        config = cli.load_config(path)
        with pytest.raises(ValueError, match="pde.type.*HermiteME"):
            cli.build_pde(config, cli.build_topography(config),
                          cli.build_wet_dry(config))

    def test_a_quoted_boolean_is_rejected(self, tmp_path):
        """`linear_source: "false"` is a truthy string in YAML, which under a
        naive loader would silently enable the implicit source path."""
        path = write(tmp_path, MINIMAL.replace(
            'linear_source: false', 'linear_source: "false"'))
        config = cli.load_config(path)
        with pytest.raises(ValueError, match="must be true or false"):
            cli.build_pde(config, cli.build_topography(config),
                          cli.build_wet_dry(config))

    def test_an_empty_config_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="empty"):
            cli.load_config(write(tmp_path, "\n"))

    def test_linear_source_without_implicit_euler_is_rejected(self, tmp_path):
        """The linear source returns a *matrix*, which only ImplicitEuler
        applies. Pairing it with ExplicitEuler was previously a silent wrong
        answer, prevented only by how main.py happened to compose them."""
        path = write(tmp_path, MINIMAL.replace(
            'linear_source: false', 'linear_source: true'))
        config = cli.load_config(path)
        with pytest.raises(ValueError, match="only ImplicitEuler"):
            cli.build_pde(config, cli.build_topography(config),
                          cli.build_wet_dry(config))

    def test_hswme_type_conflicting_with_hyperbolic_flag_is_rejected(self, tmp_path):
        path = write(tmp_path, MINIMAL.replace(
            'type: SWME1D', 'type: HSWME1D\n  hyperbolic: false'))
        config = cli.load_config(path)
        with pytest.raises(ValueError, match="conflict"):
            cli.build_pde(config, cli.build_topography(config),
                          cli.build_wet_dry(config))


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

class TestDispatch:

    @pytest.mark.parametrize("name,cls", [
        ('Roe', sd.Roe), ('Osher', sd.Osher), ('LF', sd.LF), ('PRICE', sd.PRICE)])
    def test_every_scheme_is_reachable(self, tmp_path, name, cls):
        config = cli.load_config(write(tmp_path, MINIMAL.replace('pvm: Roe', f'pvm: {name}')))
        assert isinstance(cli.build_scheme(config), cls)

    @pytest.mark.parametrize("name,cls", [
        ('ExplicitEuler', timeIntegration.ExplicitEuler),
        ('Exact', timeIntegration.Exact)])
    def test_time_integrators_are_reachable(self, tmp_path, name, cls):
        config = cli.load_config(write(tmp_path, MINIMAL.replace(
            'time_integrator: ExplicitEuler', f'time_integrator: {name}')))
        assert isinstance(cli.build_time_integration(config), cls)

    def test_implicit_euler_pairs_with_the_linear_source(self, tmp_path):
        config = cli.load_config(write(tmp_path, MINIMAL
            .replace('linear_source: false', 'linear_source: true')
            .replace('time_integrator: ExplicitEuler',
                     'time_integrator: ImplicitEuler')))
        assert isinstance(cli.build_time_integration(config),
                          timeIntegration.ImplicitEuler)
        cli.build_pde(config, cli.build_topography(config), cli.build_wet_dry(config))

    def test_hswme_selects_the_hyperbolic_closure(self, tmp_path):
        config = cli.load_config(write(tmp_path, MINIMAL.replace(
            'type: SWME1D', 'type: HSWME1D')))
        built = cli.build_pde(config, cli.build_topography(config),
                              cli.build_wet_dry(config))
        assert isinstance(built, pde_module.SWME1D)
        assert built.hyperbolic is True

    def test_optional_sections_default_cleanly_when_absent(self, tmp_path):
        """A config with no topography/wet_dry must reproduce pre-Step-5/6
        behaviour: flat bed, default thresholds, non-augmented path."""
        config = cli.load_config(write(tmp_path, MINIMAL))
        topography = cli.build_topography(config)
        assert topography.bed_elevation is None
        mesh = cli.build_mesh(config, topography, 'INFLOW_OUTFLOW')
        assert mesh.has_topography is False
        thresholds = cli.build_wet_dry(config)
        assert (thresholds.h_dry, thresholds.h_wet) == (1e-4, 1e-3)

    def test_topography_section_is_wired_through_to_the_mesh(self, tmp_path):
        config = cli.load_config(write(tmp_path, MINIMAL + textwrap.dedent("""
            topography:
              bed_profile: gaussian_bump
              amplitude: 0.3
              center: 0.0
              width: 0.2
              reference_water_level: 2.0
        """)))
        topography = cli.build_topography(config)
        mesh = cli.build_mesh(config, topography, 'INFLOW_OUTFLOW')
        assert mesh.has_topography is True
        assert 0.25 < mesh.bed_elevation.max() <= 0.3

    def test_a_typo_in_a_bed_profile_parameter_is_rejected(self, tmp_path):
        config = cli.load_config(write(tmp_path, MINIMAL + textwrap.dedent("""
            topography:
              bed_profile: gaussian_bump
              amplitud: 0.3
        """)))
        with pytest.raises(TypeError, match="Bad parameters"):
            cli.build_topography(config)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_a_run_writes_csvs_for_a_non_recharge_model(self, tmp_path, capsys):
        """Regression for the defect where all CSV output sat inside the
        RechargeSWME1D branch, so SWME1D/HSWME1D runs produced nothing."""
        path = write(tmp_path, MINIMAL)
        cli.run(str(path), output_dir=str(tmp_path / 'out'))
        written = sorted(p.name for p in (tmp_path / 'out').glob('*.csv'))
        assert written == [
            'swme_N1_field_history.csv',
            'swme_N1_final.csv',
            'swme_N1_summary_history.csv',
        ]

    def test_recharge_output_filenames_are_unchanged(self, tmp_path):
        """processing/*.py matches these names exactly; they must not drift."""
        assert cli._output_prefix('/o', 'RechargeSWME1D', False, 1, 'horton') \
            == '/o/recharge_swme_N1_horton'
        assert cli._output_prefix('/o', 'RechargeSWME1D', True, 2, 'constant') \
            == '/o/recharge_hswme_N2_constant'
        assert cli._output_prefix('/o', 'SWME1D', False, 1, None) == '/o/swme_N1'

    def test_main_returns_none_so_the_console_script_exits_zero(self, tmp_path):
        """sys.exit(main()) with a non-int prints the object and exits 1 —
        which looks exactly like a crash even when the run succeeded."""
        path = write(tmp_path, MINIMAL)
        assert cli.main(['--config', str(path),
                         '--output-dir', str(tmp_path / 'out')]) is None

    def test_run_returns_the_final_state(self, tmp_path):
        path = write(tmp_path, MINIMAL)
        data = cli.run(str(path), output_dir=str(tmp_path / 'out'))
        assert data.shape == (20, 4)          # 20 cells, [x, h, u_m, a1]

    def test_list_configs_prints_and_exits_cleanly(self, capsys):
        assert cli.main(['--list-configs']) is None
        printed = capsys.readouterr().out.split()
        assert 'thesis_5p2_horton_at_rest' in printed
        assert 'wetdry_dam_break' in printed
