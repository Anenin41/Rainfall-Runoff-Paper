"""The short command names (`swme.commands`) and their long aliases."""

import sys
import tomllib
from pathlib import Path

import pytest

from swme import commands

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _scripts():
    with open(PYPROJECT, "rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def test_every_short_name_and_alias_is_installed():
    scripts = _scripts()
    for short in ("experiment", "report", "visual", "docs", "help"):
        assert scripts[short] == f"swme.commands:{short}"
    assert scripts["moment-sw"] == "swme.cli:main"
    assert scripts["moment-sw-report"] == "swme.report.cli:main"
    assert scripts["moment-sw-view"] == "swme.viewer.cli:main"


@pytest.mark.parametrize("short", ["experiment", "report", "visual"])
def test_a_short_name_without_arguments_prints_its_help(short, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [f"/venv/bin/{short}"])
    with pytest.raises(SystemExit) as stop:
        getattr(commands, short)()
    assert stop.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: {short} ")


def test_a_short_name_passes_its_arguments_on(monkeypatch, capsys):
    """Negative control for the test above: with an argument the command
    runs, it does not print help."""
    monkeypatch.setattr(sys, "argv", ["/venv/bin/experiment", "--list-configs"])
    assert commands.experiment() is None
    out = capsys.readouterr().out
    assert "usage:" not in out
    assert "rain_fills_lake_overtops_dam" in out.split()


def test_the_long_alias_keeps_its_name_in_the_usage_line(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["/venv/bin/moment-sw-view"])
    assert commands.invoked_as("moment-sw-view") == "moment-sw-view"
    monkeypatch.setattr(sys, "argv", ["/somewhere/pytest"])
    assert commands.invoked_as("moment-sw-view") == "moment-sw-view"


def test_help_lists_every_short_name(capsys):
    assert commands.help() == 0
    out = capsys.readouterr().out
    for short in ("experiment", "report", "visual", "docs", "purge", "help"):
        assert f"uv run {short}" in out
