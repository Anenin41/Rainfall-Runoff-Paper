"""The short command names: `uv run experiment`, `report`, `visual`, `docs`, `help`.

Each short name runs the same code as its long alias (`moment-sw`,
`moment-sw-report`, `moment-sw-view`, `mkdocs`). The one difference is that a
short name given no argument at all prints its help instead of running with
defaults: `uv run experiment` would otherwise look for a `config.yaml` in the
working directory, and `uv run visual` would start a server. The long aliases
keep their old behaviour exactly, so scripts that call them are unaffected.

Bare `uv run` cannot be changed: it is uv's own command, and it lists every
executable in the environment. `uv run help` is the overview of this project's.

This module imports nothing heavy at load time: `uv run help` should not pay
for numpy, Dash or matplotlib.
"""

from __future__ import annotations

import sys
from pathlib import Path

# short name -> long alias, for `help` and for `invoked_as`
ALIASES = {
    "experiment": "moment-sw",
    "report": "moment-sw-report",
    "visual": "moment-sw-view",
    "docs": "mkdocs",
}

OVERVIEW = """\
moment-sw: Shallow Water Moment Equations, 1D finite volumes.

Commands (run from the repository root; each prints its own help when given
no argument, or with --help):

  uv run experiment -c <config> [-o <dir>] [--report]
                           run a simulation (alias: moment-sw)
  uv run experiment --list-configs
                           list the shipped configs
  uv run report <dir>      PDF report from a finished run (alias: moment-sw-report)
  uv run visual <dir>      browse finished runs in the browser; `uv run visual
                           results` for every run (alias: moment-sw-view)
  uv run docs serve        documentation site at http://127.0.0.1:8000/
  uv run docs build        build the site into site/ (add --strict to fail on warnings)
  uv run purge             list results/ runs; deleting needs a pattern
  uv run help              this overview

Tests are run with pytest as before: uv run pytest -q
"""


def invoked_as(default: str) -> str:
    """The program name for a parser's usage line: the name it was started
    under if that is one of this project's commands, otherwise `default`
    (for `python -m ...` or a test calling `main` directly)."""
    name = Path(sys.argv[0]).name
    return name if name in ALIASES or name in ALIASES.values() else default


def _arguments() -> list[str]:
    return sys.argv[1:] or ["--help"]


def experiment():
    from .cli import main
    return main(_arguments())


def report():
    from .report.cli import main
    return main(_arguments())


def visual():
    from .viewer.cli import main
    return main(_arguments())


def docs():
    try:
        from mkdocs.__main__ import cli
    except ImportError:
        print("docs: mkdocs is not installed. It is in the `docs` dependency group, "
              "which `uv sync` installs by default; run `uv sync --group docs`.",
              file=sys.stderr)
        return 1
    return cli.main(args=_arguments(), prog_name="docs")


def help():                                                # noqa: A001 - the command's name
    print(OVERVIEW, end="")
    return 0
