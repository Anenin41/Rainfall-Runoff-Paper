"""`uv run purge` — see what is in results/, and remove what you no longer want.

Replaces the old `src/swme/Makefile`, whose entire body was three `rm -rf`
globs against `Data-processing/Results/Recharge/`. That rule was broken as
written (its paths were CWD-relative, so running it from the directory it lived
in targeted a path that never existed) and it pointed at a directory the solver
no longer writes to.

More to the point, it was the wrong shape for the job. Its purpose was to keep
track of which runs were worth keeping: every run wrote the *same* filenames, so
clearing the output directory was the only way to tell a fresh result from a
stale one. That is a curation problem, and the answer to it is to *show* what is
there, not to delete everything unconditionally.

So: listing is the default, deleting needs an explicit flag, and deleting
without `--yes` asks first. The underlying collision is also gone at the source
— runs now default to `results/<config-name>/` rather than sharing one
directory (see `swme.cli.run`).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

DEFAULT_ROOT = Path('results')


def _directory_size(path: Path) -> tuple[int, int]:
    """(total bytes, file count) for everything under `path`."""
    total = count = 0
    for item in path.rglob('*'):
        if item.is_file():
            total += item.stat().st_size
            count += 1
    return total, count


def _human(size: int) -> str:
    value = float(size)
    for unit in ('B', 'K', 'M', 'G'):
        if value < 1024 or unit == 'G':
            return f"{value:.0f}{unit}" if unit == 'B' else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}G"


def find_runs(root: Path) -> list[Path]:
    """Directories directly under `root` that look like run output.

    One level deep, plus one more where a case family groups its orders (as
    `run_thesis_configs.sh` lays them out, e.g. `Smooth_Pulse/Smooth_Pulse_N1`).
    A directory holding CSVs is a run; a directory holding only directories is a
    family and is descended into.
    """
    runs = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(child.glob('*.csv')):
            runs.append(child)
        else:
            nested = [p for p in sorted(child.iterdir()) if p.is_dir()]
            runs.extend(nested or [child])
    return runs


def list_runs(root: Path) -> int:
    if not root.is_dir():
        print(f"No such directory: {root}")
        return 1

    runs = find_runs(root)
    if not runs:
        print(f"{root}/ contains no run output.")
        return 0

    rows = []
    for run in runs:
        size, count = _directory_size(run)
        newest = max((p.stat().st_mtime for p in run.rglob('*') if p.is_file()),
                     default=run.stat().st_mtime)
        rows.append((run, size, count, newest))

    width = max(len(str(r[0].relative_to(root))) for r in rows)
    print(f"{'run':<{width}}  {'size':>8}  {'files':>6}  last modified")
    print('-' * (width + 34))
    for run, size, count, newest in sorted(rows, key=lambda r: -r[3]):
        stamp = time.strftime('%Y-%m-%d %H:%M', time.localtime(newest))
        print(f"{str(run.relative_to(root)):<{width}}  {_human(size):>8}  "
              f"{count:>6}  {stamp}")

    total = sum(r[1] for r in rows)
    print('-' * (width + 34))
    print(f"{len(rows)} run(s), {_human(total)} total in {root}/")
    return 0


def delete_runs(root: Path, patterns: list[str], assume_yes: bool) -> int:
    if not root.is_dir():
        print(f"No such directory: {root}")
        return 1

    runs = find_runs(root)
    matched = [r for r in runs
               if any(r.match(p) or p in str(r.relative_to(root)) for p in patterns)]
    if not matched:
        print(f"Nothing under {root}/ matches: {', '.join(patterns)}")
        return 1

    print("About to delete:")
    total = 0
    for run in matched:
        size, count = _directory_size(run)
        total += size
        print(f"  {run.relative_to(root)}  ({_human(size)}, {count} files)")
    print(f"  = {_human(total)} across {len(matched)} run(s)")

    if not assume_yes:
        try:
            answer = input("Delete these? [y/N] ").strip().lower()
        except EOFError:
            answer = ''
        if answer not in ('y', 'yes'):
            print("Aborted; nothing deleted.")
            return 1

    for run in matched:
        shutil.rmtree(run)
        print(f"  removed {run.relative_to(root)}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='purge',
        description='Inspect and clean solver output under results/. Lists by '
                    'default; deleting requires --delete and confirms first.')
    parser.add_argument('patterns', nargs='*',
                        help='Run names to act on (substring or glob). '
                             'Required with --delete.')
    parser.add_argument('-r', '--root', default=str(DEFAULT_ROOT),
                        help=f'Output directory to inspect (default: {DEFAULT_ROOT}).')
    parser.add_argument('--delete', action='store_true',
                        help='Delete the matching runs instead of listing them.')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip the confirmation prompt (for scripts).')
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not args.delete:
        return list_runs(root)
    if not args.patterns:
        parser.error("--delete needs at least one run name; "
                     "run without --delete to see what is there.")
    return delete_runs(root, args.patterns, args.yes)


if __name__ == '__main__':
    sys.exit(main())
