"""`moment-sw-report` - build a PDF report from a run directory.

Separate from `moment-sw --report` so the 20 thesis runs already sitting in
`results/` can be reported on without re-running anything, which for the larger
cases is hours of compute.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..purge import find_runs
from .data import from_directory
from .history import DEFAULT_CHUNK_ROWS, DEFAULT_MAX_SNAPSHOTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='moment-sw-report',
        description='Render a multi-page PDF report from a run directory '
                    'written by moment-sw.',
    )
    parser.add_argument(
        'directory', nargs='?', default='results',
        help='A run directory, or a tree of them to report on in bulk '
             '(default: results/).')
    parser.add_argument(
        '-o', '--output', default=None,
        help='Output PDF. Defaults to report.pdf inside each run directory.')
    parser.add_argument(
        '--prefix', default=None,
        help='Run filename stem, when a directory holds more than one run.')
    parser.add_argument(
        '--max-snapshots', type=int, default=DEFAULT_MAX_SNAPSHOTS,
        help=f'Field-history snapshots to keep (default: {DEFAULT_MAX_SNAPSHOTS}). '
             'Bounds both memory and the resolution of the space-time maps.')
    parser.add_argument(
        '--chunk-rows', type=int, default=DEFAULT_CHUNK_ROWS,
        help=f'Rows per read chunk (default: {DEFAULT_CHUNK_ROWS}). Bounds peak '
             'memory; does not change the result.')
    parser.add_argument(
        '--all', action='store_true',
        help='Treat the directory as a tree and report on every run beneath it.')
    return parser


def _report_one(directory: Path, output, args) -> int:
    from .assemble import build_report

    try:
        run = from_directory(directory, prefix=args.prefix,
                             max_snapshots=args.max_snapshots,
                             chunk_rows=args.chunk_rows)
    except (FileNotFoundError, ValueError) as error:
        print(f"{directory}: {error}", file=sys.stderr)
        return 1

    result = build_report(run, output or directory / 'report.pdf')
    print(f"{result.path}  ({len(result.page_keys)} pages: "
          f"{', '.join(result.page_keys)})")
    for warning in result.warnings:
        print(f"  note: {warning}")
    if result.skipped_keys:
        print(f"  skipped: {', '.join(result.skipped_keys)}")
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.directory)

    if not root.is_dir():
        print(f"No such directory: {root}", file=sys.stderr)
        return 1

    # A directory holding CSVs is itself a run; otherwise treat it as a tree.
    if args.all or not any(root.glob('*_final.csv')):
        # `find_runs` also surfaces leaf directories that hold no CSVs at all -
        # a figures folder, say. Those are not failures to report on, so they
        # are filtered out here rather than counted against the exit status.
        runs = [run for run in find_runs(root) if any(run.glob('*_final.csv'))]
        if not runs:
            print(f"{root}/ contains no run output.", file=sys.stderr)
            return 1
        if args.output and len(runs) > 1:
            print("--output cannot be combined with multiple runs.",
                  file=sys.stderr)
            return 1
        failures = sum(_report_one(run, args.output, args) for run in runs)
        if failures:
            print(f"{failures} of {len(runs)} runs failed.", file=sys.stderr)
        return 1 if failures else 0

    return _report_one(root, Path(args.output) if args.output else None, args)


if __name__ == '__main__':                                  # pragma: no cover
    raise SystemExit(main())
