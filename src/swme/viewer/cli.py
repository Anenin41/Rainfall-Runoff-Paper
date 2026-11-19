"""`uv run visual` (alias `moment-sw-view`) - browse run output in a local web app.

    uv run visual results                 # every run under results/
    uv run visual results/5p3             # one family, or one run directory
    uv run moment-sw-view                 # the alias: no argument means results/

Binds to 127.0.0.1 unless told otherwise: the viewer reads whatever is under
the results root, and nothing about that should be on a network by default.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import webbrowser
from pathlib import Path

from ..commands import invoked_as
from ..report.history import DEFAULT_CHUNK_ROWS
from .store import DEFAULT_CAPACITY, DEFAULT_MAX_SNAPSHOTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=invoked_as("moment-sw-view"),
        description="Browse moment-sw run output interactively in the browser.")
    parser.add_argument(
        "directory", nargs="?", default="results",
        help="A results tree, a family directory, or a single run (default: results/).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1, this machine only).")
    parser.add_argument("--port", type=int, default=8050, help="Port (default: 8050).")
    parser.add_argument(
        "--max-snapshots", type=int, default=DEFAULT_MAX_SNAPSHOTS,
        help=f"Stored steps kept per run for the time slider and maps "
             f"(default: {DEFAULT_MAX_SNAPSHOTS}). Bounds memory per run.")
    parser.add_argument(
        "--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS,
        help=f"Rows per read chunk when streaming field history "
             f"(default: {DEFAULT_CHUNK_ROWS}).")
    parser.add_argument(
        "--cache", type=int, default=DEFAULT_CAPACITY,
        help=f"Runs kept loaded at once (default: {DEFAULT_CAPACITY}).")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser tab on start.")
    parser.add_argument("--debug", action="store_true",
                        help="Dash debug mode: error overlay and hot reload. Also keeps "
                             "the server's log of every request, which is otherwise off.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    root = Path(args.directory)
    if not root.is_dir():
        print(f"No such directory: {root}", file=sys.stderr)
        return 1

    from .app import create_app
    from .store import RunStore

    store = RunStore(root, max_snapshots=args.max_snapshots, chunk_rows=args.chunk_rows,
                     capacity=args.cache)
    if not store.entries():
        print(f"{root}/ contains no run output (no *_final.csv found).", file=sys.stderr)
        return 1
    print(f"moment-sw-view: {len(store.entries())} run(s) under {root.resolve()}")

    app = create_app(store=store)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}/"
    print(f"Serving on {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    if not args.debug:
        # Werkzeug logs every request at INFO, several a second while playing,
        # and its development-server notice: this server only ever binds to
        # this machine by default. Errors still show.
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        app.logger.setLevel(logging.WARNING)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":                                  # pragma: no cover
    raise SystemExit(main())
