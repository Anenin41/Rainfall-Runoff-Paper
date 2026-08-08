# Time Testing Symbolic Integration Files #
# Author: Konstantinos Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Thu 05 Mar 2026 @ 16:03:43 +0100
# Modified: Wed 25 Mar 2026 @ 21:30:22 +0100

#!/usr/bin/env python3
"""
time_test.py

Minimal benchmark suite for comparing the runtime of two Python scripts.

Purpose
-------
This file is meant for quick local testing while developing optimized code.
Instead of using command-line argument parsing, you configure the benchmark by
editing the constants in the CONFIGURATION section below.

How it works
------------
1. The "old" script and the "new" script are each executed in a fresh
   subprocess.
2. A few warmup runs are performed first, to reduce noise from startup effects.
3. Each script is then timed multiple times.
4. Summary statistics are printed:
   - minimum runtime
   - median runtime
   - mean runtime
   - population standard deviation
5. A speedup ratio is reported.

Why use subprocesses?
---------------------
Running each script in a fresh subprocess makes the comparison fairer.
This avoids issues with shared interpreter state, import caching, or global
variables influencing later runs.

What is being measured?
-----------------------
The total wall-clock runtime of the entire script, measured with
time.perf_counter().

Typical usage
-------------
Edit the configuration values below, then run:

    python3 time_test.py

If you want to benchmark scripts that need options, put them in SCRIPT_ARGS.

Example:
    SCRIPT_ARGS = ["--N", "5", "--A"]
"""

from __future__ import annotations

import statistics as stats
import subprocess
import sys
import time
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

OLD_SCRIPT = Path("integral_solver.py")
NEW_SCRIPT = Path("symbo.py")

# Arguments passed to both benchmarked scripts.
SCRIPT_ARGS: list[str] = ["--N", "3", "--all"]

# Number of untimed warmup runs for each script.
WARMUP_RUNS = 3

# Number of timed runs for each script.
TIMED_RUNS = 10

# If True, suppress stdout/stderr from the benchmarked scripts.
# This is usually better for timing consistency.
SUPPRESS_OUTPUT = False


# ============================================================================
# BENCHMARK HELPERS
# ============================================================================

def run_once(script: Path, extra_args: list[str]) -> float:
    """
    Execute one script once and return its runtime in seconds.

    Parameters
    ----------
    script:
        Path to the Python script that will be executed.
    extra_args:
        Extra command-line arguments passed to the script.

    Returns
    -------
    float
        Wall-clock execution time in seconds.

    Raises
    ------
    subprocess.CalledProcessError
        Raised if the target script exits with a nonzero return code.
    """
    cmd = [sys.executable, str(script), *extra_args]
    t0 = time.perf_counter()

    if SUPPRESS_OUTPUT:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.run(cmd, check=True)

    return time.perf_counter() - t0


def warmup(old_script: Path, new_script: Path, warmup_runs: int, extra_args: list[str]) -> None:
    """
    Perform untimed warmup runs for both scripts.

    Warmup helps reduce timing noise from first-run effects such as imports,
    bytecode generation, and OS-level file caching.

    Parameters
    ----------
    old_script:
        Path to the baseline script.
    new_script:
        Path to the optimized script.
    warmup_runs:
        Number of warmup runs for each script.
    extra_args:
        Extra command-line arguments passed to both scripts.
    """
    for _ in range(warmup_runs):
        run_once(old_script, extra_args)
        run_once(new_script, extra_args)


def benchmark(script: Path, timed_runs: int, extra_args: list[str]) -> list[float]:
    """
    Run one script multiple times and collect runtimes.

    Parameters
    ----------
    script:
        Path to the script to benchmark.
    timed_runs:
        Number of timed runs.
    extra_args:
        Extra command-line arguments passed to the script.

    Returns
    -------
    list[float]
        A list of runtimes in seconds.
    """
    return [run_once(script, extra_args) for _ in range(timed_runs)]


def summarize(times: list[float]) -> dict[str, float]:
    """
    Compute summary statistics for a list of runtimes.

    Parameters
    ----------
    times:
        Measured runtimes in seconds.

    Returns
    -------
    dict[str, float]
        Dictionary with keys:
        - min
        - median
        - mean
        - stdev
        - n
    """
    return {
        "min": min(times),
        "median": stats.median(times),
        "mean": stats.mean(times),
        "stdev": stats.pstdev(times) if len(times) > 1 else 0.0,
        "n": float(len(times)),
    }


def format_summary(label: str, summary: dict[str, float]) -> str:
    """
    Format timing statistics as a readable one-line string.

    Parameters
    ----------
    label:
        Name of the benchmark target, e.g. 'OLD' or 'NEW'.
    summary:
        Summary dictionary returned by summarize().

    Returns
    -------
    str
        Human-readable timing summary.
    """
    return (
        f"{label}: "
        f"min={summary['min']:.4f}s  "
        f"med={summary['median']:.4f}s  "
        f"mean={summary['mean']:.4f}s  "
        f"sd={summary['stdev']:.4f}s  "
        f"(n={int(summary['n'])})"
    )


def validate_paths() -> None:
    """
    Check that the configured benchmark target scripts exist.

    Raises
    ------
    FileNotFoundError
        If either OLD_SCRIPT or NEW_SCRIPT does not exist.
    """
    if not OLD_SCRIPT.exists():
        raise FileNotFoundError(f"Old script not found: {OLD_SCRIPT}")
    if not NEW_SCRIPT.exists():
        raise FileNotFoundError(f"New script not found: {NEW_SCRIPT}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Run the benchmark suite and print the comparison report.
    """
    validate_paths()

    print("Benchmark configuration")
    print("-----------------------")
    print(f"OLD_SCRIPT      = {OLD_SCRIPT}")
    print(f"NEW_SCRIPT      = {NEW_SCRIPT}")
    print(f"SCRIPT_ARGS     = {SCRIPT_ARGS}")
    print(f"WARMUP_RUNS     = {WARMUP_RUNS}")
    print(f"TIMED_RUNS      = {TIMED_RUNS}")
    print(f"SUPPRESS_OUTPUT = {SUPPRESS_OUTPUT}")
    print()

    warmup(OLD_SCRIPT, NEW_SCRIPT, WARMUP_RUNS, SCRIPT_ARGS)

    old_times = benchmark(OLD_SCRIPT, TIMED_RUNS, SCRIPT_ARGS)
    new_times = benchmark(NEW_SCRIPT, TIMED_RUNS, SCRIPT_ARGS)

    old_summary = summarize(old_times)
    new_summary = summarize(new_times)

    print(format_summary("OLD", old_summary))
    print(format_summary("NEW", new_summary))

    speedup_median = old_summary["median"] / new_summary["median"]
    speedup_mean = old_summary["mean"] / new_summary["mean"]

    print()
    print(f"Speedup (median): {speedup_median:.2f}x")
    print(f"Speedup (mean):   {speedup_mean:.2f}x")


if __name__ == "__main__":
    main()
