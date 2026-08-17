#!/usr/bin/env bash
# Run the smoke-test configs in parallel, one core each.
#
# These two cases between them exercise most of the solver: recharge with
# Horton infiltration, the HSWME closure, well-balanced topography, wet-dry
# handling, and the hyperbolicity diagnostic. Running them concurrently is
# worth it because they are independent and each pins one core.
#
# Usage:
#   scripts/run_smoke_tests.sh              # both, in parallel
#   scripts/run_smoke_tests.sh --report     # also write report.pdf per run
#   scripts/run_smoke_tests.sh smoke_test_2 # just the ones matching a name
#
# Output goes to results/<config-name>/, with the console output of each run
# kept alongside its CSVs as run.log. Exit status is non-zero if any run
# failed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"

CONFIGS=(
    smoke_test_1
    smoke_test_2
)

REPORT=""
FILTERS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--report) REPORT="--report"; shift ;;
        -h|--help)
            sed -n '2,17p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) FILTERS+=("$1"); shift ;;
    esac
done

# Optional substring filter on the config name.
selected=()
for name in "${CONFIGS[@]}"; do
    if [[ ${#FILTERS[@]} -eq 0 ]]; then
        selected+=("${name}")
    else
        for f in "${FILTERS[@]}"; do
            if [[ "${name}" == *"${f}"* ]]; then
                selected+=("${name}")
                break
            fi
        done
    fi
done

if [[ ${#selected[@]} -eq 0 ]]; then
    echo "No configs matched: ${FILTERS[*]}" >&2
    echo "Available: ${CONFIGS[*]}" >&2
    exit 1
fi

TIMING_DIR="$(mktemp -d)"
trap 'rm -rf "${TIMING_DIR}"' EXIT

run_one() {
    local name="$1"
    local out="${RESULTS_DIR}/${name}"
    mkdir -p "${out}"

    local start end
    start=$(date +%s)
    uv --directory "${REPO_ROOT}" run moment-sw \
        --config "${name}" --output-dir "${out}" ${REPORT} \
        > "${out}/run.log" 2>&1
    local code=$?
    end=$(date +%s)

    # Each job times itself. Measuring in the parent around `wait` would charge
    # a fast job for the time spent waiting on a slower one ahead of it in the
    # list, which is wrong by minutes when the two differ in cost.
    echo "$((end - start))" > "${TIMING_DIR}/${name}"
    return "${code}"
}

echo "Running ${#selected[@]} smoke test(s) in parallel into results/"
echo

declare -a pids=()

for name in "${selected[@]}"; do
    run_one "${name}" &
    pids+=("$!")
    echo "  started  ${name}  (pid $!)"
done

echo
echo "Waiting..."
echo

failed=0
for i in "${!selected[@]}"; do
    name="${selected[$i]}"
    # `wait` on a specific pid yields that job's own exit status, which is what
    # makes this readable at two jobs; a shared pool would need status files.
    if wait "${pids[$i]}"; then
        status="OK  "
    else
        status="FAIL"
        failed=$((failed + 1))
    fi
    elapsed="$(cat "${TIMING_DIR}/${name}" 2>/dev/null || echo '?')"
    printf '%s  %-14s  %4ss  results/%s\n' \
        "${status}" "${name}" "${elapsed}" "${name}"
    if [[ "${status}" == "FAIL" ]]; then
        echo "      last lines of results/${name}/run.log:"
        tail -n 5 "${RESULTS_DIR}/${name}/run.log" | sed 's/^/      /'
    fi
done

echo
if [[ "${failed}" -gt 0 ]]; then
    echo "${failed} of ${#selected[@]} run(s) failed."
    exit 1
fi
echo "All ${#selected[@]} run(s) completed."
