#!/usr/bin/env bash
# Run every thesis test-case config and lay the output out exactly as the
# processing/*.py comparison scripts expect it (see RESTRUCTURE_PLAN.md
# Step 4.5 and the header comments of each processing/*.py script for the
# folder-structure documentation this mirrors).
#
# Usage:
#   scripts/run_thesis_configs.sh                # sequential
#   scripts/run_thesis_configs.sh -j 4            # 4 in parallel
#   scripts/run_thesis_configs.sh -j 4 5p3 5p6    # only cases matching these tags
#
# Output goes to results/<Section>/<case-subfolder>/, one directory per run
# (never shared - see the note in src/swme/cli.py's _output_prefix about why that matters). A log per run is kept alongside the
# CSVs; a run summary prints at the end and a non-zero exit status is
# returned if anything failed.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"
JOBS=1
FILTERS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -j|--jobs) JOBS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,17p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) FILTERS+=("$1"); shift ;;
    esac
done

# config_name:output_subdir pairs, mirroring each processing script's
# documented ROOT_DIR/<subfolder> layout exactly:
#   ersoy_alpha_comparison.py / plotter.py [comparison]  -> results/Ersoy/ErsoyData{0,1,2}
#   (5.2 has no dedicated comparison script; single-run  -> results/5p2_Horton_At_Rest
#    output for the general-purpose plotter.py)
#   non_wrapping_pulse_model_comparison.py,
#   plot_non_wrapping_zoom_profiles.py                   -> results/Non_Wrapping_Pulse/Non_Wrapping_Pulse_N{0,1,2}
#   smooth_pulse_model_comparison_cases.py                -> results/Smooth_Pulse/Smooth_Pulse_N{0,1,2}_{Mild,Aggressive}
#   inflow_outflow_comparison.py                          -> results/Smooth_Pulse_Inflow_Outflow/Smooth_Pulse_Inflow_Outflow_N{0,1,2}
#   dry_wet_ablation_comparison.py,
#   zoomed_dry_wet_comparison.py                          -> results/Dry_Wet_Test/{Dry,Wet}_N{1,2}
CASES=(
    "thesis_5p1_mixing_aR0:Ersoy/ErsoyData0"
    "thesis_5p1_mixing_aR1:Ersoy/ErsoyData1"
    "thesis_5p1_mixing_aR2:Ersoy/ErsoyData2"

    "thesis_5p2_horton_at_rest:5p2_Horton_At_Rest"

    "thesis_5p3_pulse_N0:Non_Wrapping_Pulse/Non_Wrapping_Pulse_N0"
    "thesis_5p3_pulse_N1:Non_Wrapping_Pulse/Non_Wrapping_Pulse_N1"
    "thesis_5p3_pulse_N2:Non_Wrapping_Pulse/Non_Wrapping_Pulse_N2"

    "thesis_5p4_horton_N0:Smooth_Pulse/Smooth_Pulse_N0_Mild"
    "thesis_5p4_horton_N1:Smooth_Pulse/Smooth_Pulse_N1_Mild"
    "thesis_5p4_horton_N2:Smooth_Pulse/Smooth_Pulse_N2_Mild"
    "thesis_5p4_horton_aggressive_N0:Smooth_Pulse/Smooth_Pulse_N0_Aggressive"
    "thesis_5p4_horton_aggressive_N1:Smooth_Pulse/Smooth_Pulse_N1_Aggressive"
    "thesis_5p4_horton_aggressive_N2:Smooth_Pulse/Smooth_Pulse_N2_Aggressive"

    "thesis_5p5_horton_N0:Smooth_Pulse_Inflow_Outflow/Smooth_Pulse_Inflow_Outflow_N0"
    "thesis_5p5_horton_N1:Smooth_Pulse_Inflow_Outflow/Smooth_Pulse_Inflow_Outflow_N1"
    "thesis_5p5_horton_N2:Smooth_Pulse_Inflow_Outflow/Smooth_Pulse_Inflow_Outflow_N2"

    "thesis_5p6_source_free_N1:Dry_Wet_Test/Dry_N1"
    "thesis_5p6_source_free_N2:Dry_Wet_Test/Dry_N2"
    "thesis_5p6_source_active_N1:Dry_Wet_Test/Wet_N1"
    "thesis_5p6_source_active_N2:Dry_Wet_Test/Wet_N2"
)

run_one() {
    local pair="$1"
    local name="${pair%%:*}"
    local subdir="${pair#*:}"
    local out="${RESULTS_DIR}/${subdir}"
    mkdir -p "${out}"
    local log="${out}/run.log"

    local start end elapsed
    start=$(date +%s)
    if MPLBACKEND=Agg uv --directory "${REPO_ROOT}" run moment-sw \
            --config "${name}" --output-dir "${out}" > "${log}" 2>&1
    then
        end=$(date +%s); elapsed=$((end - start))
        echo "OK    ${name}  ->  results/${subdir}  (${elapsed}s)"
    else
        end=$(date +%s); elapsed=$((end - start))
        echo "FAIL  ${name}  ->  results/${subdir}  (${elapsed}s, see ${log})"
        # Record the failure to a per-case status file rather than relying on
        # this function's return code, which xargs -P discards - and rather
        # than parsing stdout, which parallel workers can interleave.
        echo "${name}" >> "${STATUS_DIR}/failed"
    fi
}
export -f run_one
export REPO_ROOT RESULTS_DIR STATUS_DIR

# Apply optional name filters (substring match against the config name).
selected=()
for pair in "${CASES[@]}"; do
    name="${pair%%:*}"
    if [[ ${#FILTERS[@]} -eq 0 ]]; then
        selected+=("${pair}")
    else
        for f in "${FILTERS[@]}"; do
            if [[ "${name}" == *"${f}"* ]]; then
                selected+=("${pair}")
                break
            fi
        done
    fi
done

if [[ ${#selected[@]} -eq 0 ]]; then
    echo "No configs matched filters: ${FILTERS[*]}" >&2
    exit 1
fi

echo "Running ${#selected[@]} thesis config(s) into ${RESULTS_DIR}/ (jobs=${JOBS})"
echo

STATUS_DIR="$(mktemp -d)"
export STATUS_DIR
trap 'rm -rf "${STATUS_DIR}"' EXIT

if [[ "${JOBS}" -le 1 ]]; then
    for pair in "${selected[@]}"; do
        run_one "${pair}"
    done
else
    printf '%s\n' "${selected[@]}" | xargs -P "${JOBS}" -I{} bash -c 'run_one "$@"' _ {}
fi

echo
n_total=${#selected[@]}
n_failed=0
[[ -f "${STATUS_DIR}/failed" ]] && n_failed=$(wc -l < "${STATUS_DIR}/failed" | tr -d ' ')
n_ok=$((n_total - n_failed))
echo "Summary: ${n_ok}/${n_total} succeeded."
if [[ "${n_failed}" -gt 0 ]]; then
    echo "Failed:"
    sed 's/^/  /' "${STATUS_DIR}/failed"
    exit 1
fi
