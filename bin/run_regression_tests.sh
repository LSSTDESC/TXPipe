#!/usr/bin/env bash
#
# Run the TXPipe example pipelines locally, the same way CI's "Pipelines"
# and "CMB_Lensing_Pipeline" jobs do (see .github/workflows/ci.yml), so they
# can be used as a manual regression test before pushing.
#
# Usage:
#   bin/run_regression_tests.sh [options] [pipeline ...]
#
# Options:
#   --restart     Pass resume=restart to ceci, forcing every stage to be
#                 re-run from scratch instead of resuming from existing
#                 outputs. This is slow - only use it when you need a full
#                 regression check.
#   -l, --list    List the available pipeline names and exit.
#   -h, --help    Show this help message and exit.
#
# With no pipeline names given, every pipeline is run in turn. Multiple
# pipeline names may be given to run just those.
#
# Examples:
#   bin/run_regression_tests.sh                     # run every pipeline
#   bin/run_regression_tests.sh metadetect          # run just one
#   bin/run_regression_tests.sh --restart metadetect metacal
#
# Requires the example data to already be downloaded and extracted
# (example.tar.gz / example-cmb-lensing.tar.gz - see README.md), and the
# project conda environment to already be activated (conda activate ./conda).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

usage() {
    cat <<'EOF'
Usage: bin/run_regression_tests.sh [options] [pipeline ...]

Run the TXPipe example pipelines locally, the same way CI does, as a manual
regression test. With no pipeline names given, every pipeline is run.

Options:
  --restart     Pass resume=restart to ceci (forces a full re-run; slow)
  -l, --list    List the available pipeline names and exit
  -h, --help    Show this help message and exit

Examples:
  bin/run_regression_tests.sh
  bin/run_regression_tests.sh metadetect
  bin/run_regression_tests.sh --restart metadetect metacal
EOF
}

# Parallel arrays describing each regression-test pipeline. Kept as plain
# indexed arrays (not associative arrays) for bash 3.2 compatibility, since
# that's what ships as /bin/bash on macOS.
PIPELINE_NAMES=()
PIPELINE_YMLS=()
PIPELINE_CHECKS=()
PIPELINE_LOGDIRS=()
PIPELINE_ENVS=()
PIPELINE_PREPS=()
PIPELINE_ARGS=()

add_pipeline() {
    PIPELINE_NAMES+=("$1")
    PIPELINE_YMLS+=("$2")
    PIPELINE_CHECKS+=("$3")
    PIPELINE_LOGDIRS+=("$4")
    PIPELINE_ENVS+=("$5")
    PIPELINE_PREPS+=("${6:-}")
    PIPELINE_ARGS+=("${7:-}")
}

# Optional 6th arg: a shell command run before the pipeline (failure = test failure).
# Optional 7th arg: extra arguments appended to the ceci command line.
#            name                     pipeline yml                                    file that should exist if the run succeeded                              log dir                                         extra env var (NAME=VALUE), or ""
add_pipeline metadetect               examples/test/metadetect/pipeline.yml               data/test/metadetect/outputs_metadetect/shear_xi_plus.png                         data/test/metadetect/logs_metadetect                    ""
add_pipeline scalar_metadetect        examples/test/scalar_metadetect/pipeline.yml        data/test/scalar_metadetect/outputs_scalar_metadetect/shear_xi_plus.png                  data/test/scalar_metadetect/logs_scalar_metadetect             "" \
    "python bin/make_scalar_metadetect.py"
add_pipeline metacal                examples/test/metacal/pipeline.yml                  data/test/metacal/outputs_metacal/shear_xi_plus.png                            data/test/metacal/logs_metacal                       ""
add_pipeline redmagic                 examples/test/redmagic/pipeline.yml                 data/test/redmagic/outputs_redmagic/shear_xi_plus.png                           data/test/redmagic/logs_redmagic                      ""
add_pipeline lensfit                  examples/test/lensfit/pipeline.yml                  data/test/lensfit/outputs_lensfit/shear_xi_plus.png                            data/test/lensfit/logs_lensfit                       ""
add_pipeline metadetect_source_only   examples/test/metadetect_source_only/pipeline.yml   data/test/metadetect_source_only/outputs_metadetect_source_only/shear_xi_plus.png             data/test/metadetect_source_only/logs_metadetect_source_only        "TX_DASK_DEBUG=1"
add_pipeline mock_shear               examples/test/mock_shear/pipeline.yml               data/test/mock_shear/outputs_mock_shear/binned_shear_catalog.hdf5                 data/test/mock_shear/logs_mock_shear                    ""
add_pipeline cmb                      examples/test/cmb/quaia.yml                         data/test/cmb/outputs_quaia/twopoint_data_fourier_cmb_cross_density_plot.png  data/test/cmb/logs_quaia                             ""
add_pipeline metadetect_different_lengths examples/test/metadetect/pipeline.yml            data/test/metadetect/outputs_metadetect/shear_xi_plus.png                         data/test/metadetect/logs_metadetect                    "" \
    "rm -f data/test/metadetect/outputs_metadetect/shear_xi_plus.png && python bin/make_metadetect_file_different_lengths.py" \
    "resume=restart inputs.shear_catalog=data/example/inputs/metadetect_shear_catalog_different_lengths.hdf5"

list_pipelines() {
    echo "Available pipelines:"
    local i
    for i in "${!PIPELINE_NAMES[@]}"; do
        printf '  %-22s %s\n' "${PIPELINE_NAMES[$i]}" "${PIPELINE_YMLS[$i]}"
    done
}

RESTART=0
REQUESTED=()

while [ $# -gt 0 ]; do
    case "$1" in
        --restart)
            RESTART=1
            shift
            ;;
        -l|--list)
            list_pipelines
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            REQUESTED+=("$1")
            shift
            ;;
    esac
done

# Validate requested pipeline names up front, so a typo fails fast instead
# of after other pipelines have already run.
if [ "${#REQUESTED[@]}" -gt 0 ]; then
    for name in "${REQUESTED[@]}"; do
        found=0
        for known in "${PIPELINE_NAMES[@]}"; do
            if [ "$name" = "$known" ]; then
                found=1
                break
            fi
        done
        if [ "$found" -eq 0 ]; then
            echo "Unknown pipeline: $name" >&2
            echo >&2
            list_pipelines >&2
            exit 2
        fi
    done
fi

# Make sure ceci is actually available.
if ! command -v ceci >/dev/null 2>&1; then
    echo "error: 'ceci' was not found on PATH." >&2
    echo "Activate the project conda environment first: conda activate ./conda" >&2
    exit 1
fi

mkdir -p cache/workspaces

PASSED=()
FAILED=()

for i in "${!PIPELINE_NAMES[@]}"; do
    name="${PIPELINE_NAMES[$i]}"

    if [ "${#REQUESTED[@]}" -gt 0 ]; then
        want=0
        for r in "${REQUESTED[@]}"; do
            [ "$r" = "$name" ] && want=1
        done
        [ "$want" -eq 0 ] && continue
    fi

    yml="${PIPELINE_YMLS[$i]}"
    check="${PIPELINE_CHECKS[$i]}"
    log_dir="${PIPELINE_LOGDIRS[$i]}"
    envvar="${PIPELINE_ENVS[$i]}"
    prep="${PIPELINE_PREPS[$i]}"
    extra_args="${PIPELINE_ARGS[$i]}"

    echo ""
    echo "==================================================================="
    echo "Running pipeline: $name  ($yml)"
    echo "==================================================================="

    cmd=(ceci "$yml")
    if [ "$RESTART" -eq 1 ]; then
        cmd+=(resume=restart)
    fi
    if [ -n "$extra_args" ]; then
        for arg in $extra_args; do
            if [ "$arg" = "resume=restart" ] && [ "$RESTART" -eq 1 ]; then
                continue
            fi
            cmd+=("$arg")
        done
    fi

    # When restarting (either via --restart or because the pipeline always
    # restarts), clear out leftovers from any earlier interrupted run.
    # The output directory is taken to be the directory of the check file.
    restarting=0
    for arg in "${cmd[@]}"; do
        [ "$arg" = "resume=restart" ] && restarting=1
    done
    if [ "$restarting" -eq 1 ]; then
        out_dir="$(dirname "$check")"
        echo "+ removing inprogress* from $out_dir and clearing cache/workspaces"
        if [ -d "$out_dir" ]; then
            rm -rf "${out_dir}"/inprogress*
        fi
        rm -rf cache/workspaces/*
    fi

    if [ -n "$envvar" ]; then
        export_name="${envvar%%=*}"
        export_value="${envvar#*=}"
        export "$export_name"="$export_value"
        echo "+ $export_name=$export_value ${cmd[*]}"
    else
        echo "+ ${cmd[*]}"
    fi

    ran_ok=1
    if [ -n "$prep" ]; then
        echo "+ $prep"
        bash -c "$prep" || ran_ok=0
    fi
    if [ "$ran_ok" -eq 1 ]; then
        "${cmd[@]}" || ran_ok=0
    fi

    if [ -n "$envvar" ]; then
        unset "${envvar%%=*}"
    fi

    if [ "$ran_ok" -eq 1 ] && [ -f "$check" ]; then
        echo "PASSED: $name (found $check)"
        PASSED+=("$name")
    else
        echo "FAILED: $name"
        if [ "$ran_ok" -eq 1 ]; then
            echo "  ceci exited 0 but the expected output file is missing: $check"
        fi
        if [ -d "$log_dir" ]; then
            echo "  --- tail of logs in $log_dir ---"
            tail -n 40 "$log_dir"/*.out 2>/dev/null
        else
            echo "  (no log directory found at $log_dir)"
        fi
        FAILED+=("$name")
    fi
done

echo ""
echo "==================================================================="
echo "Summary"
echo "==================================================================="
echo "Passed (${#PASSED[@]}): ${PASSED[*]:-none}"
echo "Failed (${#FAILED[@]}): ${FAILED[*]:-none}"

if [ "${#FAILED[@]}" -gt 0 ]; then
    exit 1
fi
exit 0
