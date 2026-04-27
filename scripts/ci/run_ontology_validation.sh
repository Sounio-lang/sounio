#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MODE="rebuilt"
FILTER="ontology"
WRAPPER_PATH="${SOUNIO_VALIDATION_WRAPPER_PATH:-/tmp/sounio-ontology-validation-souc}"
VERBOSE="--verbose"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/ci/run_ontology_validation.sh [--mode default|rebuilt|diff] [--wrapper <path>] [--quiet|--verbose] [filter]

Examples:
  bash scripts/ci/run_ontology_validation.sh
  bash scripts/ci/run_ontology_validation.sh --mode rebuilt ontology_roles_basic
  bash scripts/ci/run_ontology_validation.sh --mode diff ontology
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --wrapper)
            WRAPPER_PATH="$2"
            shift 2
            ;;
        --quiet)
            VERBOSE=""
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -*)
            echo "error: unsupported argument: $1" >&2
            exit 2
            ;;
        *)
            FILTER="$1"
            shift
            ;;
    esac
done

if [[ "$MODE" != "default" && "$MODE" != "rebuilt" && "$MODE" != "diff" ]]; then
    echo "error: --mode must be one of: default, rebuilt, diff" >&2
    exit 2
fi

HARNESS="$ROOT_DIR/scripts/run_sio_test_suite.sh"
BUILD_WRAPPER="$ROOT_DIR/scripts/ci/build_ontology_validation_souc.sh"
REBUILT_GATE_INACTIVE_RC=97

run_mode() {
    local mode="$1"
    local output_file="$2"
    local results_dir="${3:-}"
    local label=""
    local souc_bin=""

    if [[ -n "$results_dir" ]]; then
        mkdir -p "$results_dir"
        export SOUNIO_TEST_RESULTS_DIR="$results_dir"
    else
        unset SOUNIO_TEST_RESULTS_DIR 2>/dev/null || true
    fi

    if [[ -n "$output_file" ]]; then
        set +e
        if [[ "$mode" == "rebuilt" ]]; then
            (
                set +e
                bash "$BUILD_WRAPPER" "$WRAPPER_PATH"
                build_rc=$?
                set -e
                if [[ $build_rc -ne 0 ]]; then
                    echo "error: rebuilt ontology gate inactive: wrapper build failed"
                    exit $REBUILT_GATE_INACTIVE_RC
                fi
                if [[ ! -x "$WRAPPER_PATH" ]]; then
                    echo "error: rebuilt ontology gate inactive: wrapper missing or not executable: $WRAPPER_PATH"
                    exit $REBUILT_GATE_INACTIVE_RC
                fi
                echo "Using rebuilt wrapper: $WRAPPER_PATH"
                SOUNIO_TEST_SOUC_BIN="$WRAPPER_PATH" \
                SOUNIO_TEST_MODE_LABEL="rebuilt-current-source-wrapper" \
                bash "$HARNESS" "$FILTER" $VERBOSE
            ) >"$output_file" 2>&1
            local rc=$?
        else
            SOUNIO_TEST_MODE_LABEL="default-native" \
            bash "$HARNESS" "$FILTER" $VERBOSE >"$output_file" 2>&1
            local rc=$?
        fi
        set -e
        return "$rc"
    fi

    if [[ "$mode" == "rebuilt" ]]; then
        if ! bash "$BUILD_WRAPPER" "$WRAPPER_PATH"; then
            echo "error: rebuilt ontology gate inactive: wrapper build failed" >&2
            return "$REBUILT_GATE_INACTIVE_RC"
        fi
        if [[ ! -x "$WRAPPER_PATH" ]]; then
            echo "error: rebuilt ontology gate inactive: wrapper missing or not executable: $WRAPPER_PATH" >&2
            return "$REBUILT_GATE_INACTIVE_RC"
        fi
        echo "Using rebuilt wrapper: $WRAPPER_PATH"
        SOUNIO_TEST_SOUC_BIN="$WRAPPER_PATH" \
        SOUNIO_TEST_MODE_LABEL="rebuilt-current-source-wrapper" \
        bash "$HARNESS" "$FILTER" $VERBOSE
        return 0
    fi

    SOUNIO_TEST_MODE_LABEL="default-native" \
    bash "$HARNESS" "$FILTER" $VERBOSE
}

parse_results() {
    local file="$1"
    local prefix="$2"
    while IFS= read -r line; do
        case "$line" in
            "  PASS  "*)
                local name="${line#  PASS  }"
                eval "${prefix}_status[\"\$name\"]='pass'"
                ;;
            "  FAIL  "*)
                local name="${line#  FAIL  }"
                name="${name%% (*}"
                eval "${prefix}_status[\"\$name\"]='fail'"
                ;;
            "  SKIP  "*)
                local name="${line#  SKIP  }"
                eval "${prefix}_status[\"\$name\"]='skip'"
                ;;
        esac
    done <"$file"
}

is_compile_fail_fixture() {
    local file_name="$1"
    local file_path
    file_path="$(find "$ROOT_DIR/tests" -type f -name "$file_name" | head -n 1 || true)"
    if [[ -z "$file_path" ]]; then
        return 1
    fi
    grep -q "//@ compile-fail" "$file_path"
}

classify_diff() {
    local legacy="$1"
    local kernel="$2"
    local file_name="$3"

    if [[ "$legacy" == "pass" && "$kernel" == "pass" ]]; then
        echo "legacy pass / kernel pass"
        return
    fi
    if [[ "$legacy" == "fail" && "$kernel" == "pass" ]]; then
        if is_compile_fail_fixture "$file_name"; then
            echo "legacy true-fail / kernel false-pass"
        else
            echo "legacy fail / kernel pass"
        fi
        return
    fi
    if [[ "$legacy" == "pass" && "$kernel" == "fail" ]]; then
        if is_compile_fail_fixture "$file_name"; then
            echo "legacy false-pass / kernel true-fail"
        else
            echo "legacy pass / kernel fail"
        fi
        return
    fi
    if [[ "$legacy" == "fail" && "$kernel" == "fail" ]]; then
        if is_compile_fail_fixture "$file_name"; then
            echo "legacy true-fail / kernel true-fail"
        else
            echo "legacy fail / kernel fail"
        fi
        return
    fi
    echo "legacy=$legacy / kernel=$kernel"
}

emit_structured_summary() {
    local default_results_dir="$1"
    local rebuilt_results_dir="$2"

    if ! command -v jq >/dev/null 2>&1; then
        echo "=== Agent Consensus Summary ==="
        echo "note: jq not available, install jq for structured output"
        return
    fi

    local unanimous_pass=0 unanimous_fail=0 disagreement=0 fallback_override=0 other=0

    if [[ -d "$rebuilt_results_dir" ]]; then
        for f in "$rebuilt_results_dir"/*.json; do
            [[ -f "$f" ]] || continue
            local status resolution name
            status=$(jq -r '.status // empty' "$f" 2>/dev/null)
            resolution=$(jq -r '.agent_witness.resolution // empty' "$f" 2>/dev/null)
            # Infer resolution when agent_witness is absent (common for run/compile commands)
            if [[ -z "$resolution" ]]; then
                name=$(jq -r '.name // empty' "$f" 2>/dev/null)
                local legacy_status="${default_status[$name]:-missing}"
                local kernel_status="${rebuilt_status[$name]:-missing}"
                if [[ "$legacy_status" == "pass" && "$kernel_status" == "pass" ]]; then
                    resolution="unanimous"
                    status="pass"
                elif [[ "$legacy_status" == "fail" && "$kernel_status" == "fail" ]]; then
                    resolution="unanimous"
                    status="fail"
                elif [[ "$legacy_status" != "missing" && "$kernel_status" != "missing" ]]; then
                    resolution="disagreement"
                fi
            fi
            case "$resolution" in
                unanimous)
                    if [[ "$status" == "pass" ]]; then unanimous_pass=$((unanimous_pass + 1)); else unanimous_fail=$((unanimous_fail + 1)); fi
                    ;;
                disagreement) disagreement=$((disagreement + 1)) ;;
                fallback_override) fallback_override=$((fallback_override + 1)) ;;
                *) other=$((other + 1)) ;;
            esac
        done
    fi

    local summary_file
    summary_file=$(mktemp /tmp/sounio-ontology-summary-XXXXXX.json)

    {
        echo "{"
        echo "  \"unanimous_pass\": $unanimous_pass,"
        echo "  \"unanimous_fail\": $unanimous_fail,"
        echo "  \"disagreement\": $disagreement,"
        echo "  \"fallback_override\": $fallback_override,"
        echo "  \"other\": $other,"
        echo "  \"tests\": ["

        local first=1
        while IFS= read -r name; do
            legacy="${default_status[$name]:-missing}"
            kernel="${rebuilt_status[$name]:-missing}"
            local classification
            classification=$(classify_diff "$legacy" "$kernel" "$name")
            local witness="null"
            if [[ -d "$rebuilt_results_dir" ]]; then
                local f="$rebuilt_results_dir/$name.json"
                if [[ -f "$f" ]]; then
                    witness=$(jq -c '.agent_witness // null' "$f" 2>/dev/null)
                fi
            fi
            if [[ $first -eq 1 ]]; then
                first=0
            else
                echo ","
            fi
            echo -n "    {\"name\":\"$name\",\"legacy\":\"$legacy\",\"kernel\":\"$kernel\",\"classification\":\"$classification\",\"agent_witness\":$witness}"
        done < <(printf '%s\n' "${!seen_names[@]}" | sort)
        echo ""
        echo "  ]"
        echo "}"
    } > "$summary_file"

    echo ""
    echo "=== Agent Consensus Summary ==="
    cat "$summary_file"
    rm -f "$summary_file"
}

if [[ "$MODE" != "diff" ]]; then
    run_mode "$MODE" ""
    exit 0
fi

DEFAULT_OUT="$(mktemp /tmp/sounio-ontology-default-XXXXXX.log)"
REBUILT_OUT="$(mktemp /tmp/sounio-ontology-rebuilt-XXXXXX.log)"
DEFAULT_RESULTS="$(mktemp -d /tmp/sounio-ontology-default-results-XXXXXX)"
REBUILT_RESULTS="$(mktemp -d /tmp/sounio-ontology-rebuilt-results-XXXXXX)"
trap 'rm -rf "$DEFAULT_OUT" "$REBUILT_OUT" "$DEFAULT_RESULTS" "$REBUILT_RESULTS"' EXIT

DEFAULT_RC=0
run_mode "default" "$DEFAULT_OUT" "$DEFAULT_RESULTS" || DEFAULT_RC=$?
REBUILT_RC=0
run_mode "rebuilt" "$REBUILT_OUT" "$REBUILT_RESULTS" || REBUILT_RC=$?

cat "$DEFAULT_OUT"
echo ""
cat "$REBUILT_OUT"
echo ""
echo "=== Ontology Differential Summary ==="
echo "default_exit=$DEFAULT_RC"
echo "rebuilt_exit=$REBUILT_RC"

if [[ $REBUILT_RC -eq $REBUILT_GATE_INACTIVE_RC ]]; then
    echo "diff_invalid=1"
    echo "reason=rebuilt gate inactive"
    exit "$REBUILT_GATE_INACTIVE_RC"
fi

declare -A default_status=()
declare -A rebuilt_status=()

parse_results "$DEFAULT_OUT" "default"
parse_results "$REBUILT_OUT" "rebuilt"

declare -A seen_names=()
for name in "${!default_status[@]}"; do
    seen_names["$name"]=1
done
for name in "${!rebuilt_status[@]}"; do
    seen_names["$name"]=1
done

while IFS= read -r name; do
    legacy="${default_status[$name]:-missing}"
    kernel="${rebuilt_status[$name]:-missing}"
    echo "  $name :: $(classify_diff "$legacy" "$kernel" "$name")"
done < <(printf '%s\n' "${!seen_names[@]}" | sort)

emit_structured_summary "$DEFAULT_RESULTS" "$REBUILT_RESULTS"
