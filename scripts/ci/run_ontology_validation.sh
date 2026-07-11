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
GENERATED_ONTOLOGY_GATE="$ROOT_DIR/scripts/ci/generated_ontology_gate.sh"
REBUILT_GATE_INACTIVE_RC=97
_SOUNIO_ONTOLOGY_TMP_PATHS=()

_sounio_ontology_track_tmp() {
    _SOUNIO_ONTOLOGY_TMP_PATHS+=("$1")
}

_sounio_ontology_cleanup_tmp() {
    local p
    for p in "${_SOUNIO_ONTOLOGY_TMP_PATHS[@]}"; do
        [[ -n "$p" ]] && rm -rf "$p" 2>/dev/null || true
    done
}

trap _sounio_ontology_cleanup_tmp EXIT

ONTOLOGY_COMPILE_GATES=(
    ontology_cache_compile_gate.sh
    ontology_cli_smoke_gate.sh
    ontology_model_compile_gate.sh
    ontology_query_compile_gate.sh
    ontology_reasoner_compile_gate.sh
    ontology_typed_bridge_gate.sh
)

run_ontology_compile_gates() {
    if [[ "${SOUNIO_ONTOLOGY_COMPILE_GATES:-1}" == "0" ]]; then
        echo "[ontology-validation] skipping compile gate bundle (SOUNIO_ONTOLOGY_COMPILE_GATES=0)"
        return 0
    fi

    local seq_probe_dir seq_probe_src seq_probe_out seq_probe_log
    seq_probe_dir="$(mktemp -d /tmp/sounio-ontology-seq-probe-XXXXXX)"
    seq_probe_src="$seq_probe_dir/seq_probe.sio"
    seq_probe_out="$seq_probe_dir/seq_probe"
    seq_probe_log="$seq_probe_dir/seq_probe.log"
    cat >"$seq_probe_src" <<'EOF'
fn main() -> i32 with IO, Mut, Panic, Epistemic {
    var s: Seq<i64> = seq_new()
    s.push(1)
    return 0
}
EOF
    source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
    if "$SOUC_BIN" --help 2>/dev/null | grep -q "compile <file.sio>"; then
        seq_probe_cmd=( "$SOUC_BIN" compile "$seq_probe_src" -o "$seq_probe_out" )
    else
        seq_probe_cmd=( "$SOUC_BIN" "$seq_probe_src" "$seq_probe_out" )
    fi
    if ! "${seq_probe_cmd[@]}" >"$seq_probe_log" 2>&1; then
        echo "[ontology-validation] skipping compile gate bundle: selected compiler surface lacks Seq runtime compile support"
        tail -n 8 "$seq_probe_log" || true
        rm -rf "$seq_probe_dir"
        return 0
    fi
    rm -rf "$seq_probe_dir"

    local gate script
    for gate in "${ONTOLOGY_COMPILE_GATES[@]}"; do
        script="$ROOT_DIR/scripts/ci/$gate"
        if [[ ! -f "$script" ]]; then
            echo "error: missing ontology compile gate: $script" >&2
            exit 2
        fi
        echo "[ontology-validation] running $gate"
        bash "$script"
    done
}

prepare_generated_ontology() {
    if [[ "${SOUNIO_ONTOLOGY_PREPARE_GENERATED:-1}" == "0" ]]; then
        return 0
    fi
    if [[ ! -x "$GENERATED_ONTOLOGY_GATE" && ! -f "$GENERATED_ONTOLOGY_GATE" ]]; then
        echo "error: generated ontology gate missing: $GENERATED_ONTOLOGY_GATE" >&2
        exit 2
    fi

    local check_mode="${SOUNIO_ONTOLOGY_CHECK_GENERATED:-auto}"
    if [[ "$check_mode" == "auto" ]]; then
        if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
            check_mode="1"
        else
            check_mode="0"
        fi
    fi

    if [[ "$check_mode" == "1" || "$check_mode" == "true" ]]; then
        bash "$GENERATED_ONTOLOGY_GATE" --check
    else
        bash "$GENERATED_ONTOLOGY_GATE" --refresh-only
    fi
}

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
                local run_rc=0
                set +e
                run_harness_filters "rebuilt-current-source-wrapper" "$WRAPPER_PATH" "$FILTER"
                run_rc=$?
                set -e
                exit "$run_rc"
            ) >"$output_file" 2>&1
            local rc=$?
        else
            local rc=0
            set +e
            run_harness_filters "default-native" "" "$FILTER" >"$output_file" 2>&1
            rc=$?
            set -e
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
        local run_rc=0
        set +e
        run_harness_filters "rebuilt-current-source-wrapper" "$WRAPPER_PATH" "$FILTER"
        run_rc=$?
        set -e
        return "$run_rc"
    fi

    local run_rc=0
    set +e
    run_harness_filters "default-native" "" "$FILTER"
    run_rc=$?
    set -e
    return "$run_rc"
}

emit_filter_specs() {
    local original_filter="$1"
    if [[ "$original_filter" == "ontology" ]]; then
        # The ontology validation gate owns the ontology-kernel fixtures:
        # ontology_* plus the stdlib ontology smoke. Chemistry/PBPK bridge tests
        # that happen to contain "ontology" in their basename stay in the full
        # suite, where the compiler oracle matches their intended surface.
        # Operational namespace invariant: tests/run-pass/ontology_*.sio and
        # tests/compile-fail/ontology_*.sio are reserved for ontology-kernel
        # fixtures. Cross-domain bridge tests must not use that prefix.
        printf '%s\t%s\n' "ontology_" "prefix"
        printf '%s\t%s\n' "test_ontology.sio" "exact"
    else
        printf '%s\t%s\n' "$original_filter" "contains"
    fi
}

build_validated_harness_list() {
    local filter="$1"
    local list_file="$2"
    local validation_mode="$3"
    local filter_mode="$4"
    local raw_list list_rc=0 saw=0 rel
    raw_list="$(mktemp /tmp/sounio-ontology-filter-raw-XXXXXX.list)"
    _sounio_ontology_track_tmp "$raw_list"
    case "$filter_mode" in
        prefix)
            bash "$HARNESS" --filter-prefix "$filter" --list-tests >"$raw_list" || list_rc=$?
            ;;
        exact)
            bash "$HARNESS" --filter-exact "$filter" --list-tests >"$raw_list" || list_rc=$?
            ;;
        contains)
            bash "$HARNESS" "$filter" --list-tests >"$raw_list" || list_rc=$?
            ;;
        *)
            rm -f "$raw_list"
            echo "error: unknown validation filter mode: $filter_mode" >&2
            return 2
            ;;
    esac
    if [[ "$list_rc" -ne 0 ]]; then
        rm -f "$raw_list"
        echo "error: validation filter listing failed: $filter" >&2
        return "$list_rc"
    fi
    while IFS= read -r rel; do
        [[ -n "$rel" ]] || continue
        saw=1
        if [[ "$validation_mode" == "ontology-kernel" ]]; then
            case "$filter" in
                ontology_)
                    if [[ "$rel" != tests/run-pass/ontology_*.sio && "$rel" != tests/compile-fail/ontology_*.sio ]]; then
                        rm -f "$raw_list"
                        echo "error: canonical ontology filter selected non-kernel fixture: $rel" >&2
                        return 2
                    fi
                    ;;
                test_ontology.sio)
                    if [[ "$rel" != "tests/stdlib/ontology/test_ontology.sio" ]]; then
                        rm -f "$raw_list"
                        echo "error: canonical ontology smoke filter selected unexpected fixture: $rel" >&2
                        return 2
                    fi
                    ;;
            esac
        fi
    done <"$raw_list"
    if [[ "$saw" -ne 1 ]]; then
        rm -f "$raw_list"
        echo "error: validation filter matched no harness fixtures: $filter" >&2
        return 2
    fi
    printf '# sounio-test-list\t%s\t%s\n' "$filter_mode" "$filter" >"$list_file"
    cat "$raw_list" >>"$list_file"
    rm -f "$raw_list"
    return 0
}

run_harness_filters() {
    local mode_label="$1"
    local test_souc_bin="$2"
    local original_filter="$3"
    local rc=0
    local filter filter_mode
    local list_dir list_file
    local harness_filter_args=()
    local harness_flags=()
    local validation_mode="passthrough"
    [[ "$original_filter" == "ontology" ]] && validation_mode="ontology-kernel"
    [[ -n "$VERBOSE" ]] && harness_flags+=("$VERBOSE")
    list_dir="$(mktemp -d /tmp/sounio-ontology-filters-XXXXXX)"
    _sounio_ontology_track_tmp "$list_dir"
    while IFS=$'\t' read -r filter filter_mode; do
        [[ -n "$filter" ]] || continue
        case "$filter_mode" in
            prefix)
                harness_filter_args=(--filter-prefix "$filter")
                ;;
            exact)
                harness_filter_args=(--filter-exact "$filter")
                ;;
            contains)
                harness_filter_args=("$filter")
                ;;
            *)
                rm -rf "$list_dir"
                echo "error: unknown harness filter mode: $filter_mode" >&2
                return 2
                ;;
        esac
        list_file="$(mktemp "$list_dir/${filter_mode}.XXXXXX.list")"
        build_validated_harness_list "$filter" "$list_file" "$validation_mode" "$filter_mode" || {
            rc=$?
            rm -rf "$list_dir"
            return "$rc"
        }
        echo ""
        echo "=== Ontology validation filter: $filter ==="
        if [[ -n "$test_souc_bin" ]]; then
            if SOUNIO_TEST_SOUC_BIN="$test_souc_bin" \
                SOUNIO_TEST_MODE_LABEL="$mode_label" \
                bash "$HARNESS" "${harness_filter_args[@]}" --test-list "$list_file" "${harness_flags[@]}"; then
                rc=0
            else
                rc=$?
            fi
        else
            if SOUNIO_TEST_MODE_LABEL="$mode_label" \
                bash "$HARNESS" "${harness_filter_args[@]}" --test-list "$list_file" "${harness_flags[@]}"; then
                rc=0
            else
                rc=$?
            fi
        fi
        if [[ "$rc" -ne 0 ]]; then
            rm -rf "$list_dir"
            return "$rc"
        fi
    done < <(emit_filter_specs "$original_filter")
    rm -rf "$list_dir"
    return "$rc"
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

prepare_generated_ontology
run_ontology_compile_gates

if [[ "$MODE" != "diff" ]]; then
    run_mode "$MODE" ""
    exit 0
fi

DEFAULT_OUT="$(mktemp /tmp/sounio-ontology-default-XXXXXX.log)"
REBUILT_OUT="$(mktemp /tmp/sounio-ontology-rebuilt-XXXXXX.log)"
DEFAULT_RESULTS="$(mktemp -d /tmp/sounio-ontology-default-results-XXXXXX)"
REBUILT_RESULTS="$(mktemp -d /tmp/sounio-ontology-rebuilt-results-XXXXXX)"
trap 'rm -rf "$DEFAULT_OUT" "$REBUILT_OUT" "$DEFAULT_RESULTS" "$REBUILT_RESULTS"; _sounio_ontology_cleanup_tmp' EXIT

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
