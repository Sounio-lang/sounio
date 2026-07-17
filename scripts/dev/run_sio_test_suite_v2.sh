#!/usr/bin/env bash
# Self-hosted Sounio test suite runner - Version 2 with parallel execution and JUnit output
#
# New features:
#   - Parallel test execution (background jobs with job limit)
#   - JUnit/XML output format for CI integration
#   - New annotations: @known-failure, @skip-if, @requires, @flaky
#
# Annotations:
#   //@ run-pass              — expect exit 0
#   //@ compile-fail          — expect a compiler diagnostic, not a timeout or signal
#   //@ ignore                — skip this test
#   //@ check-only            — compile only, do not execute
#   //@ expect-stdout: X      — stdout must contain X (run-pass only)
#   //@ error-pattern: X      — stderr/stdout must contain X (compile-fail only)
#   //@ known-failure: REASON — documented accepted failure
#   //@ skip-if: CONDITION    — conditional skip (e.g., skip-if: no-gpu)
#   //@ requires: FEATURE     — feature dependency (e.g., requires: gpu)
#   //@ flaky                 — known flaky test
#   //@ timeout: SECONDS      — override default timeout
#
# Usage:
#   bash scripts/run_sio_test_suite_v2.sh [--filter PATTERN] [--verbose] [--format junit] [--jobs N]
#   bash scripts/run_sio_test_suite_v2.sh --filter-prefix PREFIX [--verbose] [--format junit] [--jobs N]
#   bash scripts/run_sio_test_suite_v2.sh --filter-exact BASENAME [--verbose] [--format junit] [--jobs N]
#   bash scripts/run_sio_test_suite_v2.sh [--filter PATTERN] --list-tests
#   bash scripts/run_sio_test_suite_v2.sh --test-list FILE [--verbose] [--format junit] [--jobs N]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../lib/resolve_souc.sh" && -d "$SCRIPT_DIR/../../tests" ]]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$ROOT_DIR"

# Harness-local override. SOUNIO_TEST_SOUC_BIN may point at one of two
# distinct interfaces:
#
#   1. A wrapper-style executable that accepts the harness subcommand
#      interface (check/run/compile/info), e.g. the rebuilt ontology
#      validation wrapper produced by scripts/ci/build_ontology_validation_souc.sh.
#      These are bash scripts (#!/usr/bin/env bash). Use them directly as
#      $SOUC_BIN so the harness calls them as `$SOUC_BIN check file.sio`.
#
#   2. A raw self-hosted ELF binary, e.g. the CI `souc-stage2` artifact,
#      which only accepts the raw `<src> <out>` interface and would treat
#      `check` as a source path. For these, route through the repo wrapper
#      at scripts/ci/souc-native-wrapper.sh, which translates subcommands to
#      the raw interface
#      via SOUNIO_SOUC_BIN.
#
# We discriminate by sniffing the first two bytes for a shebang.
if [[ -n "${SOUNIO_TEST_SOUC_BIN:-}" ]]; then
    if [[ -r "$SOUNIO_TEST_SOUC_BIN" ]] \
       && [[ "$(head -c 2 "$SOUNIO_TEST_SOUC_BIN" 2>/dev/null)" == "#!" ]]; then
        export SOUNIO_SOUC_BIN="$SOUNIO_TEST_SOUC_BIN"
        SOUC_BIN="$SOUNIO_TEST_SOUC_BIN"
    else
        export SOUNIO_SOUC_BIN="$SOUNIO_TEST_SOUC_BIN"
        SOUC_BIN="$ROOT_DIR/scripts/ci/souc-native-wrapper.sh"
    fi
fi
if [[ -n "${SOUNIO_TEST_NATIVE_BIN:-}" ]]; then
    export SOUC_NATIVE_BIN="$SOUNIO_TEST_NATIVE_BIN"
fi

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

# If SOUC_BIN resolved to a raw ELF (not a shell script), route through the
# subcommand wrapper so the harness can call `check`/`run`/`compile` correctly.
if [[ "$(head -c 2 "$SOUC_BIN" 2>/dev/null)" != "#!" ]]; then
    _NATIVE_WRAPPER="$ROOT_DIR/scripts/ci/souc-native-wrapper.sh"
    if [[ -f "$_NATIVE_WRAPPER" ]]; then
        export SOUNIO_SOUC_BIN="$SOUC_BIN"
        SOUC_BIN="$_NATIVE_WRAPPER"
        chmod +x "$_NATIVE_WRAPPER"
    fi
    unset _NATIVE_WRAPPER
fi

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# Parse arguments
FILTER=""
FILTER_MODE="contains"
VERBOSE=""
FORMAT="text"
JOBS="${SOUNIO_TEST_JOBS:-$(nproc 2>/dev/null || echo 4)}"
LIST_TESTS=0
TEST_LIST_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --filter)
            FILTER="$2"
            FILTER_MODE="contains"
            shift 2
            ;;
        --filter-prefix)
            FILTER="$2"
            FILTER_MODE="prefix"
            shift 2
            ;;
        --filter-exact)
            FILTER="$2"
            FILTER_MODE="exact"
            shift 2
            ;;
        --verbose)
            VERBOSE="1"
            shift
            ;;
        --format)
            FORMAT="$2"
            shift 2
            ;;
        --jobs)
            JOBS="$2"
            shift 2
            ;;
        --list-tests)
            LIST_TESTS=1
            shift
            ;;
        --test-list)
            TEST_LIST_FILE="$2"
            shift 2
            ;;
        *)
            FILTER="$1"
            FILTER_MODE="contains"
            shift
            ;;
    esac
done

# Test counters
PASS=0
FAIL=0
SKIP=0
KNOWN_FAILURE=0
FLAKY=0
ERRORS=""

# Repo-level blocker manifest. This lets CI stay strict about new failures while
# keeping old, audited hardening backlog items visible as xfails instead of noise.
KNOWN_FAILURES_FILE="${SOUNIO_TEST_KNOWN_FAILURES_FILE:-}"
declare -A KNOWN_FAILURE_MAP=()
if [[ -z "$KNOWN_FAILURES_FILE" && -z "$FILTER" && "$FORMAT" == "junit" ]]; then
    KNOWN_FAILURES_FILE="$ROOT_DIR/tests/known_failures/hardened_diagnostics_full_suite.txt"
fi
if [[ -n "$KNOWN_FAILURES_FILE" && -f "$KNOWN_FAILURES_FILE" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue
        KNOWN_FAILURE_MAP["$line"]=1
    done < "$KNOWN_FAILURES_FILE"
fi

# JUnit XML output file
JUNIT_FILE="${SOUNIO_TEST_JUNIT_FILE:-$ROOT_DIR/test-results.xml}"

# Temporary directory for parallel execution results
TMPDIR="${TMPDIR:-/tmp}"
TEST_TMP=$(mktemp -d "$TMPDIR/sounio-test-XXXXXX")
if [[ -z "${SOUNIO_TEST_PRESERVE_JSON:-}" ]]; then
    trap "rm -rf $TEST_TMP" EXIT
fi
if [[ -n "${SOUNIO_TEST_RESULTS_DIR:-}" ]]; then
    mkdir -p "$SOUNIO_TEST_RESULTS_DIR"
fi

test_matches_filter() {
    local basename="$1"
    case "$FILTER_MODE" in
        contains)
            [[ -z "$FILTER" || "$basename" == *"$FILTER"* ]]
            ;;
        prefix)
            [[ -z "$FILTER" || "$basename" == "$FILTER"* ]]
            ;;
        exact)
            [[ -z "$FILTER" || "$basename" == "$FILTER" ]]
            ;;
        *)
            echo "error: unknown filter mode: $FILTER_MODE" >&2
            return 2
            ;;
    esac
}

# Function to run a single test
run_test() {
    local file="$1"
    local idx="$2"
    local output_file="$TEST_TMP/result_$idx.json"
    local basename
    basename="$(basename "$file")"
    local rel_file="${file#$ROOT_DIR/}"
    
    local is_run_pass=false
    local is_compile_fail=false
    local is_ignored=false
    local is_check_only=false
    local is_known_failure=false
    local is_flaky=false
    local timeout_val=30
    local skip_if=""
    local requires=""
    local known_reason=""
    
    # Parse annotations
    while IFS= read -r line; do
        if [[ ! "$line" =~ ^[[:space:]]*//@\  && ! "$line" =~ ^[[:space:]]*//\  && ! "$line" =~ ^[[:space:]]*$ ]]; then
            break
        fi
        case "$line" in
            *"//@ run-pass"*) is_run_pass=true ;;
            *"//@ compile-fail"*) is_compile_fail=true ;;
            *"//@ ignore"*) is_ignored=true ;;
            *"//@ check-only"*) is_check_only=true ;;
            *"//@ known-failure"*) 
                is_known_failure=true
                if [[ "$line" =~ known-failure:[[:space:]]*(.+) ]]; then
                    known_reason="${BASH_REMATCH[1]}"
                fi
                ;;
            *"//@ flaky"*) is_flaky=true ;;
            *"//@ timeout"*) 
                if [[ "$line" =~ ([0-9]+) ]]; then
                    timeout_val="${BASH_REMATCH[1]}"
                fi
                ;;
            *"//@ skip-if"*)
                if [[ "$line" =~ skip-if:[[:space:]]*(.+) ]]; then
                    skip_if="${BASH_REMATCH[1]}"
                fi
                ;;
            *"//@ requires"*)
                if [[ "$line" =~ requires:[[:space:]]*(.+) ]]; then
                    requires="${BASH_REMATCH[1]}"
                fi
                ;;
        esac
    done < "$file"

    if [[ -n "${KNOWN_FAILURE_MAP[$rel_file]:-}" ]]; then
        is_known_failure=true
        known_reason="${known_reason:-hardened diagnostics blocker manifest}"
    fi
    
    # Check filter
    if ! test_matches_filter "$basename"; then
        return
    fi
    
    # Check ignored
    if $is_ignored; then
        echo "{\"status\":\"skip\",\"reason\":\"ignored\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"
        return
    fi
    
    # Check skip-if
    if [[ -n "$skip_if" ]]; then
        case "$skip_if" in
            no-gpu) [[ -z "${SOUNIO_GPU_AVAILABLE:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"skip-if:no-gpu\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
            no-llvm) [[ -z "${SOUNIO_LLVM_AVAILABLE:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"skip-if:no-llvm\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
            ci-only) [[ -n "${CI:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"skip-if:ci-only\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
        esac
    fi
    
    # Check requires
    if [[ -n "$requires" ]]; then
        case "$requires" in
            gpu) [[ -z "${SOUNIO_GPU_AVAILABLE:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"requires:gpu\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
            llvm) [[ -z "${SOUNIO_LLVM_AVAILABLE:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"requires:llvm\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
            # `requires: madaros` — feature lives only in the modular Madaros compiler
            # (check.sio), not in the lean_single bootstrap that builds the suite's
            # stage2 binary. Skipped unless SOUNIO_MADAROS_AVAILABLE is set (a future
            # Madaros-based test job sets it). Tracked: Madaros-official migration.
            madaros) [[ -z "${SOUNIO_MADAROS_AVAILABLE:-}" ]] && { echo "{\"status\":\"skip\",\"reason\":\"requires:madaros\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"; return; } ;;
        esac
    fi
    
    # Skip tests with no annotation
    if ! $is_run_pass && ! $is_compile_fail && ! $is_check_only; then
        echo "{\"status\":\"skip\",\"reason\":\"no-annotation\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"
        return
    fi
    
    # Read expected patterns
    local expect_stdout=()
    local error_patterns=()
    while IFS= read -r line; do
        if [[ ! "$line" =~ ^[[:space:]]*//@\  && ! "$line" =~ ^[[:space:]]*//\  && ! "$line" =~ ^[[:space:]]*$ ]]; then
            break
        fi
        if [[ "$line" =~ "//@ expect-stdout:\ "(.*) ]]; then
            expect_stdout+=("${BASH_REMATCH[1]}")
        fi
        if [[ "$line" =~ ^[[:space:]]*//@[[:space:]]*error-pattern:[[:space:]]*(.*)$ ]]; then
            error_patterns+=("${BASH_REMATCH[1]}")
        fi
    done < "$file"
    
    # Execute test
    local output=""
    local exit_code=0
    local test_output=""
    local start_time end_time
    
    start_time=$(date +%s)
    
    if $is_run_pass || $is_check_only; then
        if $is_check_only; then
            output=$(timeout "$timeout_val" "$SOUC_BIN" check "$file" 2>&1) || exit_code=$?
            if [[ $exit_code -eq 124 ]]; then
                test_output="check timed out after ${timeout_val}s"
            elif [[ $exit_code -ne 0 ]]; then
                test_output="check exited $exit_code"
            fi
        else
            output=$(timeout "$timeout_val" "$SOUC_BIN" run "$file" 2>&1) || exit_code=$?
            if [[ $exit_code -eq 124 ]]; then
                test_output="run timed out after ${timeout_val}s"
            elif [[ $exit_code -ne 0 ]]; then
                test_output="run exited $exit_code"
            fi
        fi
        
        # Check expected stdout patterns
        if [[ $exit_code -eq 0 ]]; then
            for pattern in "${expect_stdout[@]}"; do
                if ! echo "$output" | grep -qF "$pattern"; then
                    exit_code=1
                    test_output="missing stdout: $pattern"
                    break
                fi
            done
        fi
        
    elif $is_compile_fail; then
        local tmp_out
        local compile_exit_code=0
        local check_error_patterns=false
        tmp_out="$(mktemp /tmp/sounio-cf-XXXXXX.elf)"
        output=$(timeout "$timeout_val" "$SOUC_BIN" compile "$file" -o "$tmp_out" 2>&1) || compile_exit_code=$?
        rm -f "$tmp_out"

        if [[ $compile_exit_code -eq 124 ]]; then
            test_output="compile timed out after ${timeout_val}s"
            exit_code=1
        # Shells encode signal termination as 128 + signal. A crash is never
        # a valid compile-fail rejection, even when its output matches.
        elif [[ $compile_exit_code -ge 128 && $compile_exit_code -le 192 ]]; then
            local signal_num=$((compile_exit_code - 128))
            test_output="compile terminated by signal ${signal_num} (exit ${compile_exit_code})"
            exit_code=1
        elif [[ $compile_exit_code -ge 125 && $compile_exit_code -le 127 ]]; then
            test_output="compile harness exited ${compile_exit_code}"
            exit_code=1
        elif [[ $compile_exit_code -eq 0 ]] && echo "$output" | grep -qF "typecheck: failed"; then
            check_error_patterns=true
            exit_code=0
        elif [[ $compile_exit_code -eq 0 ]]; then
            test_output="expected compile failure but passed"
            exit_code=1
        else
            check_error_patterns=true
            exit_code=0
        fi

        if $check_error_patterns; then
            for pattern in "${error_patterns[@]}"; do
                if ! echo "$output" | grep -qiF "$pattern"; then
                    exit_code=1
                    test_output="missing error: $pattern"
                    break
                fi
            done
        fi
    fi
    
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Determine final status
    local status=""
    local category=""
    
    if [[ $exit_code -eq 0 ]]; then
        if $is_known_failure; then
            status="xpas"
            category="known-failure"
        elif $is_flaky; then
            status="pass"
            category="flaky"
        else
            status="pass"
            category="pass"
        fi
    else
        if $is_known_failure; then
            status="xfail"
            category="known-failure"
        elif $is_flaky; then
            status="fail"
            category="flaky"
        else
            status="fail"
            category="fail"
        fi
    fi
    
    # Parse agent witness from raw output (if present)
    local agent_witness=""
    agent_witness=$(echo "$output" | sed -n 's/^agent_witness=//p' | head -n 1)
    
    # Escape output for JSON
    local escaped_output
    escaped_output=$(echo "$test_output" | sed 's/"/\\"/g' | tr '\n' ' ' | sed 's/  */ /g' | head -c 200)
    
    local json="{\"status\":\"$status\",\"category\":\"$category\",\"name\":\"$basename\",\"time\":$duration,\"output\":\"$escaped_output\",\"idx\":$idx"
    if [[ -n "$agent_witness" ]]; then
        json="$json,\"agent_witness\":$agent_witness"
    fi
    json="$json}"
    echo "$json" > "$output_file"
    if [[ -n "${SOUNIO_TEST_RESULTS_DIR:-}" ]]; then
        cp "$output_file" "$SOUNIO_TEST_RESULTS_DIR/"
    fi
}

# Collect all test files
TEST_FILES=()
TEST_LIST_HEADER_MODE=""
TEST_LIST_HEADER_FILTER=""
if [[ -n "$TEST_LIST_FILE" ]]; then
    if [[ ! -f "$TEST_LIST_FILE" ]]; then
        echo "error: --test-list file not found: $TEST_LIST_FILE" >&2
        exit 2
    fi
    while IFS= read -r f; do
        if [[ "$f" == "# sounio-test-list"$'\t'* ]]; then
            IFS=$'\t' read -r _ TEST_LIST_HEADER_MODE TEST_LIST_HEADER_FILTER <<< "$f"
            continue
        fi
        f="${f%%#*}"
        f="${f#"${f%%[![:space:]]*}"}"
        f="${f%"${f##*[![:space:]]}"}"
        [[ -z "$f" ]] && continue
        if [[ "$f" != /* ]]; then
            f="$ROOT_DIR/$f"
        fi
        if [[ ! -f "$f" ]]; then
            echo "error: --test-list entry not found: $f" >&2
            exit 2
        fi
        TEST_FILES+=("$f")
    done <"$TEST_LIST_FILE"
else
    for f in "$ROOT_DIR"/tests/run-pass/*.sio; do
        [[ -f "$f" ]] && TEST_FILES+=("$f")
    done
    for f in "$ROOT_DIR"/tests/compile-fail/*.sio; do
        [[ -f "$f" ]] && TEST_FILES+=("$f")
    done
    for f in "$ROOT_DIR"/tests/ui/type/*.sio "$ROOT_DIR"/tests/ui/effect/*.sio "$ROOT_DIR"/tests/ui/ownership/*.sio "$ROOT_DIR"/tests/ui/resolve/*.sio "$ROOT_DIR"/tests/ui/pattern/*.sio; do
        [[ -f "$f" ]] && TEST_FILES+=("$f")
    done
    for f in "$ROOT_DIR"/tests/stdlib/*/test_*.sio; do
        [[ -f "$f" ]] && TEST_FILES+=("$f")
    done
    for f in "$ROOT_DIR"/tests/gpu/*.sio; do
        [[ -f "$f" ]] && TEST_FILES+=("$f")
    done
fi

if [[ -n "$TEST_LIST_HEADER_MODE" ]]; then
    if [[ "$TEST_LIST_HEADER_MODE" != "$FILTER_MODE" || "$TEST_LIST_HEADER_FILTER" != "$FILTER" ]]; then
        echo "error: --test-list filter header does not match active filter" >&2
        echo "  header: mode=$TEST_LIST_HEADER_MODE filter=$TEST_LIST_HEADER_FILTER" >&2
        echo "  active: mode=$FILTER_MODE filter=$FILTER" >&2
        exit 2
    fi
fi

if [[ -n "$TEST_LIST_FILE" && -n "$FILTER" ]]; then
    for f in "${TEST_FILES[@]}"; do
        basename="$(basename "$f")"
        if ! test_matches_filter "$basename"; then
            echo "error: --test-list entry does not match active filter: ${f#$ROOT_DIR/}" >&2
            exit 2
        fi
    done
fi

if [[ "$LIST_TESTS" == "1" ]]; then
    for f in "${TEST_FILES[@]}"; do
        basename="$(basename "$f")"
        test_matches_filter "$basename" || continue
        printf '%s\n' "${f#$ROOT_DIR/}"
    done
    exit 0
fi

export SOUC_BIN ROOT_DIR FILTER TEST_TMP SOUNIO_STDLIB_PATH CI SOUNIO_GPU_AVAILABLE SOUNIO_LLVM_AVAILABLE

# Header
echo "=== Sounio Test Suite ==="
echo "Using souc: $SOUC_BIN"
echo "Parallel jobs: $JOBS"
if [[ -n "${SOUC_NATIVE_BIN:-}" ]]; then
    echo "Using native backend: $SOUC_NATIVE_BIN"
fi
echo ""

echo "Found ${#TEST_FILES[@]} test files"
echo ""

# Run tests in parallel with job limit
job_count=0
idx=0
for f in "${TEST_FILES[@]}"; do
    ((idx++))
    run_test "$f" "$idx" &
    ((job_count++))
    if [[ $job_count -ge $JOBS ]]; then
        wait -n 2>/dev/null || wait
        ((job_count--))
    fi
done
wait

# Collect results
for f in "$TEST_TMP"/result_*.json; do
    [[ -f "$f" ]] || continue
    result=$(cat "$f")
    status=$(echo "$result" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    category=$(echo "$result" | grep -o '"category":"[^"]*"' | cut -d'"' -f4)
    name=$(echo "$result" | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
    
    case "$status" in
        pass)
            ((PASS++))
            if [[ "$VERBOSE" == "1" ]]; then
                echo "  PASS  $name"
            fi
            ;;
        fail)
            ((FAIL++))
            output=$(echo "$result" | grep -o '"output":"[^"]*"' | cut -d'"' -f4)
            ERRORS="${ERRORS}  FAIL  $name"
            [[ -n "$output" ]] && ERRORS="$ERRORS ($output)"
            ERRORS="$ERRORS
"
            ;;
        xfail)
            ((KNOWN_FAILURE++))
            ;;
        xpas)
            ((PASS++))
            if [[ "$VERBOSE" == "1" ]]; then
                echo "  XPAS  $name (known failure now passes)"
            fi
            ;;
        skip)
            ((SKIP++))
            reason=$(echo "$result" | grep -o '"reason":"[^"]*"' | cut -d'"' -f4)
            if [[ "$VERBOSE" == "1" ]]; then
                echo "  SKIP  $name ($reason)"
            fi
            ;;
    esac
done

# Generate output
echo ""
echo "=== Results ==="
echo "  Pass: $PASS"
echo "  Fail: $FAIL"
[[ $KNOWN_FAILURE -gt 0 ]] && echo "  Known failures: $KNOWN_FAILURE"
[[ $FLAKY -gt 0 ]] && echo "  Flaky: $FLAKY"
echo "  Skip: $SKIP"
echo "  Total: $((PASS + FAIL + SKIP + KNOWN_FAILURE))"

# Generate JUnit XML if requested
if [[ "$FORMAT" == "junit" ]]; then
    cat > "$JUNIT_FILE" << 'XMLEOF'
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
XMLEOF
    
    echo "  <testsuite name=\"sounio-test-suite\" tests=\"$((PASS + FAIL + KNOWN_FAILURE))\" failures=\"$FAIL\" skipped=\"$SKIP\" errors=\"0\">" >> "$JUNIT_FILE"
    
    for f in "$TEST_TMP"/result_*.json; do
        [[ -f "$f" ]] || continue
        result=$(cat "$f")
        name=$(echo "$result" | grep -o '"name":"[^"]*"' | cut -d'"' -f4 | sed 's/\.sio$//')
        [[ -n "$name" ]] || continue
        status=$(echo "$result" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
        time=$(echo "$result" | grep -o '"time":[^,}]*' | cut -d':' -f2)
        output=$(echo "$result" | grep -o '"output":"[^"]*"' | cut -d'"' -f4)
        
        case "$status" in
            pass)
                echo "    <testcase name=\"$name\" time=\"$time\"/>" >> "$JUNIT_FILE"
                ;;
            xpas)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <system-out>Known failure now passes</system-out>" >> "$JUNIT_FILE"
                echo "    </testcase>" >> "$JUNIT_FILE"
                ;;
            fail)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <failure message=\"$output\"/>" >> "$JUNIT_FILE"
                echo "    </testcase>" >> "$JUNIT_FILE"
                ;;
            xfail)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <skipped message=\"Known failure\"/>" >> "$JUNIT_FILE"
                echo "    </testcase>" >> "$JUNIT_FILE"
                ;;
            skip)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <skipped/>" >> "$JUNIT_FILE"
                echo "    </testcase>" >> "$JUNIT_FILE"
                ;;
        esac
    done
    
    echo "  </testsuite>" >> "$JUNIT_FILE"
    echo "</testsuites>" >> "$JUNIT_FILE"
    
    echo ""
    echo "JUnit XML written to: $JUNIT_FILE"
fi

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "=== Failures ==="
    echo -n "$ERRORS"
    exit 1
fi

echo ""
echo "All tests passed!"
