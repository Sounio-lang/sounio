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
#   //@ expect-stdout-contains: X — stdout must contain X (run-pass only)
#   //@ error-pattern: X      — stderr/stdout must contain X (compile-fail only)
#   //@ known-failure: REASON — documented accepted failure
#   //@ skip-if: CONDITION    — conditional skip (e.g., skip-if: no-gpu)
#   //@ requires: FEATURE     — feature dependency (e.g., requires: gpu)
#   //@ flaky                 — known flaky test
#   //@ timeout: SECONDS      — override default timeout
#
# Unknown `expect-*` / `expected-*` header keys fail the test. They used to
# be skipped silently, so `expect-stdout-contains` asserted nothing.
#
# Usage:
#   bash scripts/dev/run_sio_test_suite_v2.sh [--filter PATTERN] [--verbose] [--format junit] [--jobs N]
#   bash scripts/dev/run_sio_test_suite_v2.sh --filter-prefix PREFIX [--verbose] [--format junit] [--jobs N]
#   bash scripts/dev/run_sio_test_suite_v2.sh --filter-exact BASENAME [--verbose] [--format junit] [--jobs N]
#   bash scripts/dev/run_sio_test_suite_v2.sh [--filter PATTERN] --list-tests
#   bash scripts/dev/run_sio_test_suite_v2.sh --test-list FILE [--verbose] [--format junit] [--jobs N]

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
XPAS=0
XPAS_LIST=""
FLAKY=0
VACUOUS_KNOWN=0
VACUOUS_STALE=""
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

# Vacuous-expect-stdout/error-pattern baseline. The //@ expect-stdout: / //@
# error-pattern: extraction below used to quote the whole `=~` pattern, which
# makes bash treat it as a literal string instead of a regex, so the capture
# group never captured and every such assertion matched vacuously (an empty
# expected string always matches). Fixing the extraction (see the comment at
# the expect_stdout/error_patterns parse loop) makes every test whose
# annotation was actually wrong fail for real, for the first time -- these are
# PRE-EXISTING wrong annotations, not regressions caused by the fix. Mirrors
# scripts/ci/madaros_corpus_regression_gate.sh: compare the failure LIST
# against a checked-in baseline (tests/vacuous_expect_baseline.txt) and
# tolerate only entries already listed there, so the fix can land without
# turning the required `full-test-suite` CI job red. Unlike
# KNOWN_FAILURES_FILE above, this baseline is always active (not gated to
# --format junit with no filter) so it also covers local/manual runs.
#
# Regenerate: SOUNIO_VACUOUS_BASELINE_REFRESH=1 bash scripts/run_sio_test_suite.sh
VACUOUS_BASELINE_FILE="$ROOT_DIR/tests/vacuous_expect_baseline.txt"
VACUOUS_REFRESH="${SOUNIO_VACUOUS_BASELINE_REFRESH:-0}"
declare -A VACUOUS_BASELINE_MAP=()
if [[ "$VACUOUS_REFRESH" != "1" && -f "$VACUOUS_BASELINE_FILE" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue
        VACUOUS_BASELINE_MAP["$line"]=1
    done < "$VACUOUS_BASELINE_FILE"
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
    local is_typecheck_fail=false
    local is_ignored=false
    local is_check_only=false
    local is_known_failure=false
    local is_flaky=false
    local is_vacuous_baseline=false
    local timeout_val=30
    local skip_if=""
    local requires=""
    local known_reason=""
    local unknown_expect=""
    
    # Parse annotations
    while IFS= read -r line; do
        if [[ ! "$line" =~ ^[[:space:]]*//@\  && ! "$line" =~ ^[[:space:]]*//\  && ! "$line" =~ ^[[:space:]]*$ ]]; then
            break
        fi
        # Fail closed on invented stdout assertions. `expect-stdout-contains`
        # was silently ignored because the harness only extracted
        # `expect-stdout:`; the same hole would swallow `expected-output` or
        # `expect-stdout-has`. Key extraction is identifier-only; the payload
        # is still read by parameter expansion below (the vacuous-regex bug).
        local expect_line="${line%"${line##*[![:space:]]}"}"
        expect_line="${expect_line#"${expect_line%%[![:space:]]*}"}"
        expect_line="${expect_line%$'\r'}"
        if [[ "$expect_line" == "//@ expect"* || "$expect_line" == "//@ expected"* ]]; then
            local expect_key="${expect_line#//@ }"
            expect_key="${expect_key%%:*}"
            expect_key="${expect_key%% *}"
            case "$expect_key" in
                expect-stdout|expect-stdout-contains) ;;
                *)
                    if [[ -z "$unknown_expect" ]]; then
                        unknown_expect="$expect_key"
                    fi
                    ;;
            esac
        fi
        case "$line" in
            *"//@ run-pass"*) is_run_pass=true ;;
            *"//@ compile-fail"*) is_compile_fail=true ;;
            *"//@ typecheck-fail"*) is_typecheck_fail=true ;;
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

    # See the VACUOUS_BASELINE_MAP loading comment above. Membership is
    # recorded regardless of exit code, same as is_known_failure -- whether it
    # counts as a tolerated failure (vxfail) or a "now passes, shrink the
    # baseline" notice (vxpas) is decided once exit_code is known below.
    if [[ "$VACUOUS_REFRESH" != "1" && -n "${VACUOUS_BASELINE_MAP[$rel_file]:-}" ]]; then
        is_vacuous_baseline=true
    fi

    # Check filter
    if ! test_matches_filter "$basename"; then
        return
    fi

    if [[ -n "$unknown_expect" ]]; then
        echo "{\"status\":\"fail\",\"category\":\"fail\",\"name\":\"$basename\",\"relfile\":\"$rel_file\",\"time\":0,\"output\":\"unknown annotation: $unknown_expect (expected: expect-stdout|expect-stdout-contains)\",\"idx\":$idx}" > "$output_file"
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
            # An unrecognized skip-if value must not fall through silently: that
            # would let a typo (e.g. `no-gpu` misspelled) run the test unguarded
            # while the annotation reads as if it were gating something -- the
            # same "guard that asserts nothing" defect this file exists to remove.
            *)
                echo "{\"status\":\"fail\",\"category\":\"fail\",\"name\":\"$basename\",\"output\":\"unknown skip-if: $skip_if (expected: no-gpu|no-llvm|ci-only)\",\"idx\":$idx}" > "$output_file"
                return
                ;;
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
            # An unrecognized requires value must not fall through silently: a typo
            # (e.g. `requires: madros`) would otherwise run the test against
            # whatever engine is present instead of being gated as intended, with
            # the annotation asserting nothing -- indistinguishable from the
            # vacuous-match defect this PR exists to remove.
            *)
                echo "{\"status\":\"fail\",\"category\":\"fail\",\"name\":\"$basename\",\"output\":\"unknown requires: $requires (expected: gpu|llvm|madaros)\",\"idx\":$idx}" > "$output_file"
                return
                ;;
        esac
    fi
    
    # Skip tests with no annotation
    if ! $is_run_pass && ! $is_compile_fail && ! $is_check_only && ! $is_typecheck_fail; then
        echo "{\"status\":\"skip\",\"reason\":\"no-annotation\",\"name\":\"$basename\",\"idx\":$idx}" > "$output_file"
        return
    fi
    
    # Read expected patterns
    local expect_stdout=()
    local expect_stdout_contains=()
    local error_patterns=()
    while IFS= read -r line; do
        if [[ ! "$line" =~ ^[[:space:]]*//@\  && ! "$line" =~ ^[[:space:]]*//\  && ! "$line" =~ ^[[:space:]]*$ ]]; then
            break
        fi
        # Extraction by parameter expansion, not regex: the pattern that
        # follows "//@ expect-stdout: " / "//@ error-pattern: " often contains
        # regex metacharacters (`[`, `(`, ...), and quoting the whole =~
        # pattern (as this used to) makes bash treat it as a literal string
        # instead of a regex, so the capture group never captures and
        # BASH_REMATCH[1] is always empty -- every expect-stdout/error-pattern
        # assertion then matched vacuously. Parameter expansion has no
        # metacharacter class to get this wrong for either annotation.
        if [[ "$line" == "//@ expect-stdout: "* ]]; then
            expect_stdout+=("${line#*//@ expect-stdout: }")
        fi
        if [[ "$line" == "//@ expect-stdout-contains: "* ]]; then
            expect_stdout_contains+=("${line#*//@ expect-stdout-contains: }")
        fi
        if [[ "$line" == "//@ error-pattern: "* ]]; then
            error_patterns+=("${line#*//@ error-pattern: }")
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
        
        # Check expected stdout patterns.
        #
        # PIPEFAIL RULE (2026-08-17): never feed a captured string to a
        # verdict-carrying `grep -q` through `echo "$x" | grep -q ...`.
        # `grep -q` exits on first match and closes the pipe; under
        # `set -o pipefail` an `echo` still flushing a large output then
        # fails the whole pipeline (CI log 2026-08-16 21:40:49: "line 434:
        # echo: write error: Broken pipe", in the same run as a "missing
        # error" flake), and `if ! ...` reads that as the pattern being
        # absent. The here-string form has no writer process to lose writes.
        # Guarded by scripts/ci/sigpipe_hygiene_gate.sh.
        if [[ $exit_code -eq 0 ]]; then
            for pattern in "${expect_stdout[@]}"; do
                if ! grep -qF -- "$pattern" <<<"$output"; then
                    exit_code=1
                    test_output="missing stdout: $pattern"
                    break
                fi
            done
            if [[ $exit_code -eq 0 ]]; then
                for pattern in "${expect_stdout_contains[@]}"; do
                    if ! grep -qF -- "$pattern" <<<"$output"; then
                        exit_code=1
                        test_output="missing stdout contains: $pattern"
                        break
                    fi
                done
            fi
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
        elif [[ $compile_exit_code -eq 0 ]] && grep -qF "typecheck: failed" <<<"$output"; then
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
                if ! grep -qiF -- "$pattern" <<<"$output"; then
                    exit_code=1
                    test_output="missing error: $pattern"
                    break
                fi
            done
        fi
    elif $is_typecheck_fail; then
        # Proof-carrying tests: the illegal inference must be rejected by the
        # TYPE CHECKER (`souc check`), and the reason MUST be pinned via
        # //@ error-pattern. Running `check` (not `compile`) is deliberate:
        # `check` runs the boundary-preserving visibility/type pass, whereas
        # `compile` can "pass" on unrelated backend / missing-main failures
        # without ever exercising the guarantee (see audit 2026-07-24).
        # Parse the pinned pattern(s) locally rather than reusing the shared
        # error_patterns array above: this path runs `check`, not `compile`,
        # a distinct contract (see the audit 2026-07-24 note above) that
        # deserves its own local list rather than sharing state with the
        # compile-fail path.
        local tf_patterns=()
        local pline
        while IFS= read -r pline; do
            case "$pline" in
                *"//@ error-pattern: "*) tf_patterns+=("${pline#*//@ error-pattern: }") ;;
            esac
        done < <(head -n 20 "$file")

        output=$(timeout "$timeout_val" "$SOUC_BIN" check "$file" 2>&1) || exit_code=$?
        if [[ $exit_code -eq 124 ]]; then
            test_output="check timed out after ${timeout_val}s"
        elif [[ $exit_code -eq 0 ]]; then
            test_output="expected typecheck failure but passed"
            exit_code=1
        else
            exit_code=0  # nonzero check exit == the expected rejection
            # A typecheck-fail test MUST pin the reason; no error-pattern is vacuous.
            if [[ ${#tf_patterns[@]} -eq 0 ]]; then
                exit_code=1
                test_output="typecheck-fail requires //@ error-pattern to pin the diagnostic"
            else
                for pattern in "${tf_patterns[@]}"; do
                    if ! grep -qiF -- "$pattern" <<<"$output"; then
                        exit_code=1
                        test_output="missing error: $pattern"
                        break
                    fi
                done
            fi
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
        elif $is_vacuous_baseline; then
            status="vxpas"
            category="vacuous-baseline"
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
        elif $is_vacuous_baseline; then
            status="vxfail"
            category="vacuous-baseline"
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

    local json="{\"status\":\"$status\",\"category\":\"$category\",\"name\":\"$basename\",\"relfile\":\"$rel_file\",\"time\":$duration,\"output\":\"$escaped_output\",\"idx\":$idx"
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

# Every test that passes the filter must produce exactly one result_*.json:
# each path through run_test after the filter check writes one, and nothing
# else in TEST_TMP does. Counting them here lets the collector prove below
# that the summary is a measurement of this run -- a worker killed before its
# write (OOM under the job cap is the realistic producer) otherwise vanishes
# from the totals, and a dropped failing test is a silent green.
EXPECTED_RESULTS=0
for f in "${TEST_FILES[@]}"; do
    basename="$(basename "$f")"
    if test_matches_filter "$basename"; then
        ((EXPECTED_RESULTS++))
    fi
done

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
RESULT_FILES=0
UNPARSED=0
for f in "$TEST_TMP"/result_*.json; do
    [[ -f "$f" ]] || continue
    ((RESULT_FILES++))
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
            # A known-failure that passes is a stale claim, not a green test.
            # Counted separately and always announced: swallowing it as PASS
            # is how 240 imported/native 139 tags sat green until a census
            # (docs/audit/KNOWN_FAILURE_XPAS_SIGNAL_2026-08-18.md). Same
            # lesson as vxpas below.
            ((XPAS++))
            XPAS_LIST="${XPAS_LIST}    $name
"
            echo "  XPAS  $name (known failure now passes)"
            ;;
        vxfail)
            # Tolerated because it is listed in tests/vacuous_expect_baseline.txt.
            # Counted and reported, never silently dropped: a tolerated failure
            # that does not appear in the summary is indistinguishable from a
            # test that never ran.
            ((VACUOUS_KNOWN++))
            if [[ "$VERBOSE" == "1" ]]; then
                echo "  VXFAIL  $name (vacuous-annotation baseline)"
            fi
            ;;
        vxpas)
            # Listed in the baseline but now passing. Always announced, not only
            # under --verbose: the baseline must shrink as annotations are fixed,
            # and a stale entry silently absorbing a pass is how a baseline rots
            # into a permanent mute.
            ((PASS++))
            VACUOUS_STALE="${VACUOUS_STALE}    $name
"
            ;;
        skip)
            ((SKIP++))
            reason=$(echo "$result" | grep -o '"reason":"[^"]*"' | cut -d'"' -f4)
            if [[ "$VERBOSE" == "1" ]]; then
                echo "  SKIP  $name ($reason)"
            fi
            ;;
        *)
            # A result file whose status cannot be parsed -- truncated or
            # malformed JSON, the realistic producer being a worker killed
            # mid-write. Dropping it made the totals undercount: a failing
            # test that vanishes is a silent green (the same reasoning as the
            # vxfail comment above, applied to the parse-failure path).
            # Count it and fail below instead of silently omitting it.
            ((UNPARSED++))
            echo "  UNPARSED  $f (status=[$status])" >&2
            ;;
    esac
done

# Generate output
echo ""
echo "=== Results ==="
echo "  Pass: $PASS"
echo "  Fail: $FAIL"
[[ $KNOWN_FAILURE -gt 0 ]] && echo "  Known failures: $KNOWN_FAILURE"
[[ $XPAS -gt 0 ]] && echo "  Unexpected passes (stale known-failure): $XPAS"
[[ $VACUOUS_KNOWN -gt 0 ]] && echo "  Vacuous-annotation baseline (tolerated): $VACUOUS_KNOWN"
[[ $FLAKY -gt 0 ]] && echo "  Flaky: $FLAKY"
echo "  Skip: $SKIP"
echo "  Total: $((PASS + FAIL + SKIP + KNOWN_FAILURE + VACUOUS_KNOWN + XPAS))"
[[ $UNPARSED -gt 0 ]] && echo "  Unparsed: $UNPARSED"

# Completeness: the counts above must describe every filtered test exactly
# once. Three ways to lose that, each previously silent:
#   - a result file no parser branch recognizes (UNPARSED above);
#   - a filtered test whose result file never appeared (worker died first);
#   - a run that selected nothing at all (0 == 0 reading as "all tests
#     passed" -- the vacuity failure mode; in CI an empty selection means the
#     corpus or the glob broke, so it must fail, not report green).
# CI-state readers must distinguish "the instrument did not answer" from "the
# answer was an empty selected set" (docs/governance/CI_TRUST_CONTRACT.md).
if [[ $UNPARSED -gt 0 ]]; then
    echo "INCOMPLETE: $UNPARSED result file(s) had no readable status -- verdicts unknown, failing instead of dropping them" >&2
    exit 1
fi
if [[ $RESULT_FILES -ne $EXPECTED_RESULTS ]]; then
    echo "INCOMPLETE: $EXPECTED_RESULTS filtered test(s) ran but $RESULT_FILES result file(s) were found -- a worker exited before writing its result, so the totals above undercount" >&2
    exit 1
fi
if [[ $EXPECTED_RESULTS -eq 0 ]]; then
    if [[ -n "${CI:-}" ]]; then
        echo "INCOMPLETE: zero test files matched (filter=${FILTER:-none}) -- this run measured nothing; refusing to report green on an empty suite" >&2
        exit 1
    fi
    echo "WARNING: no test files matched the active filter -- this run measured nothing" >&2
fi

if [[ -n "$XPAS_LIST" ]]; then
    echo ""
    echo "=== Known-failure tags that passed in THIS run ==="
    printf '%s' "$XPAS_LIST"
    echo "  A known-failure that passes is a stale claim about this engine,"
    echo "  not a green test. Drop the tag, or add //@ requires: <engine> if"
    echo "  the claim is about a different engine than the one that just ran."
    echo "  Madaros decides a Madaros-named tag; lean_single decides whether"
    echo "  the file needs requires: madaros. Zeros on one engine do not"
    echo "  license dropping a tag about the other."
fi

if [[ -n "$VACUOUS_STALE" ]]; then
    echo ""
    echo "=== Vacuous-annotation baseline entries that passed in THIS run ==="
    printf '%s' "$VACUOUS_STALE"
    echo "  The baseline is calibrated against the CI engine (souc-stage2 /"
    echo "  lean_single, via SOUNIO_TEST_SOUC_BIN). Under a different engine most"
    echo "  entries pass, so this list is only a removal instruction when the run"
    echo "  used the CI engine. Confirm there before deleting an entry."
fi

# Generate JUnit XML if requested
if [[ "$FORMAT" == "junit" ]]; then
    cat > "$JUNIT_FILE" << 'XMLEOF'
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
XMLEOF
    
    echo "  <testsuite name=\"sounio-test-suite\" tests=\"$((PASS + FAIL + KNOWN_FAILURE + XPAS))\" failures=\"$((FAIL + XPAS))\" skipped=\"$SKIP\" errors=\"0\">" >> "$JUNIT_FILE"
    
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
                echo "      <failure message=\"stale known-failure: test now passes on this engine\"/>" >> "$JUNIT_FILE"
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
            vxfail)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <skipped message=\"Vacuous-annotation baseline: $output\"/>" >> "$JUNIT_FILE"
                echo "    </testcase>" >> "$JUNIT_FILE"
                ;;
            vxpas)
                echo "    <testcase name=\"$name\" time=\"$time\">" >> "$JUNIT_FILE"
                echo "      <system-out>Vacuous-annotation baseline entry now passes; remove it</system-out>" >> "$JUNIT_FILE"
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

# SOUNIO_XPAS_FATAL=1 makes a stale known-failure tag fail the job.
# Default off until the remaining seed XPASses owned by other lanes
# (gum_fo_across_call, turbofish_concrete_type_mismatch) are classified.
# The Madaros known-failure recheck sets this so compiler-only PRs cannot
# rot requires:madaros tags in silence.
if [[ $XPAS -gt 0 && "${SOUNIO_XPAS_FATAL:-}" == "1" ]]; then
    echo ""
    echo "XPAS_FATAL: $XPAS known-failure tag(s) passed on this engine"
    exit 1
fi

echo ""
if [[ $XPAS -gt 0 ]]; then
    echo "Suite finished with $XPAS stale known-failure tag(s) (not a silent pass)."
else
    echo "All tests passed!"
fi
