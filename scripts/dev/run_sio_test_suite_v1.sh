#!/usr/bin/env bash
# Self-hosted Sounio test suite runner
#
# Runs all .sio tests in tests/run-pass/ and tests/compile-fail/
# using the souc binary (resolved via omega infrastructure).
#
# Annotations:
#   //@ run-pass         — expect exit 0
#   //@ compile-fail     — expect exit != 0
#   //@ expect-stdout: X — stdout must contain X (run-pass only)
#   //@ error-pattern: X — stderr/stdout must contain X (compile-fail only)
#   //@ ignore           — skip this test
#   //@ check-only       — compile only, do not execute
#
# Usage:
#   bash scripts/run_sio_test_suite.sh [--filter PATTERN] [--verbose]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/../lib/resolve_souc.sh" && -d "$SCRIPT_DIR/../../tests" ]]; then
    ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
else
    ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
cd "$ROOT_DIR"

# Harness-local override: keep repo-wide default behavior untouched unless
# callers explicitly point the suite at an alternate compiler path/backend.
if [[ -n "${SOUNIO_TEST_SOUC_BIN:-}" ]]; then
    export SOUNIO_SOUC_BIN="$SOUNIO_TEST_SOUC_BIN"
    unset SOUC_BIN
fi
if [[ -n "${SOUNIO_TEST_NATIVE_BIN:-}" ]]; then
    export SOUC_NATIVE_BIN="$SOUNIO_TEST_NATIVE_BIN"
fi

# shellcheck source=lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

# Ensure stdlib is discoverable for tests that import from it.
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

FILTER="${1:-}"
VERBOSE="${2:-}"
if [[ "$FILTER" == "--verbose" ]]; then
    VERBOSE="--verbose"
    FILTER=""
fi

PASS=0
FAIL=0
SKIP=0
ERRORS=""

run_test() {
    local file="$1"
    local basename
    basename="$(basename "$file")"

    # Apply filter
    if [[ -n "$FILTER" && "$FILTER" != "--verbose" ]]; then
        if [[ "$basename" != *"$FILTER"* ]]; then
            return
        fi
    fi

    # Read annotations from file header
    local is_run_pass=false
    local is_compile_fail=false
    local is_ignored=false
    local is_check_only=false
    local expect_stdout=()
    local error_patterns=()

    while IFS= read -r line; do
        # Stop at first non-comment/non-empty line
        if [[ ! "$line" =~ ^[[:space:]]*//@\  && ! "$line" =~ ^[[:space:]]*//\  && ! "$line" =~ ^[[:space:]]*$ ]]; then
            break
        fi
        if [[ "$line" =~ "//@ run-pass" ]]; then
            is_run_pass=true
        fi
        if [[ "$line" =~ "//@ compile-fail" ]]; then
            is_compile_fail=true
        fi
        if [[ "$line" =~ "//@ ignore" ]]; then
            is_ignored=true
        fi
        if [[ "$line" =~ "//@ check-only" ]]; then
            is_check_only=true
        fi
        if [[ "$line" =~ "//@ expect-stdout: "(.*) ]]; then
            expect_stdout+=("${BASH_REMATCH[1]}")
        fi
        if [[ "$line" =~ "//@ error-pattern: "(.*) ]]; then
            error_patterns+=("${BASH_REMATCH[1]}")
        fi
    done < "$file"

    # Skip ignored tests
    if $is_ignored; then
        SKIP=$((SKIP + 1))
        if [[ "$VERBOSE" == "--verbose" ]]; then
            echo "  SKIP  $basename"
        fi
        return
    fi

    # Skip tests with no annotation
    if ! $is_run_pass && ! $is_compile_fail; then
        SKIP=$((SKIP + 1))
        return
    fi

    # Execute
    local output
    local exit_code=0
    if $is_run_pass; then
        # Per-test timeout (seconds); override via SOUNIO_TEST_TIMEOUT env var
        local _timeout="${SOUNIO_TEST_TIMEOUT:-30}"
        if $is_check_only; then
            output=$(timeout "$_timeout" "$SOUC_BIN" check "$file" 2>&1) || exit_code=$?
            if [[ $exit_code -eq 124 ]]; then
                FAIL=$((FAIL + 1))
                ERRORS="${ERRORS}\n  FAIL  $basename (check timed out after ${_timeout}s)"
                if [[ "$VERBOSE" == "--verbose" ]]; then
                    echo "  FAIL  $basename (check timed out after ${_timeout}s)"
                fi
                return
            fi
            if [[ $exit_code -ne 0 ]]; then
                FAIL=$((FAIL + 1))
                ERRORS="${ERRORS}\n  FAIL  $basename (check exited $exit_code)"
                if [[ "$VERBOSE" == "--verbose" ]]; then
                    echo "  FAIL  $basename (check exited $exit_code)"
                    echo "        $output" | head -5
                fi
                return
            fi
        else
            output=$(timeout "$_timeout" "$SOUC_BIN" run "$file" 2>&1) || exit_code=$?
            if [[ $exit_code -eq 124 ]]; then
                FAIL=$((FAIL + 1))
                ERRORS="${ERRORS}\n  FAIL  $basename (run timed out after ${_timeout}s)"
                if [[ "$VERBOSE" == "--verbose" ]]; then
                    echo "  FAIL  $basename (run timed out after ${_timeout}s)"
                fi
                return
            fi
            if [[ $exit_code -ne 0 ]]; then
                FAIL=$((FAIL + 1))
                ERRORS="${ERRORS}\n  FAIL  $basename (run exited $exit_code)"
                if [[ "$VERBOSE" == "--verbose" ]]; then
                    echo "  FAIL  $basename (run exited $exit_code)"
                    echo "        $output" | head -5
                fi
                return
            fi
        fi

        for pattern in "${expect_stdout[@]}"; do
            if ! echo "$output" | grep -qF "$pattern"; then
                FAIL=$((FAIL + 1))
                ERRORS="${ERRORS}\n  FAIL  $basename (missing stdout: $pattern)"
                if [[ "$VERBOSE" == "--verbose" ]]; then
                    echo "  FAIL  $basename (missing stdout: $pattern)"
                fi
                return
            fi
        done

        PASS=$((PASS + 1))
        if [[ "$VERBOSE" == "--verbose" ]]; then
            echo "  PASS  $basename"
        fi

    elif $is_compile_fail; then
        local tmp_out
        tmp_out="$(mktemp /tmp/sounio-compile-fail-XXXXXX.elf)"
        local _timeout="${SOUNIO_TEST_TIMEOUT:-30}"
        output=$(timeout "$_timeout" "$SOUC_BIN" compile "$file" -o "$tmp_out" 2>&1) || exit_code=$?
        rm -f "$tmp_out"
        if [[ $exit_code -eq 124 ]]; then
            FAIL=$((FAIL + 1))
            ERRORS="${ERRORS}\n  FAIL  $basename (compile timed out after ${_timeout}s)"
            if [[ "$VERBOSE" == "--verbose" ]]; then
                echo "  FAIL  $basename (compile timed out after ${_timeout}s)"
            fi
            return
        fi
        if [[ $exit_code -eq 0 ]]; then
            FAIL=$((FAIL + 1))
            ERRORS="${ERRORS}\n  FAIL  $basename (expected compile failure but passed)"
            if [[ "$VERBOSE" == "--verbose" ]]; then
                echo "  FAIL  $basename (expected compile failure but passed)"
            fi
            return
        fi

        # Check error patterns
        for pattern in "${error_patterns[@]}"; do
            if ! echo "$output" | grep -qiF "$pattern"; then
                FAIL=$((FAIL + 1))
                ERRORS="${ERRORS}\n  FAIL  $basename (missing error: $pattern)"
                if [[ "$VERBOSE" == "--verbose" ]]; then
                    echo "  FAIL  $basename (missing error: $pattern)"
                fi
                return
            fi
        done

        PASS=$((PASS + 1))
        if [[ "$VERBOSE" == "--verbose" ]]; then
            echo "  PASS  $basename"
        fi
    fi
}

echo "=== Sounio Test Suite ==="
echo "Using souc: $SOUC_BIN"
if [[ -n "${SOUC_NATIVE_BIN:-}" ]]; then
    echo "Using native backend: $SOUC_NATIVE_BIN"
fi
echo ""

echo "--- run-pass tests ---"
for f in "$ROOT_DIR"/tests/run-pass/*.sio; do
    run_test "$f"
done

echo "--- compile-fail tests ---"
for f in "$ROOT_DIR"/tests/compile-fail/*.sio; do
    run_test "$f"
done

echo "--- ui tests ---"
for f in "$ROOT_DIR"/tests/ui/type/*.sio "$ROOT_DIR"/tests/ui/effect/*.sio "$ROOT_DIR"/tests/ui/ownership/*.sio "$ROOT_DIR"/tests/ui/resolve/*.sio "$ROOT_DIR"/tests/ui/pattern/*.sio; do
    [ -f "$f" ] && run_test "$f"
done

echo "--- stdlib e2e tests ---"
for f in "$ROOT_DIR"/tests/stdlib/*/test_*.sio; do
    [ -f "$f" ] && run_test "$f"
done

echo ""
echo "=== Results ==="
echo "  Pass: $PASS"
echo "  Fail: $FAIL"
echo "  Skip: $SKIP"
echo "  Total: $((PASS + FAIL + SKIP))"

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo "=== Failures ==="
    echo -e "$ERRORS"
    exit 1
fi

echo ""
echo "All tests passed."
