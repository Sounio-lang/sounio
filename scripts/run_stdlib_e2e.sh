#!/usr/bin/env bash
# Stdlib E2E test runner
#
# Runs all .sio tests in tests/stdlib/**/ using the souc binary.
# Each test imports from the real stdlib to validate E2E correctness.
#
# Annotations (same as run_sio_test_suite.sh):
#   //@ run-pass         — expect souc check to pass (exit 0)
#   //@ check-only       — only run souc check, not souc run
#   //@ expect-stdout: X — stdout must contain X (run-pass only)
#   //@ ignore           — skip this test
#
# Usage:
#   bash scripts/run_stdlib_e2e.sh [--filter PATTERN] [--verbose]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

# Stdlib path is mandatory for E2E tests.
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
    local relpath="${file#$ROOT_DIR/}"

    # Apply filter
    if [[ -n "$FILTER" && "$FILTER" != "--verbose" ]]; then
        if [[ "$relpath" != *"$FILTER"* ]]; then
            return
        fi
    fi

    # Read annotations from file header
    local is_run_pass=false
    local is_check_only=false
    local is_ignored=false
    local expect_stdout=()

    while IFS= read -r line; do
        if [[ ! "$line" =~ ^[[:space:]]*//@\  && ! "$line" =~ ^[[:space:]]*//\  && ! "$line" =~ ^[[:space:]]*$ ]]; then
            break
        fi
        if [[ "$line" =~ "//@ run-pass" ]]; then
            is_run_pass=true
        fi
        if [[ "$line" =~ "//@ check-only" ]]; then
            is_check_only=true
        fi
        if [[ "$line" =~ "//@ ignore" ]]; then
            is_ignored=true
        fi
        if [[ "$line" =~ "//@ expect-stdout: "(.*) ]]; then
            expect_stdout+=("${BASH_REMATCH[1]}")
        fi
    done < "$file"

    # Skip ignored tests
    if $is_ignored; then
        SKIP=$((SKIP + 1))
        if [[ "$VERBOSE" == "--verbose" ]]; then
            echo "  SKIP  $relpath"
        fi
        return
    fi

    # Skip tests with no annotation
    if ! $is_run_pass && ! $is_check_only; then
        SKIP=$((SKIP + 1))
        return
    fi

    # Run souc check
    local output
    local exit_code=0
    output=$("$SOUC_BIN" check "$file" 2>&1) || exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        FAIL=$((FAIL + 1))
        ERRORS="${ERRORS}\n  FAIL  $relpath (check exited $exit_code)"
        if [[ "$VERBOSE" == "--verbose" ]]; then
            echo "  FAIL  $relpath (check exited $exit_code)"
            echo "        $(echo "$output" | head -3)"
        fi
        return
    fi

    # check-only stops here
    if $is_check_only; then
        PASS=$((PASS + 1))
        if [[ "$VERBOSE" == "--verbose" ]]; then
            echo "  PASS  $relpath (check-only)"
        fi
        return
    fi

    # run-pass: check expect-stdout patterns
    if [[ ${#expect_stdout[@]} -gt 0 ]]; then
        exit_code=0
        output=$("$SOUC_BIN" run "$file" 2>&1) || exit_code=$?
        if [[ $exit_code -ne 0 ]]; then
            FAIL=$((FAIL + 1))
            ERRORS="${ERRORS}\n  FAIL  $relpath (run exited $exit_code)"
            if [[ "$VERBOSE" == "--verbose" ]]; then
                echo "  FAIL  $relpath (run exited $exit_code)"
            fi
            return
        fi
        for pattern in "${expect_stdout[@]}"; do
            if ! echo "$output" | grep -qF "$pattern"; then
                FAIL=$((FAIL + 1))
                ERRORS="${ERRORS}\n  FAIL  $relpath (missing stdout: $pattern)"
                if [[ "$VERBOSE" == "--verbose" ]]; then
                    echo "  FAIL  $relpath (missing stdout: $pattern)"
                fi
                return
            fi
        done
    fi

    PASS=$((PASS + 1))
    if [[ "$VERBOSE" == "--verbose" ]]; then
        echo "  PASS  $relpath"
    fi
}

echo "=== Stdlib E2E Test Suite ==="
echo "Using souc: $SOUC_BIN"
echo "Stdlib path: $SOUNIO_STDLIB_PATH"
echo ""

# Walk all .sio files under tests/stdlib/
shopt -s globstar nullglob
for f in "$ROOT_DIR"/tests/stdlib/**/*.sio; do
    run_test "$f"
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
echo "All stdlib E2E tests passed."
