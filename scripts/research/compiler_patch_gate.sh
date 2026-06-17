#!/usr/bin/env bash
# compiler_patch_gate.sh — Validation pipeline for model-generated compiler patches.
#
# Usage: bash scripts/research/compiler_patch_gate.sh <patch_file>
#
# Exit codes: 0 always (gate result reported via GATE= line on stdout).
# Used by the DRL loop: reward=1.0 if GATE=PASS, 0.0 otherwise.

set -euo pipefail

PATCH_FILE="${1:-}"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
GATE="GATE=FAIL_PATCH"
PATCHED_FILES=()
TEST_PASS=0
TEST_FAIL=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
report() {
    echo "$GATE"
    echo "TEST_PASS=$TEST_PASS"
    echo "TEST_FAIL=$TEST_FAIL"
}

restore() {
    if [[ ${#PATCHED_FILES[@]} -gt 0 ]]; then
        git -C "$REPO_ROOT" checkout -- "${PATCHED_FILES[@]}" 2>/dev/null || true
    fi
}

# Always restore on exit
trap restore EXIT

# ---------------------------------------------------------------------------
# Step 1: validate input
# ---------------------------------------------------------------------------
if [[ -z "$PATCH_FILE" ]]; then
    echo "Usage: $0 <patch_file>" >&2
    report
    exit 0
fi

if [[ ! -f "$PATCH_FILE" ]]; then
    echo "ERROR: patch file not found: $PATCH_FILE" >&2
    report
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 2: extract file list from patch for targeted restore
# ---------------------------------------------------------------------------
while IFS= read -r line; do
    if [[ "$line" =~ ^\+\+\+\ b/(.+)$ ]]; then
        PATCHED_FILES+=("${BASH_REMATCH[1]}")
    fi
done < "$PATCH_FILE"

# ---------------------------------------------------------------------------
# Step 3: dry-run check
# ---------------------------------------------------------------------------
if ! git -C "$REPO_ROOT" apply --check "$PATCH_FILE" 2>/dev/null; then
    echo "git apply --check failed" >&2
    GATE="GATE=FAIL_PATCH"
    report
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 4: apply patch
# ---------------------------------------------------------------------------
if ! git -C "$REPO_ROOT" apply "$PATCH_FILE" 2>/dev/null; then
    echo "git apply failed" >&2
    GATE="GATE=FAIL_PATCH"
    report
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 5: bootstrap fixed-point check
# ---------------------------------------------------------------------------
BUILD_OUT="$(make -C "$REPO_ROOT" build 2>&1)" || true
if ! echo "$BUILD_OUT" | grep -q "FIXED POINT OK"; then
    echo "Bootstrap fixed-point check failed" >&2
    GATE="GATE=FAIL_FIXEDPOINT"
    report
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 6: run test suite
# ---------------------------------------------------------------------------
TEST_OUT="$(bash "$REPO_ROOT/scripts/run_sio_test_suite.sh" --timeout 120 2>&1)" || true

# Parse "X passed" / "X failed" style lines from test output
PASS_LINE="$(echo "$TEST_OUT" | grep -oE '[0-9]+ passed' | tail -1 || true)"
FAIL_LINE="$(echo "$TEST_OUT" | grep -oE '[0-9]+ failed' | tail -1 || true)"
TEST_PASS="${PASS_LINE%% *}"
TEST_FAIL="${FAIL_LINE%% *}"
TEST_PASS="${TEST_PASS:-0}"
TEST_FAIL="${TEST_FAIL:-0}"

TOTAL=$(( TEST_PASS + TEST_FAIL ))
if [[ "$TOTAL" -eq 0 ]]; then
    # Fallback: count PASS/FAIL prefixed lines
    TEST_PASS="$(echo "$TEST_OUT" | grep -c '^PASS' || true)"
    TEST_FAIL="$(echo "$TEST_OUT" | grep -c '^FAIL' || true)"
    TOTAL=$(( TEST_PASS + TEST_FAIL ))
fi

# ---------------------------------------------------------------------------
# Step 7: evaluate pass rate
# ---------------------------------------------------------------------------
if [[ "$TOTAL" -eq 0 ]]; then
    echo "WARNING: no test results detected; treating as FAIL_TESTS" >&2
    GATE="GATE=FAIL_TESTS"
elif awk "BEGIN { exit !(($TEST_PASS / $TOTAL) >= 0.95) }"; then
    GATE="GATE=PASS"
else
    GATE="GATE=FAIL_TESTS"
fi

report
exit 0
