#!/bin/bash
# Demetrios stdlib test runner
# Runs ALL stdlib D programs with main() functions
# Exit code != 0 blocks merge

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DC="${PROJECT_ROOT}/compiler/target/release/dc"

if [ ! -f "$DC" ]; then
    echo "ERROR: Compiler not found at $DC"
    echo "Run: cd compiler && cargo build --release"
    exit 1
fi

# Find timeout command (gtimeout on macOS, timeout on Linux)
if command -v gtimeout &> /dev/null; then
    TIMEOUT_CMD="gtimeout"
elif command -v timeout &> /dev/null; then
    TIMEOUT_CMD="timeout"
else
    # No timeout available, run without timeout
    TIMEOUT_CMD=""
fi

# Known broken files that need repair
# Each entry MUST have a tracking issue or be fixed
# Format: path:reason
KNOWN_BROKEN=(
    # Parse errors - reserved keywords
    "stdlib/autodiff/tape.d:reserved keyword 'Grad'"
    "stdlib/epistemic/causal.d:reserved keyword 'Counterfactual'"
    "stdlib/epistemic/test_causal.d:reserved keyword 'Counterfactual'"
    # Parse errors - character issues
    "stdlib/async/executor.d:unsupported char literal"
    "stdlib/async/mod.d:unsupported char literal"
    "stdlib/profile/async_profile.d:unsupported escape sequence"
    "stdlib/profile/mod.d:unsupported escape sequence"
    # Parse errors - syntax
    "stdlib/epistemic/test_stats.d:missing semicolon"
    "stdlib/nn/autograd.d:duplicate definition"
    # Runtime issues
    "stdlib/darwin_pbpk/core/rodgers_rowland.d:codegen issue"
    "stdlib/darwin_pbpk/simulation.d:codegen issue"
    "stdlib/darwin_pbpk/tsit5_pbpk14.d:timeout (needs optimization)"
)

# Modules that require special setup
SKIP_MODULES=(
    "stdlib/quantum/"
    "stdlib/ml/"
)

is_known_broken() {
    local file="$1"
    local rel_path="${file#$PROJECT_ROOT/}"
    for entry in "${KNOWN_BROKEN[@]}"; do
        local path="${entry%%:*}"
        if [[ "$rel_path" == "$path" ]]; then
            return 0
        fi
    done
    return 1
}

get_broken_reason() {
    local file="$1"
    local rel_path="${file#$PROJECT_ROOT/}"
    for entry in "${KNOWN_BROKEN[@]}"; do
        local path="${entry%%:*}"
        local reason="${entry#*:}"
        if [[ "$rel_path" == "$path" ]]; then
            echo "$reason"
            return
        fi
    done
}

should_skip() {
    local file="$1"
    for pattern in "${SKIP_MODULES[@]}"; do
        if [[ "$file" == *"$pattern"* ]]; then
            return 0
        fi
    done
    return 1
}

PASSED=0
FAILED=0
SKIPPED=0
KNOWN=0
NEW_FAILURES=""

echo "=========================================="
echo "Demetrios stdlib test suite"
echo "=========================================="
echo ""

for f in $(find "$PROJECT_ROOT/stdlib" -name "*.d" -type f | sort); do
    if ! grep -q "fn main" "$f"; then
        continue
    fi

    rel_path="${f#$PROJECT_ROOT/}"

    if should_skip "$f"; then
        echo "[SKIP] $rel_path (module requires special setup)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    if is_known_broken "$f"; then
        reason=$(get_broken_reason "$f")
        echo "[KNOWN] $rel_path ($reason)"
        KNOWN=$((KNOWN + 1))
        continue
    fi

    echo -n "[TEST] $rel_path ... "

    # Use timeout command if available, otherwise run without timeout
    if [ -n "$TIMEOUT_CMD" ]; then
        RUN_CMD="$TIMEOUT_CMD 30 $DC run $f"
    else
        RUN_CMD="$DC run $f"
    fi

    if $RUN_CMD > /dev/null 2>&1; then
        echo "PASS"
        PASSED=$((PASSED + 1))
    else
        EXIT_CODE=$?
        echo "FAIL (exit: $EXIT_CODE)"
        FAILED=$((FAILED + 1))
        NEW_FAILURES="$NEW_FAILURES\n  $rel_path"
    fi
done

echo ""
echo "=========================================="
echo "Results:"
echo "  Passed:       $PASSED"
echo "  Failed:       $FAILED"
echo "  Known broken: $KNOWN"
echo "  Skipped:      $SKIPPED"
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "NEW FAILURES (not in known list):$NEW_FAILURES"
    echo ""
    echo "Either fix these tests or add them to KNOWN_BROKEN with a reason."
    exit 1
fi

if [ $KNOWN -gt 0 ]; then
    echo ""
    echo "WARNING: $KNOWN known broken tests need repair"
    echo "These are tracked but blocking future regressions."
fi

echo ""
echo "All active tests passed!"
exit 0
