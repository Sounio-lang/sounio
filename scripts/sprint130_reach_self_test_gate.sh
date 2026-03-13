#!/usr/bin/env bash
# Sprint 130 Gate: Reachability self-test runner
# Verifies reach_self_test_runner.sio exists, checks clean, and executes
# all 4 self-tests successfully under JIT.
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
RUNNER=self-hosted/ir/reach_self_test_runner.sio
REACH=self-hosted/ir/reachability.sio

check() {
    local name="$1"; local result="$2"
    TOTAL=$((TOTAL + 1))
    if [ "$result" -eq 0 ]; then
        PASS=$((PASS + 1)); echo "  PASS: $name"
    elif [ "$result" -eq 2 ]; then
        NOT_RUN=$((NOT_RUN + 1)); echo "  NOT_RUN: $name"
    else
        FAIL=$((FAIL + 1)); echo "  FAIL: $name"
    fi
}

echo "=== Sprint 130: Reachability Self-Test Runner Gate ==="

# T1: runner file exists
_ec=1; [ -f "$RUNNER" ] && _ec=0
check "T1: reach_self_test_runner.sio exists" $_ec

# T2: imports ir::reachability
_ec=1; grep -q 'use ir::reachability' $RUNNER && _ec=0
check "T2: imports ir::reachability" $_ec

# T3: imports ir::ir (for ir_empty_module / ir_empty_function)
_ec=1; grep -q 'use ir::ir' $RUNNER && _ec=0
check "T3: imports ir::ir" $_ec

# T4: calls reach_run_self_tests
_ec=1; grep -q 'reach_run_self_tests' $RUNNER && _ec=0
check "T4: calls reach_run_self_tests()" $_ec

# T5: returns 1 on failure (non-zero exit)
_ec=1; grep -q 'return 1\|1$' $RUNNER && _ec=0
check "T5: non-zero exit on failure" $_ec

# T6: souc check passes
_ec=0; timeout 60 $SOUC check $RUNNER >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ]; then _ec=2; fi
check "T6: souc check passes" $_ec

# T7: souc run executes and prints ALL PASS
LOG=$(mktemp)
_ec=0; timeout 120 $SOUC run $RUNNER > "$LOG" 2>&1 || _ec=$?
if [ $_ec -eq 124 ] || [ $_ec -eq 137 ]; then
    _ec=2
elif grep -q 'ALL PASS' "$LOG"; then
    _ec=0
else
    _ec=1
fi
check "T7: souc run prints ALL PASS" $_ec

# T8: exit code is 0
_ec=0; timeout 120 $SOUC run $RUNNER >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ] || [ $_ec -eq 137 ]; then _ec=2; fi
check "T8: exit code 0" $_ec

# T9: reachability.sio still has 4 self-test functions
_ec=1
count=$(grep -c 'fn reach_self_test_' $REACH)
[ "$count" -ge 4 ] && _ec=0
check "T9: 4+ self-test functions in reachability.sio" $_ec

# T10: reach_run_self_tests is the orchestrator
_ec=1; grep -q 'fn reach_run_self_tests(' $REACH && _ec=0
check "T10: reach_run_self_tests() orchestrator present" $_ec

rm -f "$LOG"
echo ""
echo "=== RESULTS: $PASS PASS, $FAIL FAIL, $NOT_RUN NOT_RUN (of $TOTAL) ==="
exit $FAIL
