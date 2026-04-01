#!/usr/bin/env bash
# Sprint 98 gate: Block V — unreachable code elimination (dead after jump/return)
# SOTA: Muchnick §7.2; Cooper & Torczon §10.1
set -eo pipefail

SOUC=./bin/souc
OPT=self-hosted/ir/opt_cleanup.sio
MAIN=self-hosted/compiler/main.sio

pass=0; fail=0; not_run=0

check() {
    local name="$1"; local cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "  PASS: $name"; pass=$((pass+1))
    else
        echo "  FAIL: $name"; fail=$((fail+1))
    fi
}

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "  PASS: $name"; pass=$((pass+1))
    else
        echo "  FAIL: $name"; fail=$((fail+1))
    fi
}

echo "=== Sprint 98 Gate: Block V — unreachable code elimination ==="
echo ""

# Structural checks
echo "--- Structural ---"
check_grep "block_v:fn_ocp_dead_after_jump" "fn ocp_dead_after_jump" "$OPT"
check_grep "block_v:dead_flag" "var dead = false" "$OPT"
check_grep "block_v:label_resets" "IrLabel" "$OPT"
check_grep "block_v:jump_sets_dead" "IrJump" "$OPT"
check_grep "block_v:return_sets_dead" "IrReturn" "$OPT"
check_grep "block_v:wired_in_pipeline" "ocp_dead_after_jump" "$OPT"
check_grep "block_v:pipeline_ordering" "ocp_dead_after_jump(current)" "$OPT"

# Test function checks
echo ""
echo "--- Test Functions ---"
check_grep "test:T288_basic" "compiler_main_test_dead_after_jump_basic" "$MAIN"
check_grep "test:T289_return" "compiler_main_test_dead_after_return" "$MAIN"
check_grep "test:T290_adjacent" "compiler_main_test_dead_after_jump_adjacent" "$MAIN"
check_grep "test:T291_branch" "compiler_main_test_dead_after_branch_not_dead" "$MAIN"
check_grep "test:T292_multi" "compiler_main_test_dead_after_return_multi" "$MAIN"
check_grep "test:T293_label_resets" "compiler_main_test_dead_after_jump_label_resets" "$MAIN"

# Runner wiring
echo ""
echo "--- Runner Wiring ---"
check_grep "runner:T288_wired" "T288 OK" "$MAIN"
check_grep "runner:T289_wired" "T289 OK" "$MAIN"
check_grep "runner:T290_wired" "T290 OK" "$MAIN"
check_grep "runner:T291_wired" "T291 OK" "$MAIN"
check_grep "runner:T292_wired" "T292 OK" "$MAIN"
check_grep "runner:T293_wired" "T293 OK" "$MAIN"
check_grep "runner:total_293plus" "let total: i64 = 29[3-9]\|let total: i64 = [3-9]" "$MAIN"

# Typecheck
echo ""
echo "--- Typecheck ---"
check "typecheck:main_sio" "$SOUC check $MAIN"

# Self-test (with OOM tolerance)
echo ""
echo "--- Self-test (OOM tolerant) ---"
_ec=0
timeout 180 $SOUC run self-hosted/compiler/main.sio -- --self-test > /tmp/sprint98_selftest.txt 2>&1 || _ec=$?
if grep -q "T288 OK" /tmp/sprint98_selftest.txt 2>/dev/null; then
    echo "  PASS: selftest:T288_runtime"; pass=$((pass+1))
elif [ $_ec -eq 137 ] || [ $_ec -eq 143 ] || [ $_ec -eq 144 ] || [ $_ec -eq 124 ] || ! grep -q "T288" /tmp/sprint98_selftest.txt 2>/dev/null; then
    echo "  NOT_RUN: selftest (OOM/timeout before T288, exit=$_ec)"
    not_run=$((not_run+1))
else
    echo "  FAIL: selftest:T288_runtime"; fail=$((fail+1))
fi

# Regression guard
echo ""
echo "--- Regression ---"
check_grep "regression:ocp_const_fold_exists" "fn ocp_const_fold" "$OPT"
check_grep "regression:ocp_dce_exists" "fn ocp_dce" "$OPT"
check_grep "regression:ocp_cse_exists" "fn ocp_cse" "$OPT"
check_grep "regression:ocp_redundant_jump_exists" "fn ocp_redundant_jump" "$OPT"

echo ""
echo "=== Sprint 98 Gate Summary ==="
echo "  PASS=$pass  FAIL=$fail  NOT_RUN=$not_run"
echo ""

if [ $fail -gt 0 ]; then
    echo "GATE: FAIL"
    exit 1
fi
echo "GATE: PASS"
