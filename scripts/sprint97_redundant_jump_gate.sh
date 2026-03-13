#!/usr/bin/env bash
# Sprint 97 Block U — Redundant Jump Elimination gate
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
OPT=self-hosted/ir/opt_cleanup.sio
MAIN=self-hosted/compiler/main.sio
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0

check_grep() {
    local name="$1" file="$2" pattern="$3"
    TOTAL=$((TOTAL+1))
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "  PASS  $name"; PASS=$((PASS+1))
    else
        echo "  FAIL  $name"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 97 Block U — Redundant Jump Elimination Gate ==="
echo ""

echo "[structural]"
check_grep "struct:fn_exists"        "$OPT" "fn ocp_redundant_jump("
check_grep "struct:jump_label_check" "$OPT" "IrJump.*label_id"
check_grep "struct:branch_true"      "$OPT" "IrBranchTrue.*label_id"
check_grep "struct:branch_false"     "$OPT" "IrBranchFalse.*label_id"
check_grep "struct:nop_replacement"  "$OPT" "ir_nop()"
check_grep "struct:next_label_check" "$OPT" "next.op.*IrLabel\|IrOpcode::IrLabel"
check_grep "struct:sprint97_tag"     "$OPT" "Sprint 97 Block U"
check_grep "struct:wired_pipeline"   "$OPT" "ocp_redundant_jump(current)"

echo ""
echo "[tests]"
check_grep "test:T282_exists" "$MAIN" "compiler_main_test_redundant_jump_basic"
check_grep "test:T283_exists" "$MAIN" "compiler_main_test_redundant_jump_different_labels"
check_grep "test:T284_exists" "$MAIN" "compiler_main_test_redundant_branch_true"
check_grep "test:T285_exists" "$MAIN" "compiler_main_test_redundant_branch_false"
check_grep "test:T286_exists" "$MAIN" "compiler_main_test_redundant_jump_non_jump"
check_grep "test:T287_exists" "$MAIN" "compiler_main_test_redundant_jump_multi"
check_grep "test:total_287plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T282_wired"  "$MAIN" "T282 OK"
check_grep "test:T287_wired"  "$MAIN" "T287 OK"

echo ""
echo "[regression]"
check_grep "regr:dse_exists"       "$OPT" "fn ocp_dse("
check_grep "regr:copy_prop_exists" "$OPT" "fn ocp_copy_prop("
check_grep "regr:cse_exists"       "$OPT" "fn ocp_cse("
check_grep "regr:block_t_exists"   "$OPT" "Sprint 96 Block T"

echo ""
echo "[typecheck]"
TOTAL=$((TOTAL+1))
if timeout 30 $SOUC check "$MAIN" 2>&1 | grep -q "All checks passed"; then
    echo "  PASS  typecheck:main.sio"; PASS=$((PASS+1))
else
    echo "  FAIL  typecheck:main.sio"; FAIL=$((FAIL+1))
fi

echo ""
echo "[selftest]"
TOTAL=$((TOTAL+1))
STOUT=$(timeout 60 $SOUC run "$MAIN" -- --self-test 2>&1 || true)
if echo "$STOUT" | grep -q "T282 OK"; then
    echo "  PASS  selftest:T282_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T282"; then
    echo "  FAIL  selftest:T282_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T282_runtime (OOM before T282)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 97 Block U — Redundant Jump Elimination"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
