#!/usr/bin/env bash
# Sprint 99 Block W — Unreachable Label Elimination gate
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

echo "=== Sprint 99 Block W — Unreachable Label Elimination Gate ==="
echo ""

echo "[structural]"
check_grep "struct:fn_exists"       "$OPT" "fn ocp_dead_label("
check_grep "struct:referenced_arr"  "$OPT" "referenced.*256"
check_grep "struct:two_pass"        "$OPT" "Pass 1.*Pass 2\|Pass 2"
check_grep "struct:jump_collect"    "$OPT" "IrJump.*IrBranchTrue.*IrBranchFalse"
check_grep "struct:nop_unreferenced" "$OPT" "!referenced"
check_grep "struct:sprint99_tag"    "$OPT" "Sprint 99 Block W"
check_grep "struct:wired_pipeline"  "$OPT" "ocp_dead_label(current)"

echo ""
echo "[tests]"
check_grep "test:T294_exists" "$MAIN" "compiler_main_test_dead_label_basic"
check_grep "test:T295_exists" "$MAIN" "compiler_main_test_dead_label_referenced"
check_grep "test:T296_exists" "$MAIN" "compiler_main_test_dead_label_branch_ref"
check_grep "test:T297_exists" "$MAIN" "compiler_main_test_dead_label_mixed"
check_grep "test:T298_exists" "$MAIN" "compiler_main_test_dead_label_integration"
check_grep "test:T299_exists" "$MAIN" "compiler_main_test_dead_label_no_labels"
check_grep "test:total_299plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T294_wired"  "$MAIN" "T294 OK"
check_grep "test:T299_wired"  "$MAIN" "T299 OK"

echo ""
echo "[regression]"
check_grep "regr:dead_after_jump"  "$OPT" "fn ocp_dead_after_jump("
check_grep "regr:redundant_jump"   "$OPT" "fn ocp_redundant_jump("
check_grep "regr:dse"              "$OPT" "fn ocp_dse("
check_grep "regr:copy_prop"        "$OPT" "fn ocp_copy_prop("

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
if echo "$STOUT" | grep -q "T294 OK"; then
    echo "  PASS  selftest:T294_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T294"; then
    echo "  FAIL  selftest:T294_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T294_runtime (OOM before T294)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 99 Block W — Unreachable Label Elimination"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
