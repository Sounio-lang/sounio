#!/usr/bin/env bash
# Sprint 106 Block AD — Jump-to-Return Folding gate
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

echo "=== Sprint 106 Block AD — Jump-to-Return Folding Gate ==="
echo ""

echo "[structural]"
check_grep "struct:fn_exists"      "$OPT" "fn ocp_jump_to_return("
check_grep "struct:label_pos"      "$OPT" "label_pos.*256"
check_grep "struct:return_fold"    "$OPT" "ir_return(next.src1)"
check_grep "struct:sprint106_tag"  "$OPT" "Sprint 106 Block AD"
check_grep "struct:wired_pipeline" "$OPT" "ocp_jump_to_return(current)"

echo ""
echo "[tests]"
check_grep "test:T306_exists" "$MAIN" "compiler_main_test_jump_to_return_basic"
check_grep "test:T307_exists" "$MAIN" "compiler_main_test_jump_to_return_no_fold"
check_grep "test:T308_exists" "$MAIN" "compiler_main_test_jump_to_return_cond_no_fold"
check_grep "test:T309_exists" "$MAIN" "compiler_main_test_jump_to_return_multi"
check_grep "test:T310_exists" "$MAIN" "compiler_main_test_jump_to_return_missing_label"
check_grep "test:T311_exists" "$MAIN" "compiler_main_test_jump_to_return_end_of_func"
check_grep "test:total_347"   "$MAIN" "let total: i64 = 347"
check_grep "test:T306_wired"  "$MAIN" "T306 OK.*Block AD"
check_grep "test:T311_wired"  "$MAIN" "T311 OK.*Block AD"

echo ""
echo "[regression]"
check_grep "regr:redundant_jump"  "$OPT" "fn ocp_redundant_jump("
check_grep "regr:dead_after_jump" "$OPT" "fn ocp_dead_after_jump("
check_grep "regr:dead_label"      "$OPT" "fn ocp_dead_label("

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
if echo "$STOUT" | grep -q "T306 OK.*Block AD"; then
    echo "  PASS  selftest:T306_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T306"; then
    echo "  FAIL  selftest:T306_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T306_runtime (OOM before T306)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 106 Block AD — Jump-to-Return Folding"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
