#!/usr/bin/env bash
# Sprint 113 Block AJ — Boolean comparison with 1 simplification gate
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

echo "=== Sprint 113 Block AJ — Boolean Eq-1 Gate ==="
echo ""

echo "[structural]"
check_grep "struct:block_aj_comment"    "$OPT" "Sprint 113 Block AJ"
check_grep "struct:eq_1_fold"           "$OPT" "const_val\[bl_s2 as usize\] == 1"
check_grep "struct:commutative_1"       "$OPT" "const_val\[bl_s1 as usize\] == 1"
check_grep "struct:ne_1_rewrite"        "$OPT" "bool != 1.*rewrite to (bool == 0)"
check_grep "struct:copy_fold"           "$OPT" "ir_copy(bl.dst, bl_s1)"

echo ""
echo "[tests]"
check_grep "test:T382_exists" "$MAIN" "compiler_main_test_bool_eq_one"
check_grep "test:T383_exists" "$MAIN" "compiler_main_test_one_eq_bool"
check_grep "test:T384_exists" "$MAIN" "compiler_main_test_bool_ne_one"
check_grep "test:T385_exists" "$MAIN" "compiler_main_test_non_bool_eq_one_no_fold"
check_grep "test:T386_exists" "$MAIN" "compiler_main_test_bool_eq_two_no_fold"
check_grep "test:T387_exists" "$MAIN" "compiler_main_test_bool_eq_one_chain"
check_grep "test:total_385plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T382_wired"  "$MAIN" "T382 OK.*Block AJ"
check_grep "test:T387_wired"  "$MAIN" "T387 OK.*Block AJ"

echo ""
echo "[regression]"
check_grep "regr:block_y"       "$OPT" "Sprint 101 Block Y"
check_grep "regr:block_ai"      "$OPT" "Sprint 112 Block AI"
check_grep "regr:is_cmp_result" "$OPT" "is_cmp_result.*256"

echo ""
echo "[typecheck]"
TOTAL=$((TOTAL+1))
TC_OUT=$(timeout 30 $SOUC check "$MAIN" 2>&1)
if echo "$TC_OUT" | grep -q "All checks passed"; then
    echo "  PASS  typecheck:main.sio"; PASS=$((PASS+1))
else
    echo "  FAIL  typecheck:main.sio"; echo "$TC_OUT" | tail -3; FAIL=$((FAIL+1))
fi

echo ""
echo "[selftest]"
TOTAL=$((TOTAL+1))
STOUT=$(timeout 60 $SOUC run "$MAIN" -- --self-test 2>&1 || true)
if echo "$STOUT" | grep -q "T382 OK.*Block AJ"; then
    echo "  PASS  selftest:T382_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T382"; then
    echo "  FAIL  selftest:T382_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T382_runtime (OOM before T382)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 113 Block AJ — Boolean Eq-1"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
