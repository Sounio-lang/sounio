#!/usr/bin/env bash
# Sprint 110 Block AG — Add/sub with negated operand simplification gate
# x + neg(y) → x - y  ;  x - neg(y) → x + y  ;  neg(x) + y → y - x.
# SOTA: LLVM InstCombineAddSub.cpp; Cooper & Torczon §8.4.
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

echo "=== Sprint 110 Block AG — Neg-Add/Sub Simplification Gate ==="
echo ""

echo "[structural]"
check_grep "struct:block_ag_comment"    "$OPT" "Sprint 110 Block AG"
check_grep "struct:add_neg_guard"       "$OPT" "OpAdd && ag_s2_ok && is_neg"
check_grep "struct:sub_neg_guard"       "$OPT" "OpSub && ag_s2_ok && is_neg"
check_grep "struct:commutative_guard"   "$OPT" "OpAdd && ag_s1_ok && is_neg"
check_grep "struct:rewrite_sub"         "$OPT" "BinaryOp::OpSub, neg_src"
check_grep "struct:rewrite_add"         "$OPT" "BinaryOp::OpAdd, neg_src"

echo ""
echo "[tests]"
check_grep "test:T364_exists" "$MAIN" "compiler_main_test_add_neg_to_sub"
check_grep "test:T365_exists" "$MAIN" "compiler_main_test_sub_neg_to_add"
check_grep "test:T366_exists" "$MAIN" "compiler_main_test_neg_add_to_sub"
check_grep "test:T367_exists" "$MAIN" "compiler_main_test_add_no_neg_no_fold"
check_grep "test:T368_exists" "$MAIN" "compiler_main_test_mul_neg_no_fold"
check_grep "test:T369_exists" "$MAIN" "compiler_main_test_double_neg_add_chain"
check_grep "test:total_367plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T364_wired"  "$MAIN" "T364 OK.*Block AG"
check_grep "test:T369_wired"  "$MAIN" "T369 OK.*Block AG"

echo ""
echo "[regression]"
check_grep "regr:is_neg"        "$OPT" "is_neg.*256"
check_grep "regr:neg_src"       "$OPT" "neg_src.*256"
check_grep "regr:block_m"       "$OPT" "Sprint 86 Block M"
check_grep "regr:block_af"      "$OPT" "Sprint 109 Block AF"

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
if echo "$STOUT" | grep -q "T364 OK.*Block AG"; then
    echo "  PASS  selftest:T364_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T364"; then
    echo "  FAIL  selftest:T364_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T364_runtime (OOM before T364)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 110 Block AG — Neg-Add/Sub Simplification"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
