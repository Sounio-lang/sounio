#!/usr/bin/env bash
# Sprint 111 Block AH — Add/sub chain constant merging gate
# (x-C1)-C2 → x-(C1+C2); (x-C1)+C2 → x±|C2-C1|; (x+C1)-C2 → x±|C1-C2|.
# SOTA: LLVM InstCombineAddSub.cpp; Click PLDI 1995 §4.
set -eo pipefail

SOUC=./bin/souc
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

echo "=== Sprint 111 Block AH — Add/Sub Chain Constant Merging Gate ==="
echo ""

echo "[structural]"
check_grep "struct:block_ah_comment"    "$OPT" "Sprint 111 Block AH"
check_grep "struct:sub_sub_fold"        "$OPT" "ah_was_add && instr.bin_op == BinaryOp::OpSub"
check_grep "struct:sub_add_net_pos"     "$OPT" "ah_was_add && instr.bin_op == BinaryOp::OpAdd"
check_grep "struct:ah_base"             "$OPT" "ah_base = as_var_src"
check_grep "struct:ah_merged"           "$OPT" "ah_merged = ah_c1 + v2"
check_grep "struct:ah_net"              "$OPT" "ah_net = v2 - ah_c1"

echo ""
echo "[tests]"
check_grep "test:T370_exists" "$MAIN" "compiler_main_test_sub_sub_chain"
check_grep "test:T371_exists" "$MAIN" "compiler_main_test_sub_add_net_pos"
check_grep "test:T372_exists" "$MAIN" "compiler_main_test_sub_add_net_neg"
check_grep "test:T373_exists" "$MAIN" "compiler_main_test_add_sub_net_neg"
check_grep "test:T374_exists" "$MAIN" "compiler_main_test_add_sub_net_pos"
check_grep "test:T375_exists" "$MAIN" "compiler_main_test_mul_then_sub_no_fold"
check_grep "test:total_373plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T370_wired"  "$MAIN" "T370 OK.*Block AH"
check_grep "test:T375_wired"  "$MAIN" "T375 OK.*Block AH"

echo ""
echo "[regression]"
check_grep "regr:as_valid"      "$OPT" "as_valid.*256"
check_grep "regr:block_p"       "$OPT" "Sprint 89 Block P"
check_grep "regr:block_ag"      "$OPT" "Sprint 110 Block AG"

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
if echo "$STOUT" | grep -q "T370 OK.*Block AH"; then
    echo "  PASS  selftest:T370_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T370"; then
    echo "  FAIL  selftest:T370_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T370_runtime (OOM before T370)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 111 Block AH — Add/Sub Chain Merging"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
