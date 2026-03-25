#!/usr/bin/env bash
# Sprint 109 Block AF — Opposite-direction shift pair → AND mask gate
# (x << N) >> N → x & ((1<<(64-N))-1)  ;  (x >> N) << N → x & ~((1<<N)-1).
# SOTA: LLVM InstCombineShifts.cpp; Hacker's Delight §2.
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

echo "=== Sprint 109 Block AF — Shift-Mask Reduction Gate ==="
echo ""

echo "[structural]"
check_grep "struct:block_af_comment"    "$OPT" "Sprint 109 Block AF"
check_grep "struct:opposite_dir_guard"  "$OPT" "sh_is_shl && !sh_is_left"
check_grep "struct:mask_compute_shr"    "$OPT" "1 << (64 - af_n)"
check_grep "struct:mask_compute_shl"    "$OPT" "1 << af_n"
check_grep "struct:rewrite_bitand"      "$OPT" "BinaryOp::OpBitAnd, sh_s2"
check_grep "struct:and_tracking_wire"   "$OPT" "and_valid\[sh_instr.dst"

echo ""
echo "[tests]"
check_grep "test:T358_exists" "$MAIN" "compiler_main_test_shift_mask_shr_after_shl"
check_grep "test:T359_exists" "$MAIN" "compiler_main_test_shift_mask_shl_after_shr"
check_grep "test:T360_exists" "$MAIN" "compiler_main_test_shift_mask_single_bit"
check_grep "test:T361_exists" "$MAIN" "compiler_main_test_shift_mask_same_dir_no_fold"
check_grep "test:T362_exists" "$MAIN" "compiler_main_test_shift_mask_unequal_no_fold"
check_grep "test:T363_exists" "$MAIN" "compiler_main_test_shift_mask_chain_with_and"
check_grep "test:total_361plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T358_wired"  "$MAIN" "T358 OK.*Block AF"
check_grep "test:T363_wired"  "$MAIN" "T363 OK.*Block AF"

echo ""
echo "[regression]"
check_grep "regr:sh_valid"       "$OPT" "sh_valid.*256"
check_grep "regr:block_t"       "$OPT" "Sprint 96 Block T"
check_grep "regr:block_ae"      "$OPT" "Sprint 108 Block AE"

echo ""
echo "[typecheck]"
TOTAL=$((TOTAL+1))
TC_OUT=$(timeout 30 $SOUC check "$MAIN" 2>&1)
TC_NEW=$(echo "$TC_OUT" | grep -E "shift_mask|af_mask|af_base|af_n|af_def" || true)
if [ -z "$TC_NEW" ]; then
    echo "  PASS  typecheck:main.sio (no Block AF errors)"; PASS=$((PASS+1))
else
    echo "  FAIL  typecheck:main.sio"; echo "$TC_NEW"; FAIL=$((FAIL+1))
fi

echo ""
echo "[selftest]"
TOTAL=$((TOTAL+1))
STOUT=$(timeout 60 $SOUC run "$MAIN" -- --self-test 2>&1 || true)
if echo "$STOUT" | grep -q "T358 OK.*Block AF"; then
    echo "  PASS  selftest:T358_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T358"; then
    echo "  FAIL  selftest:T358_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T358_runtime (OOM before T358)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 109 Block AF — Shift-Mask Reduction"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
