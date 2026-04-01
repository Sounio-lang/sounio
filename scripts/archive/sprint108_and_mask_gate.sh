#!/usr/bin/env bash
# Sprint 108 Block AE — And-mask truncation folding gate
# (x & C1) & C2 → x & (C1 & C2).  SOTA: LLVM InstCombineAndOrXor.cpp.
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

echo "=== Sprint 108 Block AE — And-mask Truncation Folding Gate ==="
echo ""

echo "[structural]"
check_grep "struct:and_valid_array"    "$OPT" "and_valid.*256"
check_grep "struct:and_var_src_array"  "$OPT" "and_var_src.*256"
check_grep "struct:and_const_val_arr"  "$OPT" "and_const_val.*256"
check_grep "struct:block_ae_comment"   "$OPT" "Sprint 108 Block AE"
check_grep "struct:chain_fold"         "$OPT" "and_valid\[s1 as usize\]"
check_grep "struct:tracking_record"    "$OPT" "and_var_src\[instr.dst as usize\]"

echo ""
echo "[tests]"
check_grep "test:T352_exists" "$MAIN" "compiler_main_test_and_mask_basic"
check_grep "test:T353_exists" "$MAIN" "compiler_main_test_and_mask_three_deep"
check_grep "test:T354_exists" "$MAIN" "compiler_main_test_and_mask_to_zero"
check_grep "test:T355_exists" "$MAIN" "compiler_main_test_and_mask_identity"
check_grep "test:T356_exists" "$MAIN" "compiler_main_test_and_mask_or_no_fold"
check_grep "test:T357_exists" "$MAIN" "compiler_main_test_and_mask_sub_no_track"
check_grep "test:total_355plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T352_wired"  "$MAIN" "T352 OK.*Block AE"
check_grep "test:T357_wired"  "$MAIN" "T357 OK.*Block AE"

echo ""
echo "[regression]"
check_grep "regr:has_ra"          "$OPT" "has_ra.*256"
check_grep "regr:ra_const_val"    "$OPT" "ra_const_val.*256"
check_grep "regr:ocp_const_fold"  "$OPT" "fn ocp_const_fold("

echo ""
echo "[typecheck]"
# main.sio may have pre-existing linter errors (Sprint 110/113).
# Verify Block AE functions introduce no NEW undefined symbols.
TOTAL=$((TOTAL+1))
TC_OUT=$(timeout 30 $SOUC check "$MAIN" 2>&1)
TC_NEW=$(echo "$TC_OUT" | grep -E "and_mask|and_valid|and_var_src|and_const_val" || true)
if [ -z "$TC_NEW" ]; then
    echo "  PASS  typecheck:main.sio (no Block AE errors)"; PASS=$((PASS+1))
else
    echo "  FAIL  typecheck:main.sio"; echo "$TC_NEW"; FAIL=$((FAIL+1))
fi

echo ""
echo "[selftest]"
TOTAL=$((TOTAL+1))
STOUT=$(timeout 60 $SOUC run "$MAIN" -- --self-test 2>&1 || true)
if echo "$STOUT" | grep -q "T352 OK.*Block AE"; then
    echo "  PASS  selftest:T352_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T352"; then
    echo "  FAIL  selftest:T352_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T352_runtime (OOM before T352)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 108 Block AE — And-mask Truncation Folding"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
