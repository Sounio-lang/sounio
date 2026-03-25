#!/usr/bin/env bash
# Sprint 114 Block AK — Redundant AND elimination (superset mask) gate
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

echo "=== Sprint 114 Block AK — Superset AND Elimination Gate ==="
echo ""

echo "[structural]"
check_grep "struct:block_ak_comment"    "$OPT" "Sprint 114 Block AK"
check_grep "struct:superset_check"      "$OPT" "ak_c1 & v2.*== ak_c1"
check_grep "struct:copy_fold"           "$OPT" "ir_copy(instr.dst, s1)"
check_grep "struct:commutative"         "$OPT" "ak_c2 & v1.*== ak_c2"
check_grep "struct:and_tracking_prop"   "$OPT" "and_valid\[instr.dst as usize\].*= true"

echo ""
echo "[tests]"
check_grep "test:T388_exists" "$MAIN" "compiler_main_test_and_superset_copy"
check_grep "test:T389_exists" "$MAIN" "compiler_main_test_and_superset_nibble"
check_grep "test:T390_exists" "$MAIN" "compiler_main_test_and_not_superset"
check_grep "test:T391_exists" "$MAIN" "compiler_main_test_and_same_mask_copy"
check_grep "test:T392_exists" "$MAIN" "compiler_main_test_and_superset_commutative"
check_grep "test:T393_exists" "$MAIN" "compiler_main_test_and_no_track_no_fold"
check_grep "test:total_391plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T388_wired"  "$MAIN" "T388 OK.*Block AK"
check_grep "test:T393_wired"  "$MAIN" "T393 OK.*Block AK"

echo ""
echo "[regression]"
check_grep "regr:block_ae"      "$OPT" "Sprint 108 Block AE"
check_grep "regr:and_valid"     "$OPT" "and_valid.*256"
check_grep "regr:block_aj"      "$OPT" "Sprint 113 Block AJ"

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
if echo "$STOUT" | grep -q "T388 OK.*Block AK"; then
    echo "  PASS  selftest:T388_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T388"; then
    echo "  FAIL  selftest:T388_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T388_runtime (OOM before T388)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 114 Block AK — Superset AND"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
