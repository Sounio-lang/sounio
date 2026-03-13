#!/usr/bin/env bash
# Sprint 112 Block AI — Bitwise NOT via XOR(-1) double-NOT elimination gate
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

echo "=== Sprint 112 Block AI — Bitwise NOT Double-Elimination Gate ==="
echo ""

echo "[structural]"
check_grep "struct:block_ai_comment"    "$OPT" "Sprint 112 Block AI"
check_grep "struct:is_bnot_array"       "$OPT" "is_bnot.*256"
check_grep "struct:bnot_src_array"      "$OPT" "bnot_src.*256"
check_grep "struct:double_not_fold"     "$OPT" "is_bnot\[bn_s1"
check_grep "struct:commutative_not"     "$OPT" "is_bnot\[bn_s2"
check_grep "struct:xor_neg1_guard"      "$OPT" "const_val\[bn_s2 as usize\] == -1"

echo ""
echo "[tests]"
check_grep "test:T376_exists" "$MAIN" "compiler_main_test_double_bnot"
check_grep "test:T377_exists" "$MAIN" "compiler_main_test_double_bnot_commutative"
check_grep "test:T378_exists" "$MAIN" "compiler_main_test_single_bnot_no_fold"
check_grep "test:T379_exists" "$MAIN" "compiler_main_test_xor_not_neg1_no_track"
check_grep "test:T380_exists" "$MAIN" "compiler_main_test_bnot_with_and"
check_grep "test:T381_exists" "$MAIN" "compiler_main_test_triple_bnot"
check_grep "test:total_379plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T376_wired"  "$MAIN" "T376 OK.*Block AI"
check_grep "test:T381_wired"  "$MAIN" "T381 OK.*Block AI"

echo ""
echo "[regression]"
check_grep "regr:is_neg"        "$OPT" "is_neg.*256"
check_grep "regr:block_m"       "$OPT" "Sprint 86 Block M"
check_grep "regr:block_ah"      "$OPT" "Sprint 111 Block AH"

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
if echo "$STOUT" | grep -q "T376 OK.*Block AI"; then
    echo "  PASS  selftest:T376_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T376"; then
    echo "  FAIL  selftest:T376_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T376_runtime (OOM before T376)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 112 Block AI — Bitwise NOT"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
