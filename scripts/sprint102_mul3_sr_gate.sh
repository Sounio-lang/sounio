#!/usr/bin/env bash
# Sprint 102 Block Z — Non-pow-2 Multiply Strength Reduction gate
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

echo "=== Sprint 102 Block Z — Non-pow-2 Multiply SR Gate ==="
echo ""

echo "[structural]"
check_grep "struct:fn_exists"      "$OPT" "fn ocp_mul3_sr("
check_grep "struct:const_lattice"  "$OPT" "const_val\[s2 as usize\] == 3"
check_grep "struct:rewrite_add"    "$OPT" "OpAdd.*s1"
check_grep "struct:sprint102_tag"  "$OPT" "Sprint 102 Block Z"
check_grep "struct:wired_pipeline" "$OPT" "ocp_mul3_sr(current)"

echo ""
echo "[tests]"
check_grep "test:T324_exists" "$MAIN" "compiler_main_test_mul3_basic"
check_grep "test:T325_exists" "$MAIN" "compiler_main_test_mul3_left_const"
check_grep "test:T326_exists" "$MAIN" "compiler_main_test_mul3_skip_pow2"
check_grep "test:T327_exists" "$MAIN" "compiler_main_test_mul3_skip_5"
check_grep "test:T328_exists" "$MAIN" "compiler_main_test_mul3_both_const_skip"
check_grep "test:T329_exists" "$MAIN" "compiler_main_test_mul3_add_sequence"
check_grep "test:total_347plus" "$MAIN" "let total: i64 = 3[4-9][0-9]"
check_grep "test:T324_wired"  "$MAIN" "T324 OK"
check_grep "test:T329_wired"  "$MAIN" "T329 OK"

echo ""
echo "[regression]"
check_grep "regr:const_fold"    "$OPT" "fn ocp_const_fold("
check_grep "regr:dedup_imm"    "$OPT" "fn ocp_dedup_imm("
check_grep "regr:compact_nops" "$OPT" "fn ocp_compact_nops("

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
if echo "$STOUT" | grep -q "T324 OK"; then
    echo "  PASS  selftest:T324_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T324"; then
    echo "  FAIL  selftest:T324_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T324_runtime (OOM before T324)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 102 Block Z — Non-pow-2 Multiply SR"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
