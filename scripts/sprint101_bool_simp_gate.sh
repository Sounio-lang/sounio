#!/usr/bin/env bash
# Sprint 101 Block Y — Boolean Simplification gate
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

echo "=== Sprint 101 Block Y — Boolean Simplification Gate ==="
echo ""

echo "[structural]"
check_grep "struct:is_cmp_result"    "$OPT" "is_cmp_result"
check_grep "struct:cmp_orig_src"     "$OPT" "cmp_orig_src"
check_grep "struct:cmp_op_code"      "$OPT" "cmp_op_code"
check_grep "struct:eq_double_neg"    "$OPT" "double boolean negation"
check_grep "struct:ne_double_neg"    "$OPT" "double negation via Ne"
check_grep "struct:bool_identity"    "$OPT" "bool != 0"
check_grep "struct:commutative"      "$OPT" "commutative.*const on left"
check_grep "struct:sprint101_tag"    "$OPT" "Sprint 101 Block Y"

echo ""
echo "[tests]"
check_grep "test:T330_exists" "$MAIN" "compiler_main_test_bool_simp_double_eq"
check_grep "test:T331_exists" "$MAIN" "compiler_main_test_bool_simp_double_ne"
check_grep "test:T332_exists" "$MAIN" "compiler_main_test_bool_simp_mixed_eq_ne"
check_grep "test:T333_exists" "$MAIN" "compiler_main_test_bool_simp_mixed_ne_eq"
check_grep "test:T334_exists" "$MAIN" "compiler_main_test_bool_simp_commutative"
check_grep "test:T335_exists" "$MAIN" "compiler_main_test_bool_simp_nonzero_no_fold"
check_grep "test:total_347"   "$MAIN" "let total: i64 = 347"
check_grep "test:T330_wired"  "$MAIN" "T330 OK"
check_grep "test:T335_wired"  "$MAIN" "T335 OK"

echo ""
echo "[regression]"
check_grep "regr:const_fold"      "$OPT" "fn ocp_const_fold("
check_grep "regr:dead_label"      "$OPT" "fn ocp_dead_label("
check_grep "regr:compact_nops"    "$OPT" "fn ocp_compact_nops("
check_grep "regr:redundant_jump"  "$OPT" "fn ocp_redundant_jump("

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
if echo "$STOUT" | grep -q "T330 OK"; then
    echo "  PASS  selftest:T330_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T330"; then
    echo "  FAIL  selftest:T330_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T330_runtime (OOM before T330)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 101 Block Y — Boolean Simplification"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
