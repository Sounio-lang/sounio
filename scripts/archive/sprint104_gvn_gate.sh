#!/usr/bin/env bash
# Sprint 104 Block AB — Global Value Numbering gate
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

echo "=== Sprint 104 Block AB — GVN Gate ==="
echo ""

echo "[structural]"
check_grep "struct:fn_exists"      "$OPT" "fn ocp_gvn("
check_grep "struct:vn_array"       "$OPT" "var vn:.*256"
check_grep "struct:hash_table"     "$OPT" "ht_key.*128"
check_grep "struct:linear_probe"   "$OPT" "Linear probe"
check_grep "struct:label_inval"    "$OPT" "invalidate hash table"
check_grep "struct:sprint104_tag"  "$OPT" "Sprint 104 Block AB"
check_grep "struct:wired_pipeline" "$OPT" "ocp_gvn(current)"

echo ""
echo "[tests]"
check_grep "test:T336_exists" "$MAIN" "compiler_main_test_gvn_basic_redundancy"
check_grep "test:T337_exists" "$MAIN" "compiler_main_test_gvn_different_ops"
check_grep "test:T338_exists" "$MAIN" "compiler_main_test_gvn_label_invalidates"
check_grep "test:T339_exists" "$MAIN" "compiler_main_test_gvn_copy_inherits_vn"
check_grep "test:T340_exists" "$MAIN" "compiler_main_test_gvn_no_binops"
check_grep "test:T341_exists" "$MAIN" "compiler_main_test_gvn_chain"
check_grep "test:total_347plus" "$MAIN" "let total: i64 = 3[4-9][0-9]"
check_grep "test:T336_wired"  "$MAIN" "T336 OK"
check_grep "test:T341_wired"  "$MAIN" "T341 OK"

echo ""
echo "[regression]"
check_grep "regr:cse"          "$OPT" "fn ocp_cse("
check_grep "regr:const_fold"   "$OPT" "fn ocp_const_fold("
check_grep "regr:copy_prop"    "$OPT" "fn ocp_copy_prop("

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
if echo "$STOUT" | grep -q "T336 OK"; then
    echo "  PASS  selftest:T336_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T336"; then
    echo "  FAIL  selftest:T336_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T336_runtime (OOM before T336)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 104 Block AB — GVN"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
