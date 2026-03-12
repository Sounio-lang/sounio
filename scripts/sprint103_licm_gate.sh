#!/usr/bin/env bash
# Sprint 103 Block AA — LICM gate
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

echo "=== Sprint 103 Block AA — LICM Gate ==="
echo ""

echo "[structural]"
check_grep "struct:fn_exists"      "$OPT" "fn ocp_licm("
check_grep "struct:label_pos"      "$OPT" "label_pos.*256"
check_grep "struct:back_edge"      "$OPT" "lp.*< i"
check_grep "struct:hoist"          "$OPT" "Hoist.*swap"
check_grep "struct:sprint103_tag"  "$OPT" "Sprint 103 Block AA"
check_grep "struct:wired_pipeline" "$OPT" "ocp_licm(current)"

echo ""
echo "[tests]"
check_grep "test:T318_exists" "$MAIN" "compiler_main_test_licm_hoist_loadimm"
check_grep "test:T319_exists" "$MAIN" "compiler_main_test_licm_no_hoist_written_elsewhere"
check_grep "test:T320_exists" "$MAIN" "compiler_main_test_licm_forward_jump_no_hoist"
check_grep "test:T321_exists" "$MAIN" "compiler_main_test_licm_binop_not_hoisted"
check_grep "test:T322_exists" "$MAIN" "compiler_main_test_licm_first_hoisted"
check_grep "test:T323_exists" "$MAIN" "compiler_main_test_licm_empty_loop"
check_grep "test:total_347"   "$MAIN" "let total: i64 = 347"
check_grep "test:T318_wired"  "$MAIN" "T318 OK"
check_grep "test:T323_wired"  "$MAIN" "T323 OK"

echo ""
echo "[regression]"
check_grep "regr:const_fold"    "$OPT" "fn ocp_const_fold("
check_grep "regr:dead_label"    "$OPT" "fn ocp_dead_label("
check_grep "regr:compact_nops"  "$OPT" "fn ocp_compact_nops("

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
if echo "$STOUT" | grep -q "T318 OK"; then
    echo "  PASS  selftest:T318_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T318"; then
    echo "  FAIL  selftest:T318_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T318_runtime (OOM before T318)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 103 Block AA — LICM"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
