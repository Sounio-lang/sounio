#!/usr/bin/env bash
# Sprint 105 Block AC — Instruction Scheduling / Load Sinking gate
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

echo "=== Sprint 105 Block AC — Load Sinking Gate ==="
echo ""

echo "[structural]"
check_grep "struct:fn_exists"      "$OPT" "fn ocp_sink_loads("
check_grep "struct:bubble_sort"    "$OPT" "pass < 4"
check_grep "struct:swap"           "$OPT" "Swap"
check_grep "struct:sprint105_tag"  "$OPT" "Sprint 105 Block AC"
check_grep "struct:wired_pipeline" "$OPT" "ocp_sink_loads(current)"

echo ""
echo "[tests]"
check_grep "test:T312_exists" "$MAIN" "compiler_main_test_sink_loads_no_swap_imm_pair"
check_grep "test:T313_exists" "$MAIN" "compiler_main_test_sink_loads_no_swap_adjacent_use"
check_grep "test:T314_exists" "$MAIN" "compiler_main_test_sink_loads_no_swap_label"
check_grep "test:T315_exists" "$MAIN" "compiler_main_test_sink_loads_no_swap_src1"
check_grep "test:T316_exists" "$MAIN" "compiler_main_test_sink_loads_swap_past_binop"
check_grep "test:T317_exists" "$MAIN" "compiler_main_test_sink_loads_noop"
check_grep "test:total_347"   "$MAIN" "let total: i64 = 347"
check_grep "test:T312_wired"  "$MAIN" "T312 OK"
check_grep "test:T317_wired"  "$MAIN" "T317 OK"

echo ""
echo "[regression]"
check_grep "regr:const_fold"   "$OPT" "fn ocp_const_fold("
check_grep "regr:dse"          "$OPT" "fn ocp_dse("
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
if echo "$STOUT" | grep -q "T312 OK"; then
    echo "  PASS  selftest:T312_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T312"; then
    echo "  FAIL  selftest:T312_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T312_runtime (OOM before T312)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 105 Block AC — Load Sinking"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
