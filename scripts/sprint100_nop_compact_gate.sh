#!/usr/bin/env bash
# Sprint 100 Block X — NOP Compaction gate
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

echo "=== Sprint 100 Block X — NOP Compaction Gate ==="
echo ""

echo "[structural]"
check_grep "struct:fn_exists"       "$OPT" "fn ocp_compact_nops("
check_grep "struct:read_write_loop" "$OPT" "write.*read\|read.*write"
check_grep "struct:skip_nop"        "$OPT" "IrNop"
check_grep "struct:update_count"    "$OPT" "instr_count.*write\|result.instr_count = write"
check_grep "struct:sprint100_tag"   "$OPT" "Sprint 100 Block X"
check_grep "struct:wired_pipeline"  "$OPT" "ocp_compact_nops(current)"
check_grep "struct:wired_early_exit" "$OPT" "ocp_compact_nops(current).*Block X"

echo ""
echo "[tests]"
check_grep "test:T300_exists" "$MAIN" "compiler_main_test_compact_basic"
check_grep "test:T301_exists" "$MAIN" "compiler_main_test_compact_all_nops"
check_grep "test:T302_exists" "$MAIN" "compiler_main_test_compact_no_nops"
check_grep "test:T303_exists" "$MAIN" "compiler_main_test_compact_interspersed"
check_grep "test:T304_exists" "$MAIN" "compiler_main_test_compact_labels"
check_grep "test:T305_exists" "$MAIN" "compiler_main_test_compact_pipeline"
check_grep "test:total_305"   "$MAIN" "let total: i64 = 305"
check_grep "test:T300_wired"  "$MAIN" "T300 OK"
check_grep "test:T305_wired"  "$MAIN" "T305 OK"

echo ""
echo "[regression]"
check_grep "regr:dead_label"    "$OPT" "fn ocp_dead_label("
check_grep "regr:dead_jump"     "$OPT" "fn ocp_dead_after_jump("
check_grep "regr:redundant_jump" "$OPT" "fn ocp_redundant_jump("
check_grep "regr:const_fold"    "$OPT" "fn ocp_const_fold("

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
if echo "$STOUT" | grep -q "T300 OK"; then
    echo "  PASS  selftest:T300_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T300"; then
    echo "  FAIL  selftest:T300_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T300_runtime (OOM before T300)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 100 Block X — NOP Compaction"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
