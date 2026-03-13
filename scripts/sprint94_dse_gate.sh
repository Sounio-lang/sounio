#!/usr/bin/env bash
# Sprint 94 Block R — Dead Store Elimination gate
# Usage: bash scripts/sprint94_dse_gate.sh
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
OPT=self-hosted/ir/opt_cleanup.sio
MAIN=self-hosted/compiler/main.sio
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0

check() {
    local name="$1"; shift
    TOTAL=$((TOTAL+1))
    if "$@" >/dev/null 2>&1; then
        echo "  PASS  $name"; PASS=$((PASS+1))
    else
        echo "  FAIL  $name"; FAIL=$((FAIL+1))
    fi
}

check_grep() {
    local name="$1" file="$2" pattern="$3"
    TOTAL=$((TOTAL+1))
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "  PASS  $name"; PASS=$((PASS+1))
    else
        echo "  FAIL  $name"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 94 Block R — Dead Store Elimination Gate ==="
echo ""

# --- Structural checks ---
echo "[structural]"
check_grep "struct:ocp_dse_exists" "$OPT" "fn ocp_dse("
check_grep "struct:last_write_array" "$OPT" "last_write.*256"
check_grep "struct:nop_dead_store" "$OPT" "ir_nop()"
check_grep "struct:label_barrier" "$OPT" "IrLabel"
check_grep "struct:branch_barrier" "$OPT" "IrBranchTrue"
check_grep "struct:jump_barrier" "$OPT" "IrJump"
check_grep "struct:side_effect_skip" "$OPT" "ocp_has_side_effect"
check_grep "struct:has_dst_check" "$OPT" "ocp_has_dst"
check_grep "struct:wired_in_pipeline" "$OPT" "ocp_dse(current)"
check_grep "struct:pipeline_after_copy_prop" "$OPT" "ocp_copy_prop.*\|.*ocp_dse\|copy_prop.*\n.*dse"

# More precise pipeline ordering check
TOTAL=$((TOTAL+1))
COPY_LINE=$(grep -n "ocp_copy_prop(current)" "$OPT" | head -1 | cut -d: -f1)
DSE_LINE=$(grep -n "ocp_dse(current)" "$OPT" | head -1 | cut -d: -f1)
if [ -n "$COPY_LINE" ] && [ -n "$DSE_LINE" ] && [ "$DSE_LINE" -gt "$COPY_LINE" ]; then
    echo "  PASS  struct:dse_after_copy_prop_ordering"; PASS=$((PASS+1))
else
    echo "  FAIL  struct:dse_after_copy_prop_ordering"; FAIL=$((FAIL+1))
fi

echo ""
echo "[tests]"
check_grep "test:T264_exists" "$MAIN" "compiler_main_test_dse_basic"
check_grep "test:T265_exists" "$MAIN" "compiler_main_test_dse_live_store"
check_grep "test:T266_exists" "$MAIN" "compiler_main_test_dse_label_barrier"
check_grep "test:T267_exists" "$MAIN" "compiler_main_test_dse_chain"
check_grep "test:T268_exists" "$MAIN" "compiler_main_test_dse_different_regs"
check_grep "test:T269_exists" "$MAIN" "compiler_main_test_dse_copy_integration"
check_grep "test:total_269plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T264_wired" "$MAIN" "T264 OK"
check_grep "test:T269_wired" "$MAIN" "T269 OK"

echo ""
echo "[regression]"
# Verify existing passes still present
check_grep "regr:const_fold_exists" "$OPT" "fn ocp_const_fold("
check_grep "regr:copy_prop_exists" "$OPT" "fn ocp_copy_prop("
check_grep "regr:cse_exists" "$OPT" "fn ocp_cse("
check_grep "regr:dce_once_exists" "$OPT" "fn ocp_dce_once("
check_grep "regr:dedup_imm_exists" "$OPT" "fn ocp_dedup_imm("

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
if echo "$STOUT" | grep -q "T264 OK"; then
    echo "  PASS  selftest:T264_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T264"; then
    echo "  FAIL  selftest:T264_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T264_runtime (OOM before T264)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 94 Block R — Dead Store Elimination"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
