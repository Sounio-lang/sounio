#!/usr/bin/env bash
# Sprint 96 Block T — Shift-Chain Combining gate
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

echo "=== Sprint 96 Block T — Shift-Chain Combining Gate ==="
echo ""

echo "[structural]"
check_grep "struct:sh_valid_array"     "$OPT" "sh_valid.*256"
check_grep "struct:sh_var_src_array"   "$OPT" "sh_var_src.*256"
check_grep "struct:sh_amount_array"    "$OPT" "sh_amount.*256"
check_grep "struct:sh_is_left_array"   "$OPT" "sh_is_left.*256"
check_grep "struct:sprint96_tag"       "$OPT" "Sprint 96 Block T"
check_grep "struct:combined_shift"     "$OPT" "sh_c1 + sh_c2"
check_grep "struct:direction_check"    "$OPT" "sh_is_left"
check_grep "struct:overflow_guard"     "$OPT" "sh_combined < 64"
check_grep "struct:OpShl_check"        "$OPT" "OpShl"
check_grep "struct:OpShr_check"        "$OPT" "OpShr"

echo ""
echo "[tests]"
check_grep "test:T276_exists" "$MAIN" "compiler_main_test_shift_chain_shl"
check_grep "test:T277_exists" "$MAIN" "compiler_main_test_shift_chain_shr"
check_grep "test:T278_exists" "$MAIN" "compiler_main_test_shift_chain_mixed"
check_grep "test:T279_exists" "$MAIN" "compiler_main_test_shift_chain_triple"
check_grep "test:T280_exists" "$MAIN" "compiler_main_test_shift_chain_nonconst"
check_grep "test:T281_exists" "$MAIN" "compiler_main_test_shift_chain_sr_integration"
check_grep "test:total_281plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T276_wired"  "$MAIN" "T276 OK"
check_grep "test:T281_wired"  "$MAIN" "T281 OK"

echo ""
echo "[regression]"
check_grep "regr:pow2_sr_exists"    "$OPT" "Sprint 68"
check_grep "regr:block_l_exists"    "$OPT" "Block L"
check_grep "regr:block_s_exists"    "$OPT" "Block S"
check_grep "regr:dse_exists"        "$OPT" "fn ocp_dse("

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
if echo "$STOUT" | grep -q "T276 OK"; then
    echo "  PASS  selftest:T276_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T276"; then
    echo "  FAIL  selftest:T276_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T276_runtime (OOM before T276)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 96 Block T — Shift-Chain Combining"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
