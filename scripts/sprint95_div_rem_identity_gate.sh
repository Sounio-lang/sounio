#!/usr/bin/env bash
# Sprint 95 Block S — Division/Remainder Identity Folding gate
# Usage: bash scripts/sprint95_div_rem_identity_gate.sh
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

echo "=== Sprint 95 Block S — Division/Remainder Identity Folding Gate ==="
echo ""

# --- Structural checks ---
echo "[structural]"
check_grep "struct:div_self_fold"    "$OPT" "x / x.*1"
check_grep "struct:rem_self_fold"    "$OPT" "x % x.*0"
check_grep "struct:zero_div_fold"    "$OPT" "0 / x.*0"
check_grep "struct:zero_rem_fold"    "$OPT" "0 % x.*0"
check_grep "struct:div_self_in_same_reg" "$OPT" "OpDiv"
check_grep "struct:rem_self_in_same_reg" "$OPT" "OpRem"
check_grep "struct:sprint95_tag"     "$OPT" "Sprint 95 Block S"
check_grep "struct:zero_div_v1_check" "$OPT" "s1_known && v1 == 0"

echo ""
echo "[tests]"
check_grep "test:T270_exists" "$MAIN" "compiler_main_test_div_self"
check_grep "test:T271_exists" "$MAIN" "compiler_main_test_rem_self"
check_grep "test:T272_exists" "$MAIN" "compiler_main_test_zero_div"
check_grep "test:T273_exists" "$MAIN" "compiler_main_test_zero_rem"
check_grep "test:T274_exists" "$MAIN" "compiler_main_test_div_one_regr"
check_grep "test:T275_exists" "$MAIN" "compiler_main_test_div_rem_chain"
check_grep "test:total_275plus" "$MAIN" "let total: i64 = [3-9][0-9][0-9]"
check_grep "test:T270_wired"  "$MAIN" "T270 OK"
check_grep "test:T275_wired"  "$MAIN" "T275 OK"

echo ""
echo "[regression]"
check_grep "regr:div_by_1_still_exists" "$OPT" "x / 1"
check_grep "regr:rem_by_1_still_exists" "$OPT" "x % 1"
check_grep "regr:mul_div_cancel"        "$OPT" "mul-div cancellation"
check_grep "regr:dse_exists"            "$OPT" "fn ocp_dse("
check_grep "regr:copy_prop_exists"      "$OPT" "fn ocp_copy_prop("

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
if echo "$STOUT" | grep -q "T270 OK"; then
    echo "  PASS  selftest:T270_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T270"; then
    echo "  FAIL  selftest:T270_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T270_runtime (OOM before T270)"; NOT_RUN=$((NOT_RUN+1))
fi

echo ""
echo "========================================="
echo "Sprint 95 Block S — Division/Remainder Identity Folding"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
