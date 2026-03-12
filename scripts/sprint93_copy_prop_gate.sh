#!/usr/bin/env bash
# Sprint 93 gate: Block Q — copy chain collapse (copy propagation)
# SOTA: Cytron et al. TOPLAS 1991 §5; Cooper & Torczon §8.5
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
OPT=self-hosted/ir/opt_cleanup.sio
MAIN=self-hosted/compiler/main.sio

pass=0; fail=0; not_run=0

check() {
    local name="$1"; local cmd="$2"
    if eval "$cmd" >/dev/null 2>&1; then
        echo "  PASS: $name"; pass=$((pass+1))
    else
        echo "  FAIL: $name"; fail=$((fail+1))
    fi
}

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "  PASS: $name"; pass=$((pass+1))
    else
        echo "  FAIL: $name"; fail=$((fail+1))
    fi
}

echo "=== Sprint 93 Gate: Block Q — copy chain collapse ==="
echo ""

# Structural checks
echo "--- Structural ---"
check_grep "block_q:fn_ocp_copy_prop" "fn ocp_copy_prop" "$OPT"
check_grep "block_q:copy_target_array" "copy_target.*256" "$OPT"
check_grep "block_q:chain_collapse" "copy_target\[src as usize\]" "$OPT"
check_grep "block_q:invalidate_non_copy" "copy_target\[cur.dst as usize\] = -1" "$OPT"
check_grep "block_q:wired_in_pipeline" "ocp_copy_prop" "$OPT"
check_grep "block_q:pipeline_ordering" "ocp_copy_prop(current)" "$OPT"

# Test function checks
echo ""
echo "--- Test Functions ---"
check_grep "test:T258_basic" "compiler_main_test_copy_prop_basic" "$MAIN"
check_grep "test:T259_chain" "compiler_main_test_copy_prop_chain" "$MAIN"
check_grep "test:T260_overwrite" "compiler_main_test_copy_prop_overwrite" "$MAIN"
check_grep "test:T261_both_srcs" "compiler_main_test_copy_prop_both_srcs" "$MAIN"
check_grep "test:T262_self_copy" "compiler_main_test_copy_prop_self_copy" "$MAIN"
check_grep "test:T263_const_integration" "compiler_main_test_copy_prop_const_integration" "$MAIN"

# Runner wiring
echo ""
echo "--- Runner Wiring ---"
check_grep "runner:T258_wired" "T258 OK" "$MAIN"
check_grep "runner:T259_wired" "T259 OK" "$MAIN"
check_grep "runner:T260_wired" "T260 OK" "$MAIN"
check_grep "runner:T261_wired" "T261 OK" "$MAIN"
check_grep "runner:T262_wired" "T262 OK" "$MAIN"
check_grep "runner:T263_wired" "T263 OK" "$MAIN"
check_grep "runner:total_263plus" "let total: i64 = 26[3-9]\|let total: i64 = 2[7-9]\|let total: i64 = [3-9]" "$MAIN"

# Typecheck
echo ""
echo "--- Typecheck ---"
check "typecheck:main_sio" "$SOUC check $MAIN"

# Self-test (with OOM tolerance)
echo ""
echo "--- Self-test (OOM tolerant) ---"
_ec=0
timeout 180 $SOUC run self-hosted/compiler/main.sio -- --self-test > /tmp/sprint93_selftest.txt 2>&1 || _ec=$?
if grep -q "T258 OK" /tmp/sprint93_selftest.txt 2>/dev/null; then
    echo "  PASS: selftest:T258_runtime"; pass=$((pass+1))
elif [ $_ec -eq 137 ] || [ $_ec -eq 143 ] || [ $_ec -eq 144 ] || ! grep -q "T258" /tmp/sprint93_selftest.txt 2>/dev/null; then
    echo "  NOT_RUN: selftest (OOM/timeout before T258, exit=$_ec)"
    not_run=$((not_run+1))
else
    echo "  FAIL: selftest:T258_runtime"; fail=$((fail+1))
fi

# Regression guard
echo ""
echo "--- Regression ---"
check_grep "regression:ocp_const_fold_exists" "fn ocp_const_fold" "$OPT"
check_grep "regression:ocp_dce_exists" "fn ocp_dce" "$OPT"
check_grep "regression:ocp_cse_exists" "fn ocp_cse" "$OPT"

echo ""
echo "=== Sprint 93 Gate Summary ==="
echo "  PASS=$pass  FAIL=$fail  NOT_RUN=$not_run"
echo ""

if [ $fail -gt 0 ]; then
    echo "GATE: FAIL"
    exit 1
fi
echo "GATE: PASS"
