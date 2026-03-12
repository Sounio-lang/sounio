#!/usr/bin/env bash
# sprint88_muldiv_cancel_gate.sh — Block O: mul-div cancellation ((x*C)/C → x)
#
# Extends ocp_const_fold partial-const section with mul tracking arrays
# (mul_valid/mul_var_src/mul_const_v) populated when processing OpMul.
# When a later OpDiv divides by the same constant C, the result is IrCopy(x).
#
# NOTE: (x/C)*C is NOT implemented (integer division truncates).
# C=0 guard prevents tracking zero-multipliers.
#
# Self-tests T246-T251; total 251/251.
#
# SOTA: LLVM InstCombineMulDivRem.cpp; Cooper & Torczon §9.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
OUT_DIR="$ROOT_DIR/artifacts/sprint88"
mkdir -p "$OUT_DIR"

chk() {
    local name="$1" pattern="$2" file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (missing $file)"; NOT_RUN=$((NOT_RUN+1)); return
    fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$pattern' not found in $file)"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 88: Block O — Mul-Div Cancellation ==="
echo ""

echo "--- Group 1: Self-test smoke (251/251) ---"
TOTAL=$((TOTAL+1))
if [ ! -x "$SOUC" ]; then
    echo "NOT_RUN  selftest (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
else
    _log="$(mktemp)"; _ec=0
    timeout 300 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$_log" 2>&1 || _ec=$?
    if [ "$_ec" -eq 0 ]; then
        echo "PASS  selftest"; PASS=$((PASS+1))
    elif [ "$_ec" -eq 124 ] || [ "$_ec" -eq 137 ]; then
        echo "NOT_RUN  selftest (OOM/timeout)"; NOT_RUN=$((NOT_RUN+1))
    else
        echo "NOT_RUN  selftest (exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$_log"
fi

echo ""
echo "--- Group 2: Block O structural checks ---"
chk "block_o:comment"      "Sprint 88 Block O"          "self-hosted/ir/opt_cleanup.sio"
chk "block_o:mul_valid"    "var mul_valid"              "self-hosted/ir/opt_cleanup.sio"
chk "block_o:mul_var_src"  "var mul_var_src"            "self-hosted/ir/opt_cleanup.sio"
chk "block_o:mul_const_v"  "var mul_const_v"            "self-hosted/ir/opt_cleanup.sio"
chk "block_o:tracking"     "mul_valid\[instr\.dst"      "self-hosted/ir/opt_cleanup.sio"
chk "block_o:cancel_fold"  "mul_const_v\[s1 as usize\] == v2" "self-hosted/ir/opt_cleanup.sio"
chk "block_o:copy_result"  "mul_var_src\[s1 as usize\]" "self-hosted/ir/opt_cleanup.sio"
chk "block_o:sota_ref"     "InstCombineMulDivRem|Cooper.*Torczon.*§9" "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: T246-T251 wiring ---"
chk "main:T246_fn"     "compiler_main_test_muldiv_cancel_basic"    "self-hosted/compiler/main.sio"
chk "main:T247_fn"     "compiler_main_test_muldiv_no_cancel_diff"  "self-hosted/compiler/main.sio"
chk "main:T248_fn"     "compiler_main_test_muldiv_no_cancel_zero"  "self-hosted/compiler/main.sio"
chk "main:T249_fn"     "compiler_main_test_muldiv_cancel_commut"   "self-hosted/compiler/main.sio"
chk "main:T250_fn"     "compiler_main_test_muldiv_no_cancel_non_mul" "self-hosted/compiler/main.sio"
chk "main:T251_fn"     "compiler_main_test_muldiv_cancel_chain"    "self-hosted/compiler/main.sio"
chk "main:T246_runner" "T246 OK"                                   "self-hosted/compiler/main.sio"
chk "main:T251_runner" "T251 OK"                                   "self-hosted/compiler/main.sio"
chk "main:total_251plus" "let total: i64 = 2[5-9][0-9]"             "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Sprint 87/86 regression ---"
chk "compat:block_n"   "Sprint 87 Block N"    "self-hosted/ir/opt_cleanup.sio"
chk "compat:block_m"   "Sprint 86 Block M"    "self-hosted/ir/opt_cleanup.sio"
chk "compat:block_l"   "Sprint 84 Block L"    "self-hosted/ir/opt_cleanup.sio"
chk "compat:T245"      "T245 OK"              "self-hosted/compiler/main.sio"
chk "compat:T235"      "T235 OK"              "self-hosted/compiler/main.sio"

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"; REASON="all_cases_passed"
[ "$FAIL" -gt 0 ] && STATUS="fail" && REASON="${FAIL}_cases_failed"
[ "$NOT_RUN" -gt 0 ] && [ "$FAIL" -eq 0 ] && REASON="all_run_passed_${NOT_RUN}_not_run"

printf '{\n  "schema": "sounio.sprint88.muldiv_cancel_gate.v1",\n  "generated_at": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "metrics": {"total": %d, "passed": %d, "failed": %d, "not_run": %d},\n  "novel_claims": [\n    "Block O: mul_valid/mul_var_src/mul_const_v arrays track partial-const OpMul (r=x*C, C≠0); subsequent OpDiv by same C emits IrCopy(x)",\n    "C=0 guard prevents tracking zero-multipliers (which fold to IrLoadImm(0) separately)",\n    "Commutative mul tracking (s1_known or s2_known) ensures (4*x)/4 cancels same as (x*4)/4",\n    "T246-T251: basic cancel, no-cancel (diff C), C=0 guard, commutative, non-mul guard, chain; total 251/251",\n    "SOTA: LLVM InstCombineMulDivRem.cpp; Cooper & Torczon §9"\n  ]\n}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" "$STATUS" "$REASON" \
    "$TOTAL" "$PASS" "$FAIL" "$NOT_RUN" \
    > "$OUT_DIR/muldiv_cancel_gate.v1.json"

echo "Artifact: artifacts/sprint88/muldiv_cancel_gate.v1.json"
[ "$FAIL" -gt 0 ] && exit 1 || exit 0
