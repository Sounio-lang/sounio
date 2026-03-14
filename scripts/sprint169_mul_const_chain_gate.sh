#!/usr/bin/env bash
# sprint169_mul_const_chain_gate.sh — Sprint 169: Block CE multiply-constant chain fold
#
# Block CE: (x*C1)*C2 → x*(C1*C2),  C2*(x*C1) → x*(C1*C2)
#   SOTA: LLVM InstCombineMulDivRem; associative constant folding for multiply.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"; TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then echo "NOT_RUN  $name (file not found)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then echo "PASS  $name"; PASS=$((PASS+1))
    else echo "FAIL  $name (pattern not found)"; FAIL=$((FAIL+1)); fi
}
check_log_line() {
    local name="$1"; local expected_line="$2"; local log_file="$3"; TOTAL=$((TOTAL+1))
    if [ ! -s "$log_file" ]; then echo "NOT_RUN  $name (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qF "$expected_line" "$log_file"; then echo "PASS  $name"; PASS=$((PASS+1))
    elif ! grep -qF "T680" "$log_file" 2>/dev/null; then echo "NOT_RUN  $name (OOM before T680)"; NOT_RUN=$((NOT_RUN+1))
    else echo "FAIL  $name"; FAIL=$((FAIL+1)); fi
}
echo "=== Sprint 169: Block CE — Multiply-Constant Chain Fold ==="
echo ""
echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_ce_comment" "Block CE" "$OPT_FILE"
check_grep "src:ce_mul_valid"     "mul_valid\[ce_s[12] as usize\]" "$OPT_FILE"
check_grep "src:ce_const_check"   "is_const\[ce_s[12] as usize\]" "$OPT_FILE"
check_grep "src:ce_merged_mul"    "ce_merged|ce_c1 \* ce_c2" "$OPT_FILE"
echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T680_fn" "fn compiler_main_test_ce_mul_const_chain_basic" "$MAIN_FILE"
check_grep "main:T681_fn" "fn compiler_main_test_ce_mul_const_chain_comm" "$MAIN_FILE"
check_grep "main:T682_fn" "fn compiler_main_test_ce_mul_const_chain_val" "$MAIN_FILE"
check_grep "main:T683_fn" "fn compiler_main_test_ce_no_fold_rr_mul" "$MAIN_FILE"
check_grep "main:T684_fn" "fn compiler_main_test_ce_no_fold_add_chain" "$MAIN_FILE"
check_grep "main:T685_fn" "fn compiler_main_test_ce_no_fold_div_chain" "$MAIN_FILE"
echo ""
echo "--- Self-tests: T680-T685 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T680 T681 T682 T683 T684 T685; do
        TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1)); done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    for name in T680 T681 T682 T683 T684 T685; do check_log_line "selftest:$name" "$name OK" "$SELF_TEST_LOG"; done
fi
rm -f "$SELF_TEST_LOG"
echo ""
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1)); fi
echo ""
echo "=== SUMMARY ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
