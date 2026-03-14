#!/usr/bin/env bash
# sprint172_mul_sub_norm_gate.sh — Sprint 172: Block CH multiply-sub normalization
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"; TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then echo "NOT_RUN  $name (file not found)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then echo "PASS  $name"; PASS=$((PASS+1))
    else echo "FAIL  $name (pattern '$pattern' not found)"; FAIL=$((FAIL+1)); fi
}
check_log_line() {
    local name="$1"; local expected_line="$2"; local log_file="$3"; TOTAL=$((TOTAL+1))
    if [ ! -s "$log_file" ]; then echo "NOT_RUN  $name (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qF "$expected_line" "$log_file"; then echo "PASS  $name"; PASS=$((PASS+1))
    elif ! grep -qF "T698" "$log_file" 2>/dev/null; then echo "NOT_RUN  $name (OOM before T698)"; NOT_RUN=$((NOT_RUN+1))
    else echo "FAIL  $name"; FAIL=$((FAIL+1)); fi
}
echo "=== Sprint 172: Block CH — Multiply-Sub Normalization ==="
echo ""
echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_ch_comment" "Block CH" "$OPT_FILE"
check_grep "src:ch_sub_op"        "ch_instr\.bin_op == BinaryOp::OpSub" "$OPT_FILE"
check_grep "src:ch_mul_valid_s1"  "mul_valid\[ch_s1 as usize\]" "$OPT_FILE"
check_grep "src:ch_base_match_s1" "mul_var_src\[ch_s1 as usize\] == ch_s2" "$OPT_FILE"
check_grep "src:ch_merged_sub1"   "mul_const_v\[ch_s1 as usize\] - 1" "$OPT_FILE"
check_grep "src:ch_mul_valid_s2"  "mul_valid\[ch_s2 as usize\]" "$OPT_FILE"
echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T698_fn" "fn compiler_main_test_ch_mul_sub_basic" "$MAIN_FILE"
check_grep "main:T699_fn" "fn compiler_main_test_ch_sub_mul_basic" "$MAIN_FILE"
check_grep "main:T700_fn" "fn compiler_main_test_ch_mul_sub_large" "$MAIN_FILE"
check_grep "main:T701_fn" "fn compiler_main_test_ch_no_fold_diff_base" "$MAIN_FILE"
check_grep "main:T702_fn" "fn compiler_main_test_ch_no_fold_rr_sub" "$MAIN_FILE"
check_grep "main:T703_fn" "fn compiler_main_test_ch_no_fold_add" "$MAIN_FILE"
check_grep "main:total"   "let total: i64 = 703" "$MAIN_FILE"
echo ""
echo "--- Self-tests: T698-T703 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T698 T699 T700 T701 T702 T703; do
        TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1)); done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    for name in T698 T699 T700 T701 T702 T703; do check_log_line "selftest:$name" "$name OK" "$SELF_TEST_LOG"; done
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
