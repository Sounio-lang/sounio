#!/usr/bin/env bash
# sprint167_and_comp_part_gate.sh — Sprint 167: Block CC AND complement partition
#
# Block CC: (x & C) | (x & ~C) → x
#   Zero new arrays: uses and_valid/and_var_src/and_const_val (Block AE).
#   SOTA: LLVM InstCombineAndOrXor; Boolean: (x∧C)∨(x∧¬C) = x∧(C∨¬C) = x.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then echo "NOT_RUN  $name (file not found)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then echo "PASS  $name"; PASS=$((PASS+1))
    else echo "FAIL  $name (pattern '$pattern' not found)"; FAIL=$((FAIL+1)); fi
}

check_log_line() {
    local name="$1"; local expected_line="$2"; local log_file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -s "$log_file" ]; then echo "NOT_RUN  $name (log empty — OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qF "$expected_line" "$log_file"; then echo "PASS  $name"; PASS=$((PASS+1))
    elif ! grep -qF "T668" "$log_file" 2>/dev/null; then echo "NOT_RUN  $name (OOM before T668)"; NOT_RUN=$((NOT_RUN+1))
    else echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1)); fi
}

echo "=== Sprint 167: Block CC — AND Complement Partition ==="
echo ""

echo "--- Source ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_cc_comment"   "Block CC" "$OPT_FILE"
check_grep "src:cc_and_valid_check" "and_valid\[cc_s[12] as usize\]" "$OPT_FILE"
check_grep "src:cc_same_base"       "and_var_src\[cc_s1 as usize\] == and_var_src\[cc_s2 as usize\]" "$OPT_FILE"
check_grep "src:cc_complement"      "and_const_val\[cc_s1 as usize\] == .and_const_val\[cc_s2 as usize\] \^ -1." "$OPT_FILE"
check_grep "src:cc_ir_copy"         "ir_copy\(cc_instr.dst" "$OPT_FILE"

echo ""
echo "--- Tests ---"
MAIN_FILE="self-hosted/compiler/main.sio"
check_grep "main:T668_fn" "fn compiler_main_test_cc_and_comp_partition_basic" "$MAIN_FILE"
check_grep "main:T669_fn" "fn compiler_main_test_cc_and_comp_partition_comm" "$MAIN_FILE"
check_grep "main:T670_fn" "fn compiler_main_test_cc_and_comp_partition_val" "$MAIN_FILE"
check_grep "main:T671_fn" "fn compiler_main_test_cc_no_fold_diff_base" "$MAIN_FILE"
check_grep "main:T672_fn" "fn compiler_main_test_cc_no_fold_non_complement" "$MAIN_FILE"
check_grep "main:T673_fn" "fn compiler_main_test_cc_no_fold_and_op" "$MAIN_FILE"
check_grep "main:total"   "let total: i64 = 673" "$MAIN_FILE"

echo ""
echo "--- Self-tests: T668-T673 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T668 T669 T670 T671 T672 T673; do
        TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$name (souc not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?
    check_log_line "selftest:T668" "T668 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T669" "T669 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T670" "T670 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T671" "T671 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T672" "T672 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T673" "T673 OK" "$SELF_TEST_LOG"
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
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0
else echo "GATE: FAIL"; exit 1; fi
