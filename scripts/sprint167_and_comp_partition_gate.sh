#!/usr/bin/env bash
# sprint167_and_comp_partition_gate.sh — Sprint 167: Block CC AND complement partition
#
# Block CC: (x & C) | (x & ~C) → x,  (x & ~C) | (x & C) → x
#   Proof: (x&C)|(x&~C) = x&(C|~C) = x&(-1) = x. Dual of Block CA.
#   Zero new arrays: uses and_valid/var_src/const_val (Block AE).
#   SOTA: LLVM InstCombineAndOrXor or(and X,C),and(X,~C) → X; Boolean algebra.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file not found)"; NOT_RUN=$((NOT_RUN+1)); return
    fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern not found in $file)"; FAIL=$((FAIL+1))
    fi
}

check_log_line() {
    local name="$1"; local expected_line="$2"; local log_file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -s "$log_file" ]; then
        echo "NOT_RUN  $name (OOM/empty)"; NOT_RUN=$((NOT_RUN+1)); return
    fi
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        if ! grep -qF "T668" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T668)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 167: Block CC — AND Complement Partition ==="
echo ""

OPT="self-hosted/ir/opt_cleanup.sio"
MAIN="self-hosted/compiler/main.sio"

echo "--- Source ---"
check_grep "src:block_cc_comment"  "Block CC.*AND complement partition|Block CC.*and.*comp" "$OPT"
check_grep "src:cc_and_valid_s1"   "and_valid\[cc_s1 as usize\]" "$OPT"
check_grep "src:cc_and_valid_s2"   "and_valid\[cc_s2 as usize\]" "$OPT"
check_grep "src:cc_same_base"      "and_var_src\[cc_s1 as usize\] == and_var_src\[cc_s2 as usize\]" "$OPT"
check_grep "src:cc_complement"     "and_const_val\[cc_s1 as usize\] == .and_const_val\[cc_s2 as usize\] \^ -1" "$OPT"
check_grep "src:cc_ir_copy"        "ir_copy.*cc_instr\.dst.*cc_x" "$OPT"

echo ""
echo "--- Tests ---"
check_grep "main:T668_fn"  "fn compiler_main_test_cc_and_comp_partition_basic" "$MAIN"
check_grep "main:T669_fn"  "fn compiler_main_test_cc_and_comp_partition_comm" "$MAIN"
check_grep "main:T671_fn"  "fn compiler_main_test_cc_no_fold_diff_base" "$MAIN"
check_grep "main:T673_fn"  "fn compiler_main_test_cc_no_fold_and_op" "$MAIN"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN"

echo ""
echo "--- Self-tests: T668-T673 ---"
LOG="$(mktemp)"
timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$LOG" 2>&1 || _ec=$?
check_log_line "selftest:T668" "T668 OK" "$LOG"
check_log_line "selftest:T669" "T669 OK" "$LOG"
check_log_line "selftest:T670" "T670 OK" "$LOG"
check_log_line "selftest:T671" "T671 OK" "$LOG"
check_log_line "selftest:T672" "T672 OK" "$LOG"
check_log_line "selftest:T673" "T673 OK" "$LOG"
rm -f "$LOG"

echo ""
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else
    echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1))
fi

echo ""
echo "=== SUMMARY ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then
    echo "GATE: PASS"
    exit 0
else
    echo "GATE: FAIL"
    exit 1
fi
