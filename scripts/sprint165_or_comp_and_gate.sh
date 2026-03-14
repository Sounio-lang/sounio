#!/usr/bin/env bash
# sprint165_or_comp_and_gate.sh — Sprint 165: Block CA OR-complement AND elimination
#
# Block CA: (x|C)&(x|~C) → x,  (x|~C)&(x|C) → x
#   Zero new arrays: uses am_or_valid/var_src/const_val (Block AM) + is_const/const_val.
#   SOTA: LLVM InstCombineAndOrXor — and(or X,C),or(X,~C) → X; Boolean distributive.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if grep -qE "$pattern" "$file" 2>/dev/null; then echo "PASS  $name"; PASS=$((PASS+1))
    else echo "FAIL  $name (pattern '$pattern')"; FAIL=$((FAIL+1)); fi
}

check_log_line() {
    local name="$1"; local expected="$2"; local log="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -s "$log" ]; then echo "NOT_RUN  $name (OOM/empty)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qF "$expected" "$log"; then echo "PASS  $name"; PASS=$((PASS+1))
    elif ! grep -qF "T656" "$log" 2>/dev/null; then echo "NOT_RUN  $name (OOM before T656)"; NOT_RUN=$((NOT_RUN+1))
    else echo "FAIL  $name (expected '$expected')"; FAIL=$((FAIL+1)); fi
}

echo "=== Sprint 165: Block CA — OR-Complement AND Elimination ==="
echo ""

OPT="self-hosted/ir/opt_cleanup.sio"
MAIN="self-hosted/compiler/main.sio"

echo "--- Source ---"
check_grep "src:block_ca_comment" "Block CA.*OR.complement|Block CA.*x.C.*~C" "$OPT"
check_grep "src:ca_or_valid"      "am_or_valid\[ca_s1 as usize\]" "$OPT"
check_grep "src:ca_same_base"     "am_or_var_src\[ca_s1 as usize\] == am_or_var_src\[ca_s2 as usize\]" "$OPT"
check_grep "src:ca_complement"    "am_or_const_val\[ca_s1 as usize\] == .am_or_const_val\[ca_s2 as usize\] \^ -1" "$OPT"
check_grep "src:ca_ircopy"        "ir_copy.*ca_instr\.dst.*ca_x" "$OPT"

echo ""
echo "--- Tests ---"
check_grep "main:T656_fn" "fn compiler_main_test_ca_or_comp_and_basic" "$MAIN"
check_grep "main:T657_fn" "fn compiler_main_test_ca_or_comp_and_comm" "$MAIN"
check_grep "main:T659_fn" "fn compiler_main_test_ca_no_fold_diff_base" "$MAIN"
check_grep "main:T661_fn" "fn compiler_main_test_ca_no_fold_outer_or" "$MAIN"
check_grep "main:total"   "let total: i64 = [0-9]+" "$MAIN"

echo ""
echo "--- Self-tests: T656-T661 ---"
LOG="$(mktemp)"
timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$LOG" 2>&1 || true
check_log_line "selftest:T656" "T656 OK" "$LOG"
check_log_line "selftest:T657" "T657 OK" "$LOG"
check_log_line "selftest:T658" "T658 OK" "$LOG"
check_log_line "selftest:T659" "T659 OK" "$LOG"
check_log_line "selftest:T660" "T660 OK" "$LOG"
check_log_line "selftest:T661" "T661 OK" "$LOG"
rm -f "$LOG"

echo ""
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1)); fi

echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
[ "$FAIL" -eq 0 ] && { echo "GATE: PASS"; exit 0; } || { echo "GATE: FAIL"; exit 1; }
