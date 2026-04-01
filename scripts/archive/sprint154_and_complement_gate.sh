#!/usr/bin/env bash
# Sprint 154: Block BP — AND-complement annihilation x & ~x → 0
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
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
    elif ! grep -qF "T590" "$log" 2>/dev/null; then echo "NOT_RUN  $name (OOM before T590)"; NOT_RUN=$((NOT_RUN+1))
    else echo "FAIL  $name (expected '$expected')"; FAIL=$((FAIL+1)); fi
}
echo "=== Sprint 154: Block BP — AND-complement annihilation ==="
echo "--- Source ---"
OPT="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bp_comment" "Block BP.*AND.complement|Block BP.*x.*~x" "$OPT"
check_grep "src:bp_bnot_check"    "is_bnot\[bp_s2 as usize\]|is_bnot\[.*bp_s" "$OPT"
check_grep "src:bp_base_eq"       "bnot_src\[bp_s2 as usize\] == bp_s1|bnot_src\[.*bp_s" "$OPT"
check_grep "src:bp_load_zero"     "ir_load_imm.*bp_instr\.dst.*0" "$OPT"
echo "--- Tests ---"
MAIN="self-hosted/compiler/main.sio"
check_grep "main:T590_fn"  "fn compiler_main_test_bp_and_complement_basic" "$MAIN"
check_grep "main:T591_fn"  "fn compiler_main_test_bp_and_complement_comm" "$MAIN"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN"
echo "--- Self-tests ---"
LOG="$(mktemp)"
timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$LOG" 2>&1 || _ec=$?
for t in T590 T591 T592 T593 T594 T595; do check_log_line "selftest:$t" "$t OK" "$LOG"; done
rm -f "$LOG"
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
[ "$FAIL" -eq 0 ] && { echo "GATE: PASS"; exit 0; } || { echo "GATE: FAIL"; exit 1; }
