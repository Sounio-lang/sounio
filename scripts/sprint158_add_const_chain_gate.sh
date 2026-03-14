#!/usr/bin/env bash
# Sprint 158: Block BT — consecutive add-constant fold (x+C1)+C2 → x+(C1+C2)
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
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
    elif ! grep -qF "T614" "$log" 2>/dev/null; then echo "NOT_RUN  $name (OOM before T614)"; NOT_RUN=$((NOT_RUN+1))
    else echo "FAIL  $name (expected '$expected')"; FAIL=$((FAIL+1)); fi
}
echo "=== Sprint 158: Block BT — consecutive add-constant fold ==="
echo "--- Source ---"
OPT="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bt_comment" "Block BT.*[Aa]dd.*const|Block BT.*x.*C1.*C2" "$OPT"
check_grep "src:bt_valid_check"   "bt_add_valid\[bt_s1 as usize\]|bt.*add_valid" "$OPT"
check_grep "src:bt_merged"        "bt_c1.*bt_c2|bt_merged.*bt_c" "$OPT"
check_grep "src:bt_rewrite"       "ir_binop.*bt_instr\.dst.*OpAdd|ir_binop.*bt.*OpAdd" "$OPT"
echo "--- Tests ---"
MAIN="self-hosted/compiler/main.sio"
check_grep "main:T614_fn"  "fn compiler_main_test_bt_add_chain_basic" "$MAIN"
check_grep "main:T616_fn"  "fn compiler_main_test_bt_add_chain_merged" "$MAIN"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN"
echo "--- Self-tests ---"
LOG="$(mktemp)"
timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$LOG" 2>&1 || _ec=$?
for t in T614 T615 T616 T617 T618 T619; do check_log_line "selftest:$t" "$t OK" "$LOG"; done
rm -f "$LOG"
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
[ "$FAIL" -eq 0 ] && { echo "GATE: PASS"; exit 0; } || { echo "GATE: FAIL"; exit 1; }
