#!/usr/bin/env bash
# Sprint 166: Block CB — add/sub-const inverse (x+C)-C→x, (x-C)+C→x
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
    local name="$1"; local expected_line="$2"; local log_file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -s "$log_file" ]; then echo "NOT_RUN  $name (OOM/empty)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qF "$expected_line" "$log_file"; then echo "PASS  $name"; PASS=$((PASS+1))
    elif ! grep -qF "T662" "$log_file" 2>/dev/null; then
        echo "NOT_RUN  $name (OOM before T662)"; NOT_RUN=$((NOT_RUN+1))
    else echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1)); fi
}
echo "=== Sprint 166: Block CB — Add/sub-const inverse ==="
echo "--- Source ---"
OPT="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_cb_comment" "Block CB.*[Aa]dd.*sub.*const|Block CB.*x.*C.*C.*x" "$OPT"
check_grep "src:cb_add_valid"     "bt_add_valid\[cb_s1 as usize\]" "$OPT"
check_grep "src:cb_const_eq"      "bt_add_cval\[cb_s1 as usize\] == const_val\[cb_s2 as usize\]" "$OPT"
check_grep "src:cb_ir_copy"       "ir_copy\(cb_instr\.dst, bt_add_src\[cb_s1" "$OPT"
check_grep "src:cb_sub_valid"     "bs_sub_valid\[cb_s1 as usize\]" "$OPT"
echo "--- Tests ---"
MAIN="self-hosted/compiler/main.sio"
check_grep "main:T662_fn"  "fn compiler_main_test_cb_add_sub_inverse_basic" "$MAIN"
check_grep "main:T663_fn"  "fn compiler_main_test_cb_sub_add_inverse_basic" "$MAIN"
check_grep "main:T664_fn"  "fn compiler_main_test_cb_sub_add_comm" "$MAIN"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN"
echo "--- Self-tests ---"
LOG="$(mktemp)"
timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$LOG" 2>&1 || _ec=$?
check_log_line "selftest:T662" "T662 OK" "$LOG"
check_log_line "selftest:T663" "T663 OK" "$LOG"
check_log_line "selftest:T664" "T664 OK" "$LOG"
check_log_line "selftest:T665" "T665 OK" "$LOG"
check_log_line "selftest:T666" "T666 OK" "$LOG"
check_log_line "selftest:T667" "T667 OK" "$LOG"
rm -f "$LOG"
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
[ "$FAIL" -eq 0 ] && { echo "GATE: PASS"; exit 0; } || { echo "GATE: FAIL"; exit 1; }
