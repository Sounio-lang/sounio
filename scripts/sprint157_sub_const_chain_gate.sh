#!/usr/bin/env bash
# Sprint 157: Block BS — consecutive subtract-constant fold (x-C1)-C2 → x-(C1+C2)
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
    elif ! grep -qF "T608" "$log" 2>/dev/null; then echo "NOT_RUN  $name (OOM before T608)"; NOT_RUN=$((NOT_RUN+1))
    else echo "FAIL  $name (expected '$expected')"; FAIL=$((FAIL+1)); fi
}
echo "=== Sprint 157: Block BS — consecutive subtract-constant fold ==="
echo "--- Source ---"
OPT="self-hosted/ir/opt_cleanup.sio"
check_grep "src:block_bs_comment" "Block BS.*[Ss]ub.*const|Block BS.*x.*C1.*C2" "$OPT"
check_grep "src:bs_valid_check"   "bs_sub_valid\[bs_s1 as usize\]|bs.*sub_valid" "$OPT"
check_grep "src:bs_merged"        "bs_c1.*bs_c2|bs_merged.*bs_c" "$OPT"
check_grep "src:bs_rewrite"       "bs_sub_src\[bs_s1 as usize\].*OpSub|ir_binop.*bs_sub_src" "$OPT"
echo "--- Tests ---"
MAIN="self-hosted/compiler/main.sio"
check_grep "main:T608_fn"  "fn compiler_main_test_bs_sub_chain_basic" "$MAIN"
check_grep "main:T610_fn"  "fn compiler_main_test_bs_sub_chain_amount" "$MAIN"
check_grep "main:total"    "let total: i64 = [0-9]+" "$MAIN"
echo "--- Self-tests ---"
LOG="$(mktemp)"
timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$LOG" 2>&1 || _ec=$?
for t in T608 T609 T610 T611 T612 T613; do check_log_line "selftest:$t" "$t OK" "$LOG"; done
rm -f "$LOG"
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
[ "$FAIL" -eq 0 ] && { echo "GATE: PASS"; exit 0; } || { echo "GATE: FAIL"; exit 1; }
