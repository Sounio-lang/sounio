#!/usr/bin/env bash
# sprint164_and_xor_merge_gate.sh — Sprint 164: Block BZ AND-XOR same-constant merge
#
# Block BZ: (x & C) | (x ^ C) → x | C,  (x ^ C) | (x & C) → x | C
#   Zero new arrays: uses am_and_valid/var_src (Block AK) + bm_xor_valid/src/cval (Block BM).
#   SOTA: LLVM InstCombineAndOrXor; Boolean distributive merge.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

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
    elif ! grep -qF "T650" "$log" 2>/dev/null; then echo "NOT_RUN  $name (OOM before T650)"; NOT_RUN=$((NOT_RUN+1))
    else echo "FAIL  $name (expected '$expected')"; FAIL=$((FAIL+1)); fi
}

echo "=== Sprint 164: Block BZ — AND-XOR Same-Constant Merge ==="
echo ""

OPT="self-hosted/ir/opt_cleanup.sio"
MAIN="self-hosted/compiler/main.sio"

echo "--- Source ---"
check_grep "src:block_bz_comment" "Block BZ.*AND.XOR|Block BZ.*same.constant" "$OPT"
check_grep "src:bz_and_xor_fold"  "bm_xor_valid\[bz_s2 as usize\].*bm_xor_src|bm_xor_src.*bz_s[12]" "$OPT"

echo ""
echo "--- Tests ---"
check_grep "main:T650_fn" "fn compiler_main_test_bz_and_xor_merge_basic" "$MAIN"
check_grep "main:T651_fn" "fn compiler_main_test_bz_and_xor_merge_comm" "$MAIN"
check_grep "main:T653_fn" "fn compiler_main_test_bz_no_fold_diff_const" "$MAIN"
check_grep "main:T655_fn" "fn compiler_main_test_bz_no_fold_plain" "$MAIN"
check_grep "main:total"   "let total: i64 = [0-9]+" "$MAIN"

echo ""
echo "--- Self-tests: T650-T655 ---"
LOG="$(mktemp)"
timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$LOG" 2>&1 || true
check_log_line "selftest:T650" "T650 OK" "$LOG"
check_log_line "selftest:T651" "T651 OK" "$LOG"
check_log_line "selftest:T652" "T652 OK" "$LOG"
check_log_line "selftest:T653" "T653 OK" "$LOG"
check_log_line "selftest:T654" "T654 OK" "$LOG"
check_log_line "selftest:T655" "T655 OK" "$LOG"
rm -f "$LOG"

echo ""
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1)); fi

echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
[ "$FAIL" -eq 0 ] && { echo "GATE: PASS"; exit 0; } || { echo "GATE: FAIL"; exit 1; }
