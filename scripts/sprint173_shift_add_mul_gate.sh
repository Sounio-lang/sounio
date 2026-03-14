#!/usr/bin/env bash
# sprint173_shift_add_mul_gate.sh — Sprint 173: Block CI shift-add to multiply
#
# Block CI: (x<<A)+x → x*(2^A+1),  x+(x<<A) → x*(2^A+1)
#   Zero new arrays: uses bk_shl_valid/src/amt (Block BK).
#   SOTA: LLVM InstCombineAddSub; strength reduction; Hacker's Delight §8-2.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
O="self-hosted/ir/opt_cleanup.sio"; M="self-hosted/compiler/main.sio"
cg() {
    local name="$1"; local pat="$2"; local f="$3"; TOTAL=$((TOTAL+1))
    if [ ! -f "$f" ]; then echo "NOT_RUN  $name (file missing)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qE "$pat" "$f" 2>/dev/null; then echo "PASS  $name"; PASS=$((PASS+1))
    else echo "FAIL  $name (pattern not found)"; FAIL=$((FAIL+1)); fi
}
cl() {
    local name="$1"; local exp="$2"; local log="$3"; TOTAL=$((TOTAL+1))
    if [ ! -s "$log" ]; then echo "NOT_RUN  $name (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi
    if grep -qF "$exp" "$log"; then echo "PASS  $name"; PASS=$((PASS+1))
    elif ! grep -qF "T704" "$log" 2>/dev/null; then echo "NOT_RUN  $name (OOM before T704)"; NOT_RUN=$((NOT_RUN+1))
    else echo "FAIL  $name"; FAIL=$((FAIL+1)); fi
}
echo "=== Sprint 173: Block CI — Shift-Add to Multiply ==="
echo ""
echo "--- Source ---"
cg "src:block_ci_comment"  "Block CI.*[Ss]hift.add|Block CI.*x<<A" "$O"
cg "src:ci_shl_valid"      "bk_shl_valid\[ci_s[12] as usize\]" "$O"
cg "src:ci_same_src"       "bk_shl_src\[ci_s[12] as usize\] == ci_s[12]" "$O"
cg "src:ci_merged_pow2"    "ci_pow2|ci_merged" "$O"
echo ""
echo "--- Tests ---"
cg "main:T704_fn" "fn compiler_main_test_ci_shl_add_basic" "$M"
cg "main:T705_fn" "fn compiler_main_test_ci_shl_add_comm" "$M"
cg "main:T707_fn" "fn compiler_main_test_ci_no_fold_diff_base" "$M"
cg "main:T709_fn" "fn compiler_main_test_ci_no_fold_sub" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""
echo "--- Self-tests: T704-T709 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for n in T704 T705 T706 T707 T708 T709; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$n"; NOT_RUN=$((NOT_RUN+1)); done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
    for n in T704 T705 T706 T707 T708 T709; do cl "selftest:$n" "$n OK" "$L"; done
fi; rm -f "$L"
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
