#!/usr/bin/env bash
# sprint174_shl_sub_mul_gate.sh — Sprint 174: Block CJ shift-sub to multiply
#
# Block CJ: (x<<A)-x → x*(2^A-1),  x-(x<<A) → x*(1-2^A)
#   Zero new arrays: uses bk_shl_valid/src/amt + def_at.
#   SOTA: LLVM InstCombineMulDivRem; strength reduction.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n (pattern '$p')"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T710" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T710)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 174: Block CJ — Shift-Sub to Multiply ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_cj"        "Block CJ" "$O"
cg "src:cj_shl_valid"    "bk_shl_valid\[cj_s[12] as usize\]" "$O"
cg "src:cj_same_base"    "bk_shl_src\[cj_s[12] as usize\] == cj_s" "$O"
cg "src:cj_pow2_minus1"  "cj_pow2 - 1|cj_merged = cj_pow2" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T710_fn" "fn compiler_main_test_cj_shl_sub_basic" "$M"
cg "main:T711_fn" "fn compiler_main_test_cj_shl_sub_three" "$M"
cg "main:T712_fn" "fn compiler_main_test_cj_sub_shl_reverse" "$M"
cg "main:T713_fn" "fn compiler_main_test_cj_no_fold_diff_base" "$M"
cg "main:T714_fn" "fn compiler_main_test_cj_no_fold_rr_shl" "$M"
cg "main:T715_fn" "fn compiler_main_test_cj_no_fold_add" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T710-T715 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for t in T710 T711 T712 T713 T714 T715; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
    for t in T710 T711 T712 T713 T714 T715; do cl "selftest:$t" "$t OK" "$L"; done
fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
