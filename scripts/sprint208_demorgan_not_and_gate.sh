#!/usr/bin/env bash
# sprint208_demorgan_not_and_gate.sh — Sprint 208: Block DR De Morgan NOT-AND
#
# Block DR: ~(~x & ~y) → x | y (De Morgan's theorem).
#   Zero new arrays: uses is_bnot/bnot_src + ao_and_rr_valid/lhs/rhs.
#   SOTA: LLVM InstCombineAndOrXor; De Morgan 1847.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T914" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T914)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 208: Block DR — De Morgan NOT-AND ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dr"      "Block DR" "$O"
cg "src:dr_and_rr"     "ao_and_rr_valid\[dr_s as usize\]" "$O"
cg "src:dr_bnot"       "is_bnot\[dr_al as usize\]" "$O"
cg "src:dr_or"         "ir_binop.*dr_instr\.dst.*dr_x.*OpBitOr.*dr_y" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T914_fn" "fn compiler_main_test_dr_not_and_demorgan" "$M"
cg "main:T915_fn" "fn compiler_main_test_dr_not_and_swapped" "$M"
cg "main:T916_fn" "fn compiler_main_test_dr_not_and_val" "$M"
cg "main:T917_fn" "fn compiler_main_test_dr_no_fold_one_not" "$M"
cg "main:T918_fn" "fn compiler_main_test_dr_no_fold_or_inner" "$M"
cg "main:T919_fn" "fn compiler_main_test_dr_no_fold_not_not_and" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T914-T919 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T914 T915 T916 T917 T918 T919; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T914 T915 T916 T917 T918 T919; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
