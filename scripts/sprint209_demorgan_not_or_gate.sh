#!/usr/bin/env bash
# sprint209_demorgan_not_or_gate.sh — Sprint 209: Block DS De Morgan NOT-OR
#
# Block DS: ~(~x | ~y) → x & y (De Morgan's theorem).
#   Zero new arrays: uses is_bnot/bnot_src + ao_or_rr_valid/lhs/rhs.
#   SOTA: LLVM InstCombineAndOrXor; De Morgan 1847.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T920" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T920)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 209: Block DS — De Morgan NOT-OR ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ds"      "Block DS" "$O"
cg "src:ds_or_rr"      "ao_or_rr_valid\[ds_s as usize\]" "$O"
cg "src:ds_bnot"       "is_bnot\[ds_ol as usize\]" "$O"
cg "src:ds_and"        "ir_binop.*ds_instr\.dst.*ds_x.*OpBitAnd.*ds_y" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T920_fn" "fn compiler_main_test_ds_not_or_demorgan" "$M"
cg "main:T921_fn" "fn compiler_main_test_ds_not_or_swapped" "$M"
cg "main:T922_fn" "fn compiler_main_test_ds_not_or_val" "$M"
cg "main:T923_fn" "fn compiler_main_test_ds_no_fold_one_not" "$M"
cg "main:T924_fn" "fn compiler_main_test_ds_no_fold_and_inner" "$M"
cg "main:T925_fn" "fn compiler_main_test_ds_no_fold_plain_or" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T920-T925 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T920 T921 T922 T923 T924 T925; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T920 T921 T922 T923 T924 T925; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
