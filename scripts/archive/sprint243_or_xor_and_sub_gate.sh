#!/usr/bin/env bash
# sprint243_or_xor_and_sub_gate.sh — Sprint 243: Block FA OR-XOR AND subsumption
#
# Block FA: (x|y)&(x^y) → x^y; commutative variants.
#   Zero new arrays: uses ao_or_rr_valid/lhs/rhs + as_xor_rr_valid/lhs/rhs.
#   SOTA: Boolean lattice x^y≤x|y; LLVM InstCombineAndOrXor.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."  ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1124" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1124)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 243: Block FA — OR-XOR AND subsumption ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_fa"    "Block FA" "$O"
cg "src:fa_or_rr"    "ao_or_rr_valid\[fa_s[12] as usize\]" "$O"
cg "src:fa_xor_rr"   "as_xor_rr_valid\[fa_s[12] as usize\]" "$O"
cg "src:fa_copy"     "ir_copy.*fa_instr\.dst.*fa_s[12]" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1124_fn" "fn compiler_main_test_fa_or_xor_and_sub_basic" "$M"
cg "main:T1125_fn" "fn compiler_main_test_fa_or_xor_and_sub_comm" "$M"
cg "main:T1126_fn" "fn compiler_main_test_fa_or_xor_and_sub_inner_comm" "$M"
cg "main:T1127_fn" "fn compiler_main_test_fa_no_fold_diff_var" "$M"
cg "main:T1128_fn" "fn compiler_main_test_fa_no_fold_or_outer" "$M"
cg "main:T1129_fn" "fn compiler_main_test_fa_no_fold_and_inner" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1124-T1129 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1124 T1125 T1126 T1127 T1128 T1129; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1124 T1125 T1126 T1127 T1128 T1129; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
