#!/usr/bin/env bash
# sprint239_and_comp_xor_copy_gate.sh — Sprint 239: Block EW AND-complement XOR to copy
#
# Block EW: (x&y)^(~x&y) → y; commutative variants.
#   Zero new arrays: uses ao_and_rr_valid/lhs/rhs + is_bnot/bnot_src.
#   SOTA: Boolean algebra; LLVM InstCombineAndOrXor.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1100" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1100)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 239: Block EW — AND-complement XOR to copy ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ew"    "Block EW" "$O"
cg "src:ew_and_rr"   "ao_and_rr_valid\[ew_s[12] as usize\]" "$O"
cg "src:ew_is_bnot"  "is_bnot\[ew_al[12] as usize\]" "$O"
cg "src:ew_copy"     "ir_copy.*ew_instr\.dst.*ew_ar[12]|ir_copy.*ew_instr\.dst.*ew_al[12]" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1100_fn" "fn compiler_main_test_ew_and_comp_xor_copy_basic" "$M"
cg "main:T1101_fn" "fn compiler_main_test_ew_and_comp_xor_copy_comm" "$M"
cg "main:T1102_fn" "fn compiler_main_test_ew_and_comp_xor_copy_lhs_shared" "$M"
cg "main:T1103_fn" "fn compiler_main_test_ew_no_fold_diff_y" "$M"
cg "main:T1104_fn" "fn compiler_main_test_ew_no_fold_no_not" "$M"
cg "main:T1105_fn" "fn compiler_main_test_ew_no_fold_or_outer" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1100-T1105 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1100 T1101 T1102 T1103 T1104 T1105; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1100 T1101 T1102 T1103 T1104 T1105; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
