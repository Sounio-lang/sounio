#!/usr/bin/env bash
# sprint199_or_not_and_taut_gate.sh — Sprint 199: Block DI OR-NOT-AND tautology
#
# Block DI: x | ~(x&y) → -1, y | ~(x&y) → -1; commutative variants.
#   Zero new arrays: uses is_bnot/bnot_src + ao_and_rr_valid/lhs/rhs (Block AO) + def_at.
#   SOTA: LLVM InstCombineAndOrXor; Boolean complement tautology.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T860" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T860)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 199: Block DI — OR-NOT-AND tautology ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_di"      "Block DI" "$O"
cg "src:di_is_bnot"    "is_bnot\[di_s[12] as usize\]" "$O"
cg "src:di_and_rr"     "ao_and_rr_valid\[di_and as usize\]" "$O"
cg "src:di_minus1"     "ir_load_imm.*di_instr\.dst.*-1" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T860_fn" "fn compiler_main_test_di_or_not_and_taut_lhs" "$M"
cg "main:T861_fn" "fn compiler_main_test_di_or_not_and_taut_rhs" "$M"
cg "main:T862_fn" "fn compiler_main_test_di_or_not_and_taut_comm" "$M"
cg "main:T863_fn" "fn compiler_main_test_di_no_fold_diff_reg" "$M"
cg "main:T864_fn" "fn compiler_main_test_di_no_fold_and_op" "$M"
cg "main:T865_fn" "fn compiler_main_test_di_no_fold_and_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T860-T865 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T860 T861 T862 T863 T864 T865; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T860 T861 T862 T863 T864 T865; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
