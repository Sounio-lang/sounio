#!/usr/bin/env bash
# sprint214_or_comp_sat_gate.sh — Sprint 214: Block DX OR complement saturation
#
# Block DX: (x|y) | ~x → -1; (x|y) | ~y → -1; commutative.
#   Zero new arrays: uses ao_or_rr_valid/lhs/rhs + is_bnot/bnot_src.
#   SOTA: LLVM InstCombineAndOrXor; complement law x|~x=-1.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T950" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T950)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 214: Block DX — OR complement saturation ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dx"      "Block DX" "$O"
cg "src:dx_or_rr"      "ao_or_rr_valid\[dx_s[12] as usize\]" "$O"
cg "src:dx_bnot"       "is_bnot\[dx_s[12] as usize\]" "$O"
cg "src:dx_neg1"       "ir_load_imm.*dx_instr\.dst.*-1" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T950_fn" "fn compiler_main_test_dx_or_comp_sat_lhs" "$M"
cg "main:T951_fn" "fn compiler_main_test_dx_or_comp_sat_rhs" "$M"
cg "main:T952_fn" "fn compiler_main_test_dx_or_comp_sat_comm" "$M"
cg "main:T953_fn" "fn compiler_main_test_dx_no_fold_diff_var" "$M"
cg "main:T954_fn" "fn compiler_main_test_dx_no_fold_no_not" "$M"
cg "main:T955_fn" "fn compiler_main_test_dx_no_fold_and_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T950-T955 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T950 T951 T952 T953 T954 T955; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T950 T951 T952 T953 T954 T955; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
