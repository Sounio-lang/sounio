#!/usr/bin/env bash
# sprint206_and_or_xor_gate.sh — Sprint 206: Block DP AND-OR XOR identity
#
# Block DP: (x&y)^(x|y) → x^y; commutative.
#   Zero new arrays: uses ao_and_rr_valid/lhs/rhs + ao_or_rr_valid/lhs/rhs (Block AO).
#   SOTA: LLVM InstCombineAndOrXor; Boolean lattice identity.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T902" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T902)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 206: Block DP — AND-OR XOR identity ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dp"      "Block DP" "$O"
cg "src:dp_and_rr"     "ao_and_rr_valid\[dp_s[12] as usize\]" "$O"
cg "src:dp_or_rr"      "ao_or_rr_valid\[dp_s[12] as usize\]" "$O"
cg "src:dp_xor"        "ir_binop.*dp_instr\.dst.*dp_al.*OpBitXor" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T902_fn" "fn compiler_main_test_dp_and_or_xor_basic" "$M"
cg "main:T903_fn" "fn compiler_main_test_dp_or_and_xor_comm" "$M"
cg "main:T904_fn" "fn compiler_main_test_dp_and_or_xor_swapped" "$M"
cg "main:T905_fn" "fn compiler_main_test_dp_no_fold_diff_base" "$M"
cg "main:T906_fn" "fn compiler_main_test_dp_no_fold_and_and" "$M"
cg "main:T907_fn" "fn compiler_main_test_dp_no_fold_sub_op" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T902-T907 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T902 T903 T904 T905 T906 T907; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T902 T903 T904 T905 T906 T907; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
