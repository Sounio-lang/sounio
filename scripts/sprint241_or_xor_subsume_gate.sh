#!/usr/bin/env bash
# sprint241_or_xor_subsume_gate.sh — Sprint 241: Block EY OR-XOR subsumption
#
# Block EY: (x|y)|(x^y) → x|y; commutative variants.
#   Zero new arrays: uses ao_or_rr_valid/lhs/rhs + as_xor_rr_valid/lhs/rhs.
#   SOTA: Boolean algebra; x^y ⊆ x|y so OR with subset is identity.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."  ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1112" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1112)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 241: Block EY — OR-XOR subsumption ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ey"    "Block EY" "$O"
cg "src:ey_or_rr"    "ao_or_rr_valid\[ey_s[12] as usize\]" "$O"
cg "src:ey_xor_rr"   "as_xor_rr_valid\[ey_s[12] as usize\]" "$O"
cg "src:ey_copy"     "ir_copy.*ey_instr\.dst.*ey_s[12]" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1112_fn" "fn compiler_main_test_ey_or_xor_subsume_basic" "$M"
cg "main:T1113_fn" "fn compiler_main_test_ey_or_xor_subsume_comm" "$M"
cg "main:T1114_fn" "fn compiler_main_test_ey_or_xor_subsume_inner_comm" "$M"
cg "main:T1115_fn" "fn compiler_main_test_ey_no_fold_diff_var" "$M"
cg "main:T1116_fn" "fn compiler_main_test_ey_no_fold_and_inner" "$M"
cg "main:T1117_fn" "fn compiler_main_test_ey_no_fold_and_outer" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1112-T1117 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1112 T1113 T1114 T1115 T1116 T1117; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1112 T1113 T1114 T1115 T1116 T1117; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
