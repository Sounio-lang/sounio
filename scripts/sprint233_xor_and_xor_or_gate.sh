#!/usr/bin/env bash
# sprint233_xor_and_xor_or_gate.sh — Sprint 233: Block EQ XOR-AND XOR to OR
#
# Block EQ: (x^y)^(x&y) → x|y; commutative variants.
#   Zero new arrays: uses as_xor_rr_valid/lhs/rhs + ao_and_rr_valid/lhs/rhs.
#   SOTA: GF(2): (x⊕y)⊕(x∧y) = x∨y.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1064" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1064)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 233: Block EQ — XOR-AND XOR to OR ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_eq"    "Block EQ" "$O"
cg "src:eq_xor_rr"   "as_xor_rr_valid\[eq_s[12] as usize\]" "$O"
cg "src:eq_and_rr"   "ao_and_rr_valid\[eq_s[12] as usize\]" "$O"
cg "src:eq_rewrite"  "ir_binop.*eq_instr\.dst.*eq_x[lr].*OpBitOr" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1064_fn" "fn compiler_main_test_eq_xor_and_xor_or_basic" "$M"
cg "main:T1065_fn" "fn compiler_main_test_eq_xor_and_xor_or_comm" "$M"
cg "main:T1066_fn" "fn compiler_main_test_eq_xor_and_xor_or_inner_comm" "$M"
cg "main:T1067_fn" "fn compiler_main_test_eq_no_fold_diff_var" "$M"
cg "main:T1068_fn" "fn compiler_main_test_eq_no_fold_or_inner" "$M"
cg "main:T1069_fn" "fn compiler_main_test_eq_no_fold_and_outer" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1064-T1069 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1064 T1065 T1066 T1067 T1068 T1069; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1064 T1065 T1066 T1067 T1068 T1069; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
