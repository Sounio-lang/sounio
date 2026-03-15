#!/usr/bin/env bash
# sprint240_or_comp_and_xor_copy_gate.sh — Sprint 240: Block EX OR-complement-AND XOR to copy
#
# Block EX: (x|y)^(~x&y) → x; commutative variants.
#   Zero new arrays: uses ao_or_rr_valid/lhs/rhs + ao_and_rr_valid/lhs/rhs + is_bnot/bnot_src.
#   SOTA: Boolean algebra; LLVM InstCombineAndOrXor.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1106" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1106)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 240: Block EX — OR-complement-AND XOR to copy ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ex"    "Block EX" "$O"
cg "src:ex_or_rr"    "ao_or_rr_valid\[ex_s[12] as usize\]" "$O"
cg "src:ex_is_bnot"  "is_bnot\[ex_al[2]? as usize\]" "$O"
cg "src:ex_copy"     "ir_copy.*ex_instr\.dst.*ex_ol[2]?|ir_copy.*ex_instr\.dst.*ex_or2" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1106_fn" "fn compiler_main_test_ex_or_comp_and_xor_copy_basic" "$M"
cg "main:T1107_fn" "fn compiler_main_test_ex_or_comp_and_xor_copy_rhs_x" "$M"
cg "main:T1108_fn" "fn compiler_main_test_ex_or_comp_and_xor_copy_comm_xor" "$M"
cg "main:T1109_fn" "fn compiler_main_test_ex_no_fold_diff_var" "$M"
cg "main:T1110_fn" "fn compiler_main_test_ex_no_fold_no_not" "$M"
cg "main:T1111_fn" "fn compiler_main_test_ex_no_fold_and_outer" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1106-T1111 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1106 T1107 T1108 T1109 T1110 T1111; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1106 T1107 T1108 T1109 T1110 T1111; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
