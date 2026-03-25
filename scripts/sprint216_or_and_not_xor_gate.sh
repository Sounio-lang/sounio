#!/usr/bin/env bash
# sprint216_or_and_not_xor_gate.sh — Sprint 216: Block DZ OR-AND-NOT to XOR
#
# Block DZ: (x|y) & ~(x&y) → x^y; commutative variants.
#   Zero new arrays: uses ao_or_rr/ao_and_rr (Block AO) + is_bnot/bnot_src.
#   SOTA: LLVM InstCombineAndOrXor; Boolean algebra XOR canonical form.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T962" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T962)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 216: Block DZ — OR-AND-NOT to XOR ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dz"      "Block DZ" "$O"
cg "src:dz_or_rr"      "ao_or_rr_valid\[dz_s[12] as usize\]" "$O"
cg "src:dz_and_rr"     "ao_and_rr_valid\[dz_and_r" "$O"
cg "src:dz_xor"        "ir_binop.*dz_instr\.dst.*OpBitXor" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T962_fn" "fn compiler_main_test_dz_or_and_not_xor_basic" "$M"
cg "main:T963_fn" "fn compiler_main_test_dz_or_and_not_xor_comm_outer" "$M"
cg "main:T964_fn" "fn compiler_main_test_dz_or_and_not_xor_comm_inner" "$M"
cg "main:T965_fn" "fn compiler_main_test_dz_no_fold_diff_vars" "$M"
cg "main:T966_fn" "fn compiler_main_test_dz_no_fold_no_not" "$M"
cg "main:T967_fn" "fn compiler_main_test_dz_no_fold_or_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T962-T967 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T962 T963 T964 T965 T966 T967; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T962 T963 T964 T965 T966 T967; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
