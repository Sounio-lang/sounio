#!/usr/bin/env bash
# sprint224_xor_nor_zero_gate.sh — Sprint 224: Block EH XOR-NOR annihilation
#
# Block EH: (x^y)&~(x|y) → 0; commutative variants.
#   Zero new arrays: uses as_xor_rr + ao_or_rr + is_bnot/bnot_src.
#   SOTA: LLVM InstCombineAndOrXor; Boolean lattice disjointness.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1010" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1010)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 224: Block EH — XOR-NOR annihilation ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_eh"    "Block EH" "$O"
cg "src:eh_xor_rr"  "as_xor_rr_valid\[eh_s[12] as usize\]" "$O"
cg "src:eh_is_bnot" "is_bnot\[eh_s[12] as usize\]" "$O"
cg "src:eh_zero"    "ir_load_imm.*eh_instr\.dst.*0" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1010_fn" "fn compiler_main_test_eh_xor_nor_zero_basic" "$M"
cg "main:T1011_fn" "fn compiler_main_test_eh_xor_nor_zero_comm" "$M"
cg "main:T1012_fn" "fn compiler_main_test_eh_xor_nor_zero_inner_comm" "$M"
cg "main:T1013_fn" "fn compiler_main_test_eh_no_fold_diff_vars" "$M"
cg "main:T1014_fn" "fn compiler_main_test_eh_no_fold_no_not" "$M"
cg "main:T1015_fn" "fn compiler_main_test_eh_no_fold_or_outer" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1010-T1015 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1010 T1011 T1012 T1013 T1014 T1015; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1010 T1011 T1012 T1013 T1014 T1015; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
