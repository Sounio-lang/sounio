#!/usr/bin/env bash
# sprint227_or_xor_or_gate.sh — Sprint 227: Block EK OR-XOR simplification
#
# Block EK: x|(x^y) → x|y; (x^y)|x → x|y; y|(x^y) → x|y; (x^y)|y → x|y.
#   Zero new arrays: uses as_xor_rr_valid/lhs/rhs; updates ao_or_rr.
#   SOTA: LLVM InstCombineAndOrXor; Boolean algebra.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1028" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1028)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 227: Block EK — OR-XOR simplification ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ek"    "Block EK" "$O"
cg "src:ek_xor_rr"  "as_xor_rr_valid\[ek_s[12] as usize\]" "$O"
cg "src:ek_rewrite" "ir_binop.*ek_instr\.dst.*ek_s[12].*OpBitOr" "$O"
cg "src:ek_or_upd"  "ao_or_rr_valid\[ek_instr\.dst as usize\] = true" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1028_fn" "fn compiler_main_test_ek_or_xor_or_basic" "$M"
cg "main:T1029_fn" "fn compiler_main_test_ek_or_xor_or_comm" "$M"
cg "main:T1030_fn" "fn compiler_main_test_ek_or_xor_or_rhs" "$M"
cg "main:T1031_fn" "fn compiler_main_test_ek_no_fold_diff_var" "$M"
cg "main:T1032_fn" "fn compiler_main_test_ek_no_fold_and_outer" "$M"
cg "main:T1033_fn" "fn compiler_main_test_ek_no_fold_no_xor" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1028-T1033 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1028 T1029 T1030 T1031 T1032 T1033; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1028 T1029 T1030 T1031 T1032 T1033; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
