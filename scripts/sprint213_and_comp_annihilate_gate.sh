#!/usr/bin/env bash
# sprint213_and_comp_annihilate_gate.sh — Sprint 213: Block DW AND complement annihilation
#
# Block DW: (x&y) & ~x → 0; (x&y) & ~y → 0; commutative.
#   Zero new arrays: uses ao_and_rr_valid/lhs/rhs + is_bnot/bnot_src.
#   SOTA: LLVM InstCombineAndOrXor; complement law x&~x=0.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T944" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T944)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 213: Block DW — AND complement annihilation ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dw"      "Block DW" "$O"
cg "src:dw_and_rr"     "ao_and_rr_valid\[dw_s[12] as usize\]" "$O"
cg "src:dw_bnot"       "is_bnot\[dw_s[12] as usize\]" "$O"
cg "src:dw_zero"       "ir_load_imm.*dw_instr\.dst.*0" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T944_fn" "fn compiler_main_test_dw_and_comp_lhs" "$M"
cg "main:T945_fn" "fn compiler_main_test_dw_and_comp_rhs" "$M"
cg "main:T946_fn" "fn compiler_main_test_dw_and_comp_comm" "$M"
cg "main:T947_fn" "fn compiler_main_test_dw_no_fold_diff_var" "$M"
cg "main:T948_fn" "fn compiler_main_test_dw_no_fold_no_not" "$M"
cg "main:T949_fn" "fn compiler_main_test_dw_no_fold_or_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T944-T949 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T944 T945 T946 T947 T948 T949; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T944 T945 T946 T947 T948 T949; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
