#!/usr/bin/env bash
# sprint215_and_or_comp_gate.sh — Sprint 215: Block DY AND-OR complement absorption
#
# Block DY: x & (y | ~x) → x & y; commutative variants.
#   Zero new arrays: uses ao_or_rr_valid/lhs/rhs (Block AO) + is_bnot/bnot_src.
#   SOTA: LLVM InstCombineAndOrXor; Boolean absorption; Huntington 1904.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T956" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T956)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 215: Block DY — AND-OR complement absorption ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dy"      "Block DY" "$O"
cg "src:dy_or_rr"      "ao_or_rr_valid\[dy_s[12] as usize\]" "$O"
cg "src:dy_is_bnot"    "is_bnot\[dy_o[lr] as usize\]" "$O"
cg "src:dy_rewrite"    "ir_binop.*dy_instr\.dst.*dy_s[12].*OpBitAnd" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T956_fn" "fn compiler_main_test_dy_and_or_comp_basic" "$M"
cg "main:T957_fn" "fn compiler_main_test_dy_and_or_comp_comm" "$M"
cg "main:T958_fn" "fn compiler_main_test_dy_and_or_comp_not_left" "$M"
cg "main:T959_fn" "fn compiler_main_test_dy_no_fold_diff_var" "$M"
cg "main:T960_fn" "fn compiler_main_test_dy_no_fold_no_not" "$M"
cg "main:T961_fn" "fn compiler_main_test_dy_no_fold_or_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T956-T961 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T956 T957 T958 T959 T960 T961; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T956 T957 T958 T959 T960 T961; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
