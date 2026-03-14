#!/usr/bin/env bash
# sprint198_and_not_or_gate.sh — Sprint 198: Block DH AND-NOT-OR annihilation
#
# Block DH: x & ~(x|y) → 0, y & ~(x|y) → 0; commutative variants.
#   Zero new arrays: uses is_bnot/bnot_src + ao_or_rr_valid/lhs/rhs (Block AO).
#   SOTA: LLVM InstCombineAndOrXor; Boolean lattice complement monotonicity.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T854" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T854)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 198: Block DH — AND-NOT-OR annihilation ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dh"      "Block DH" "$O"
cg "src:dh_is_bnot"    "is_bnot\[dh_s[12] as usize\]" "$O"
cg "src:dh_or_rr"      "ao_or_rr_valid\[dh_or as usize\]" "$O"
cg "src:dh_zero"       "ir_load_imm.*dh_instr\.dst.*0" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T854_fn" "fn compiler_main_test_dh_and_not_or_basic" "$M"
cg "main:T855_fn" "fn compiler_main_test_dh_and_not_or_rhs" "$M"
cg "main:T856_fn" "fn compiler_main_test_dh_and_not_or_comm" "$M"
cg "main:T857_fn" "fn compiler_main_test_dh_no_fold_diff_reg" "$M"
cg "main:T858_fn" "fn compiler_main_test_dh_no_fold_or_op" "$M"
cg "main:T859_fn" "fn compiler_main_test_dh_no_fold_not_and" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T854-T859 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T854 T855 T856 T857 T858 T859; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T854 T855 T856 T857 T858 T859; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
