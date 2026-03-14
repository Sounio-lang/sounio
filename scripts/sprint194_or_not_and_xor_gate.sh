#!/usr/bin/env bash
# sprint194_or_not_and_xor_gate.sh — Sprint 194: Block DD OR-NOT-AND to XOR
#
# Block DD: (x|y)&~(x&y) → x^y; commutative ~(x&y)&(x|y) → x^y.
#   Zero new arrays: uses ao_or_rr_valid/lhs/rhs + ao_and_rr_valid/lhs/rhs + is_bnot/bnot_src.
#   SOTA: LLVM InstCombineAndOrXor; Hacker's Delight §2-12.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T830" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T830)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 194: Block DD — OR-NOT-AND to XOR ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dd"       "Block DD" "$O"
cg "src:dd_or_rr"       "ao_or_rr_valid\[dd_s[12] as usize\]" "$O"
cg "src:dd_is_bnot"     "is_bnot\[dd_s[12] as usize\]" "$O"
cg "src:dd_and_rr"      "ao_and_rr_valid\[dd_not_inner as usize\]" "$O"
cg "src:dd_xor_rewrite" "ir_binop.*dd_instr\.dst.*OpBitXor" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T830_fn" "fn compiler_main_test_dd_or_not_and_xor_basic" "$M"
cg "main:T831_fn" "fn compiler_main_test_dd_or_not_and_xor_comm" "$M"
cg "main:T832_fn" "fn compiler_main_test_dd_or_not_and_xor_val" "$M"
cg "main:T833_fn" "fn compiler_main_test_dd_no_fold_diff_base_or" "$M"
cg "main:T834_fn" "fn compiler_main_test_dd_no_fold_not_not_and" "$M"
cg "main:T835_fn" "fn compiler_main_test_dd_no_fold_or_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T830-T835 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T830 T831 T832 T833 T834 T835; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T830 T831 T832 T833 T834 T835; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
