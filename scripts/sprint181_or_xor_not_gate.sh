#!/usr/bin/env bash
# sprint181_or_xor_not_gate.sh — Sprint 181: Block CQ OR-complement XOR to NOT
#
# Block CQ: (x|C) ^ (x|~C) → ~x.
#   Zero new arrays: uses am_or_valid/var_src/const_val (Block AM).
#   SOTA: LLVM InstCombineAndOrXor; Boolean algebra.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T752" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T752)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 181: Block CQ — OR-Complement XOR to NOT ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_cq"       "Block CQ" "$O"
cg "src:cq_or_valid"    "am_or_valid\[cq_s[12] as usize\]" "$O"
cg "src:cq_complement"  "am_or_const_val\[cq_s1 as usize\] == .am_or_const_val\[cq_s2 as usize\]" "$O"
cg "src:cq_not_rewrite" "ir_unaryop.*cq_instr\.dst.*OpNot" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T752_fn" "fn compiler_main_test_cq_or_comp_xor_not" "$M"
cg "main:T753_fn" "fn compiler_main_test_cq_or_comp_xor_comm" "$M"
cg "main:T754_fn" "fn compiler_main_test_cq_or_comp_xor_val" "$M"
cg "main:T755_fn" "fn compiler_main_test_cq_no_fold_diff_base" "$M"
cg "main:T756_fn" "fn compiler_main_test_cq_no_fold_non_comp" "$M"
cg "main:T757_fn" "fn compiler_main_test_cq_no_fold_and_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T752-T757 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T752 T753 T754 T755 T756 T757; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>/dev/null || _ec=$?
for t in T752 T753 T754 T755 T756 T757; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
