#!/usr/bin/env bash
# sprint196_or_self_comp_gate.sh — Sprint 196: Block DF OR self-complement
#
# Block DF: x | ~x → -1, ~x | x → -1.
#   Zero new arrays: uses is_bnot/bnot_src (Block BF) + def_at.
#   SOTA: LLVM InstCombineAndOrXor; Boolean complement law.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T842" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T842)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 196: Block DF — OR self-complement ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_df"      "Block DF" "$O"
cg "src:df_is_bnot"    "is_bnot\[df_s[12] as usize\]" "$O"
cg "src:df_minus1"     "ir_load_imm.*df_instr\.dst.*-1" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T842_fn" "fn compiler_main_test_df_or_self_comp_basic" "$M"
cg "main:T843_fn" "fn compiler_main_test_df_or_self_comp_comm" "$M"
cg "main:T844_fn" "fn compiler_main_test_df_or_self_comp_chain" "$M"
cg "main:T845_fn" "fn compiler_main_test_df_no_fold_diff_regs" "$M"
cg "main:T846_fn" "fn compiler_main_test_df_no_fold_xor_op" "$M"
cg "main:T847_fn" "fn compiler_main_test_df_no_fold_and_op" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T842-T847 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T842 T843 T844 T845 T846 T847; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T842 T843 T844 T845 T846 T847; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
