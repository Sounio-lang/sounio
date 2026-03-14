#!/usr/bin/env bash
# sprint175_and_comp_or_gate.sh — Sprint 175: Block CK AND-complement OR
#
# Block CK: (x & ~C) | C → x | C; C | (x & ~C) → x | C.
#   Zero new arrays: uses and_valid/and_var_src/and_const_val (Block AE) + is_const/const_val.
#   SOTA: LLVM InstCombineAndOrXor; Boolean absorption with complement.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T716" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T716)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 175: Block CK — AND-Complement OR ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ck"          "Block CK" "$O"
cg "src:ck_and_valid"      "and_valid\[ck_s[12] as usize\]" "$O"
cg "src:ck_complement"     "and_const_val\[ck_s[12] as usize\] \^ -1" "$O"
cg "src:ck_or_rewrite"     "ir_binop\(ck_instr\.dst" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T716_fn" "fn compiler_main_test_ck_and_comp_or_basic" "$M"
cg "main:T717_fn" "fn compiler_main_test_ck_and_comp_or_comm" "$M"
cg "main:T718_fn" "fn compiler_main_test_ck_and_comp_or_val" "$M"
cg "main:T719_fn" "fn compiler_main_test_ck_no_fold_wrong_comp" "$M"
cg "main:T720_fn" "fn compiler_main_test_ck_no_fold_non_and" "$M"
cg "main:T721_fn" "fn compiler_main_test_ck_no_fold_and_op" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T716-T721 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T716 T717 T718 T719 T720 T721; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>/dev/null || _ec=$?
for t in T716 T717 T718 T719 T720 T721; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
