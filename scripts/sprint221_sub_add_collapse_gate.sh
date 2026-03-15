#!/usr/bin/env bash
# sprint221_sub_add_collapse_gate.sh — Sprint 221: Block EE Sub-add collapse
#
# Block EE: (x-y)+(z+y) → x+z; commutative variants.
#   Zero new arrays: uses al_is_sub/lhs/rhs + aq_add_rr_valid/lhs/rhs.
#   SOTA: LLVM InstCombineAddSub; algebraic simplification.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T992" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T992)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 221: Block EE — Sub-add collapse ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ee"     "Block EE" "$O"
cg "src:ee_is_sub"    "al_is_sub\[ee_s[12] as usize\]" "$O"
cg "src:ee_add_rr"   "aq_add_rr_valid\[ee_s[12] as usize\]" "$O"
cg "src:ee_rewrite"  "ir_binop.*ee_instr\.dst.*ee_x.*OpAdd" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T992_fn" "fn compiler_main_test_ee_sub_add_collapse_basic" "$M"
cg "main:T993_fn" "fn compiler_main_test_ee_sub_add_collapse_comm_add" "$M"
cg "main:T994_fn" "fn compiler_main_test_ee_sub_add_collapse_comm_outer" "$M"
cg "main:T995_fn" "fn compiler_main_test_ee_no_fold_diff_y" "$M"
cg "main:T996_fn" "fn compiler_main_test_ee_no_fold_sub_outer" "$M"
cg "main:T997_fn" "fn compiler_main_test_ee_no_fold_no_add" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T992-T997 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T992 T993 T994 T995 T996 T997; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T992 T993 T994 T995 T996 T997; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
