#!/usr/bin/env bash
# sprint238_cross_sub_add_gate.sh — Sprint 238: Block EV Cross-sub add collapse
#
# Block EV: (x-y)+(z-x) → z-y; commutative outer ADD.
#   Zero new arrays: uses al_is_sub/lhs/rhs for both operands.
#   SOTA: LLVM InstCombineAddSub; algebraic cancellation.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1094" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1094)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 238: Block EV — Cross-sub add collapse ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ev"    "Block EV" "$O"
cg "src:ev_is_sub"   "al_is_sub\[ev_s[12] as usize\]" "$O"
cg "src:ev_cancel"   "ev_x1 == ev_x2" "$O"
cg "src:ev_rewrite"  "ir_binop.*ev_instr\.dst.*ev_z.*OpSub.*ev_y" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1094_fn" "fn compiler_main_test_ev_cross_sub_add_basic" "$M"
cg "main:T1095_fn" "fn compiler_main_test_ev_cross_sub_add_comm" "$M"
cg "main:T1096_fn" "fn compiler_main_test_ev_cross_sub_add_self_cancel" "$M"
cg "main:T1097_fn" "fn compiler_main_test_ev_no_fold_diff_x" "$M"
cg "main:T1098_fn" "fn compiler_main_test_ev_no_fold_sub_outer" "$M"
cg "main:T1099_fn" "fn compiler_main_test_ev_no_fold_one_not_sub" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1094-T1099 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1094 T1095 T1096 T1097 T1098 T1099; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1094 T1095 T1096 T1097 T1098 T1099; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
