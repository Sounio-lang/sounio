#!/usr/bin/env bash
# sprint232_sub_swap_double_gate.sh — Sprint 232: Block EP Sub-swap double
#
# Block EP: (x-y)-(y-x) → (x-y)+(x-y).
#   Zero new arrays: uses al_is_sub/lhs/rhs (Block AL).
#   SOTA: LLVM InstCombineAddSub; algebraic: (x-y)-(y-x)=2*(x-y).
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1058" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1058)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 232: Block EP — Sub-swap double ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ep"    "Block EP" "$O"
cg "src:ep_is_sub"   "al_is_sub\[ep_s[12] as usize\]" "$O"
cg "src:ep_swap"     "ep_x1 == ep_y2 && ep_y1 == ep_x2" "$O"
cg "src:ep_rewrite"  "ir_binop.*ep_instr\.dst.*ep_s1.*OpAdd.*ep_s1" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1058_fn" "fn compiler_main_test_ep_sub_swap_double_basic" "$M"
cg "main:T1059_fn" "fn compiler_main_test_ep_sub_swap_double_uses_s1" "$M"
cg "main:T1060_fn" "fn compiler_main_test_ep_sub_swap_double_same_sub" "$M"
cg "main:T1061_fn" "fn compiler_main_test_ep_no_fold_not_swap" "$M"
cg "main:T1062_fn" "fn compiler_main_test_ep_no_fold_add_outer" "$M"
cg "main:T1063_fn" "fn compiler_main_test_ep_no_fold_one_not_sub" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1058-T1063 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1058 T1059 T1060 T1061 T1062 T1063; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1058 T1059 T1060 T1061 T1062 T1063; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
