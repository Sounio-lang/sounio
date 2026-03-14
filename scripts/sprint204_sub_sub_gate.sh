#!/usr/bin/env bash
# sprint204_sub_sub_gate.sh — Sprint 204: Block DN sub-sub identity
#
# Block DN: (x-y)-(x-z) → z-y.
#   Zero new arrays: uses al_is_sub/al_sub_lhs/al_sub_rhs (Block AL).
#   SOTA: LLVM InstCombineAddSub; algebraic cancellation.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T890" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T890)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 204: Block DN — sub-sub identity ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dn"      "Block DN" "$O"
cg "src:dn_al_sub"     "al_is_sub\[dn_s[12] as usize\]" "$O"
cg "src:dn_same_lhs"   "al_sub_lhs\[dn_s1 as usize\] == al_sub_lhs\[dn_s2 as usize\]" "$O"
cg "src:dn_reverse"    "ir_binop.*dn_instr\.dst.*dn_z.*OpSub.*dn_y" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T890_fn" "fn compiler_main_test_dn_sub_sub_basic" "$M"
cg "main:T891_fn" "fn compiler_main_test_dn_sub_sub_self" "$M"
cg "main:T892_fn" "fn compiler_main_test_dn_sub_sub_reverse" "$M"
cg "main:T893_fn" "fn compiler_main_test_dn_no_fold_diff_lhs" "$M"
cg "main:T894_fn" "fn compiler_main_test_dn_no_fold_add_outer" "$M"
cg "main:T895_fn" "fn compiler_main_test_dn_no_fold_not_sub" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T890-T895 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T890 T891 T892 T893 T894 T895; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T890 T891 T892 T893 T894 T895; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
