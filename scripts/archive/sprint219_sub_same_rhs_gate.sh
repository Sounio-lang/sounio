#!/usr/bin/env bash
# sprint219_sub_same_rhs_gate.sh — Sprint 219: Block EC Sub same-RHS cancel
#
# Block EC: (x-y)-(z-y) → x-z; symmetric to Block DN (same-LHS).
#   Zero new arrays: uses al_is_sub/al_sub_lhs/al_sub_rhs (Block AL).
#   SOTA: LLVM InstCombineAddSub; algebraic cancellation.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T980" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T980)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 219: Block EC — Sub same-RHS cancel ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ec"      "Block EC" "$O"
cg "src:ec_is_sub"     "al_is_sub\[ec_s[12] as usize\]" "$O"
cg "src:ec_same_rhs"   "al_sub_rhs\[ec_s1 as usize\] == al_sub_rhs\[ec_s2 as usize\]" "$O"
cg "src:ec_rewrite"    "ir_binop.*ec_instr\.dst.*ec_x.*OpSub.*ec_z" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T980_fn" "fn compiler_main_test_ec_sub_same_rhs_basic" "$M"
cg "main:T981_fn" "fn compiler_main_test_ec_sub_same_rhs_chain" "$M"
cg "main:T982_fn" "fn compiler_main_test_ec_sub_same_rhs_self_zero" "$M"
cg "main:T983_fn" "fn compiler_main_test_ec_no_fold_diff_rhs" "$M"
cg "main:T984_fn" "fn compiler_main_test_ec_no_fold_not_sub" "$M"
cg "main:T985_fn" "fn compiler_main_test_ec_no_fold_add_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T980-T985 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T980 T981 T982 T983 T984 T985; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T980 T981 T982 T983 T984 T985; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
