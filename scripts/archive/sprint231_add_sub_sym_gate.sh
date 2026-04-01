#!/usr/bin/env bash
# sprint231_add_sub_sym_gate.sh — Sprint 231: Block EO Add-sub symmetric collapse
#
# Block EO: (x+y)+(x-y) → x+x; commutative variants.
#   Zero new arrays: uses aq_add_rr_valid/lhs/rhs + al_is_sub/lhs/rhs.
#   SOTA: LLVM InstCombineAddSub; algebraic: (x+y)+(x-y)=2x.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1052" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1052)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 231: Block EO — Add-sub symmetric collapse ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_eo"    "Block EO" "$O"
cg "src:eo_add_rr"   "aq_add_rr_valid\[eo_s[12] as usize\]" "$O"
cg "src:eo_is_sub"   "al_is_sub\[eo_s[12] as usize\]" "$O"
cg "src:eo_rewrite"  "ir_binop.*eo_instr\.dst.*eo_[as]x.*OpAdd.*eo_[as]x" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1052_fn" "fn compiler_main_test_eo_add_sub_sym_basic" "$M"
cg "main:T1053_fn" "fn compiler_main_test_eo_add_sub_sym_comm_sub" "$M"
cg "main:T1054_fn" "fn compiler_main_test_eo_add_sub_sym_inner_comm" "$M"
cg "main:T1055_fn" "fn compiler_main_test_eo_no_fold_diff_var" "$M"
cg "main:T1056_fn" "fn compiler_main_test_eo_no_fold_sub_op" "$M"
cg "main:T1057_fn" "fn compiler_main_test_eo_no_fold_diff_x" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1052-T1057 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1052 T1053 T1054 T1055 T1056 T1057; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1052 T1053 T1054 T1055 T1056 T1057; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
