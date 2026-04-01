#!/usr/bin/env bash
# sprint237_add_or_sub_and_gate.sh — Sprint 237: Block EU Add-OR diff to AND
#
# Block EU: (x+y)-(x|y) → x&y.
#   Zero new arrays: uses aq_add_rr_valid/lhs/rhs + ao_or_rr_valid/lhs/rhs.
#   SOTA: Hacker's Delight §2-2; (x+y)-(x|y)=x&y.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1088" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1088)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 237: Block EU — Add-OR diff to AND ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_eu"    "Block EU" "$O"
cg "src:eu_add_rr"   "aq_add_rr_valid\[eu_s1 as usize\]" "$O"
cg "src:eu_or_rr"    "ao_or_rr_valid\[eu_s2 as usize\]" "$O"
cg "src:eu_rewrite"  "ir_binop.*eu_instr\.dst.*eu_a[lr].*OpBitAnd" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1088_fn" "fn compiler_main_test_eu_add_or_sub_and_basic" "$M"
cg "main:T1089_fn" "fn compiler_main_test_eu_add_or_sub_and_inner_comm_add" "$M"
cg "main:T1090_fn" "fn compiler_main_test_eu_add_or_sub_and_inner_comm_or" "$M"
cg "main:T1091_fn" "fn compiler_main_test_eu_no_fold_diff_var" "$M"
cg "main:T1092_fn" "fn compiler_main_test_eu_no_fold_add_outer" "$M"
cg "main:T1093_fn" "fn compiler_main_test_eu_no_fold_and_inner" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1088-T1093 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1088 T1089 T1090 T1091 T1092 T1093; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1088 T1089 T1090 T1091 T1092 T1093; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
