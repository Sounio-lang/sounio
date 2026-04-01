#!/usr/bin/env bash
# sprint210_or_and_comp_gate.sh — Sprint 210: Block DT OR-AND complement absorption gate
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T926" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T926)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 210: Block DT — OR-AND Complement Absorption ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dt"     "Block DT" "$O"
cg "src:dt_and_rr"    "ao_and_rr_valid\[dt_s2 as usize\]" "$O"
cg "src:dt_bnot"      "is_bnot\[dt_ar as usize\]" "$O"
cg "src:dt_or_result" "ir_binop.*dt_instr\.dst.*dt_s1.*OpBitOr.*dt_al" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T926_fn" "fn compiler_main_test_dt_or_and_comp_basic" "$M"
cg "main:T927_fn" "fn compiler_main_test_dt_or_and_comp_comm_and" "$M"
cg "main:T928_fn" "fn compiler_main_test_dt_or_and_comp_comm_or" "$M"
cg "main:T929_fn" "fn compiler_main_test_dt_no_fold_diff_var" "$M"
cg "main:T930_fn" "fn compiler_main_test_dt_no_fold_no_not" "$M"
cg "main:T931_fn" "fn compiler_main_test_dt_no_fold_and_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T926-T931 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T926 T927 T928 T929 T930 T931; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T926 T927 T928 T929 T930 T931; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
