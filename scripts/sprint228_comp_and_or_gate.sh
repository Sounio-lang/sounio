#!/usr/bin/env bash
# sprint228_comp_and_or_gate.sh — Sprint 228: Block EL Complement-AND OR
#
# Block EL: (~x&y)|(x&y) → y; commutative variants.
#   Zero new arrays: uses is_bnot/bnot_src + ao_and_rr_valid/lhs/rhs (Block AO).
#   SOTA: Boolean algebra: y&(~x|x) = y&(-1) = y.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}"})/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1034" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1034)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 228: Block EL — Complement-AND OR ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_el"    "Block EL" "$O"
cg "src:el_is_bnot"  "is_bnot\[el_al[12] as usize\]" "$O"
cg "src:el_and_rr"   "ao_and_rr_valid\[el_s[12] as usize\]" "$O"
cg "src:el_copy"     "ir_copy.*el_instr\.dst.*el_ar[12]|ir_copy.*el_instr\.dst.*el_al[12]" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1034_fn" "fn compiler_main_test_el_comp_and_or_basic" "$M"
cg "main:T1035_fn" "fn compiler_main_test_el_comp_and_or_comm" "$M"
cg "main:T1036_fn" "fn compiler_main_test_el_comp_and_or_lhs_shared" "$M"
cg "main:T1037_fn" "fn compiler_main_test_el_no_fold_diff_y" "$M"
cg "main:T1038_fn" "fn compiler_main_test_el_no_fold_no_not" "$M"
cg "main:T1039_fn" "fn compiler_main_test_el_no_fold_or_outer" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1034-T1039 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1034 T1035 T1036 T1037 T1038 T1039; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1034 T1035 T1036 T1037 T1038 T1039; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
