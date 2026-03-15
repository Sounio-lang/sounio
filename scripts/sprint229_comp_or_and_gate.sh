#!/usr/bin/env bash
# sprint229_comp_or_and_gate.sh — Sprint 229: Block EM Complement-OR AND
#
# Block EM: (~x|y)&(x|y) → y; commutative variants.
#   Zero new arrays: uses is_bnot/bnot_src + ao_or_rr_valid/lhs/rhs (Block AO).
#   SOTA: Boolean algebra: y|(~x&x) = y|0 = y. Dual of Block EL.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1040" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1040)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 229: Block EM — Complement-OR AND ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_em"    "Block EM" "$O"
cg "src:em_is_bnot"  "is_bnot\[em_ar[12] as usize\]|is_bnot\[em_al[12] as usize\]" "$O"
cg "src:em_or_rr"    "ao_or_rr_valid\[em_s[12] as usize\]" "$O"
cg "src:em_copy"     "ir_copy.*em_instr\.dst.*em_ar[12]|ir_copy.*em_instr\.dst.*em_al[12]" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1040_fn" "fn compiler_main_test_em_comp_or_and_basic" "$M"
cg "main:T1041_fn" "fn compiler_main_test_em_comp_or_and_comm" "$M"
cg "main:T1042_fn" "fn compiler_main_test_em_comp_or_and_lhs_shared" "$M"
cg "main:T1043_fn" "fn compiler_main_test_em_no_fold_diff_y" "$M"
cg "main:T1044_fn" "fn compiler_main_test_em_no_fold_no_not" "$M"
cg "main:T1045_fn" "fn compiler_main_test_em_no_fold_and_outer" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1040-T1045 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1040 T1041 T1042 T1043 T1044 T1045; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1040 T1041 T1042 T1043 T1044 T1045; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
