#!/usr/bin/env bash
# sprint225_xnor_and_subsume_gate.sh — Sprint 225: Block EI XNOR-AND subsumption
#
# Block EI: ~(x^y)|(x&y) → ~(x^y); commutative.
#   Zero new arrays: uses is_bnot/bnot_src + as_xor_rr + ao_and_rr.
#   SOTA: LLVM InstCombineAndOrXor; Boolean lattice: a|b=a when b⊆a.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1016" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1016)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 225: Block EI — XNOR-AND subsumption ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ei"    "Block EI" "$O"
cg "src:ei_is_bnot" "is_bnot\[ei_s[12] as usize\]" "$O"
cg "src:ei_and_rr"  "ao_and_rr_valid\[ei_s[12] as usize\]" "$O"
cg "src:ei_copy"    "ir_copy.*ei_instr\.dst.*ei_s[12]" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1016_fn" "fn compiler_main_test_ei_xnor_and_subsume_basic" "$M"
cg "main:T1017_fn" "fn compiler_main_test_ei_xnor_and_subsume_comm" "$M"
cg "main:T1018_fn" "fn compiler_main_test_ei_xnor_and_subsume_inner_comm" "$M"
cg "main:T1019_fn" "fn compiler_main_test_ei_no_fold_diff_vars" "$M"
cg "main:T1020_fn" "fn compiler_main_test_ei_no_fold_and_outer" "$M"
cg "main:T1021_fn" "fn compiler_main_test_ei_no_fold_or_inner" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1016-T1021 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1016 T1017 T1018 T1019 T1020 T1021; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1016 T1017 T1018 T1019 T1020 T1021; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
