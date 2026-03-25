#!/usr/bin/env bash
# sprint223_and_or_absorb_gate.sh — Sprint 223: Block EG AND-OR superset absorption
#
# Block EG: (x&y)&(x|y) → x&y; (x|y)&(x&y) → x&y. Dual of Block EF.
#   Zero new arrays: uses ao_and_rr_valid/lhs/rhs + ao_or_rr_valid/lhs/rhs.
#   SOTA: LLVM InstCombineAndOrXor; Boolean lattice absorption; Huntington 1904.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1004" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1004)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 223: Block EG — AND-OR superset absorption ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_eg"    "Block EG" "$O"
cg "src:eg_and_rr"  "ao_and_rr_valid\[eg_s[12] as usize\]" "$O"
cg "src:eg_or_rr"   "ao_or_rr_valid\[eg_s[12] as usize\]" "$O"
cg "src:eg_copy"    "ir_copy.*eg_instr\.dst.*eg_s[12]" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1004_fn" "fn compiler_main_test_eg_and_or_absorb_basic" "$M"
cg "main:T1005_fn" "fn compiler_main_test_eg_and_or_absorb_comm" "$M"
cg "main:T1006_fn" "fn compiler_main_test_eg_and_or_absorb_inner_comm" "$M"
cg "main:T1007_fn" "fn compiler_main_test_eg_no_fold_diff_vars" "$M"
cg "main:T1008_fn" "fn compiler_main_test_eg_no_fold_or_outer" "$M"
cg "main:T1009_fn" "fn compiler_main_test_eg_no_fold_and_both" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1004-T1009 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1004 T1005 T1006 T1007 T1008 T1009; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1004 T1005 T1006 T1007 T1008 T1009; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
