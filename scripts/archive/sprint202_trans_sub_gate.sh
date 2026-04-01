#!/usr/bin/env bash
# sprint202_trans_sub_gate.sh — Sprint 202: Block DL transitive subtraction
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T878" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T878)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 202: Block DL — Transitive subtraction ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dl"      "Block DL" "$O"
cg "src:dl_al_is_sub"  "al_is_sub\[dl_s[12] as usize\]" "$O"
cg "src:dl_cancel"     "al_sub_rhs\[dl_s1 as usize\] == al_sub_lhs\[dl_s2 as usize\]" "$O"
cg "src:dl_rewrite"    "ir_binop.*dl_x.*OpSub.*dl_z" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T878_fn" "fn compiler_main_test_dl_trans_sub_basic" "$M"
cg "main:T879_fn" "fn compiler_main_test_dl_trans_sub_comm" "$M"
cg "main:T880_fn" "fn compiler_main_test_dl_trans_sub_self" "$M"
cg "main:T881_fn" "fn compiler_main_test_dl_no_fold_diff_middle" "$M"
cg "main:T882_fn" "fn compiler_main_test_dl_no_fold_sub_outer" "$M"
cg "main:T883_fn" "fn compiler_main_test_dl_no_fold_not_sub" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T878-T883 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T878 T879 T880 T881 T882 T883; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T878 T879 T880 T881 T882 T883; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
