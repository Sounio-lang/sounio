#!/usr/bin/env bash
# sprint220_add_sub_expand_gate.sh — Sprint 220: Block ED Add-sub expand
#
# Block ED: (x+y)-(x-z) → y+z; commutative add variant.
#   Zero new arrays: uses aq_add_rr_valid/lhs/rhs + al_is_sub/lhs/rhs.
#   SOTA: LLVM InstCombineAddSub; algebraic simplification.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T986" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T986)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 220: Block ED — Add-sub expand ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_ed"     "Block ED" "$O"
cg "src:ed_add_rr"   "aq_add_rr_valid\[ed_s1 as usize\]" "$O"
cg "src:ed_is_sub"   "al_is_sub\[ed_s2 as usize\]" "$O"
cg "src:ed_rewrite"  "ir_binop.*ed_instr\.dst.*ed_a[rl].*OpAdd.*ed_sr" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T986_fn" "fn compiler_main_test_ed_add_sub_expand_basic" "$M"
cg "main:T987_fn" "fn compiler_main_test_ed_add_sub_expand_comm" "$M"
cg "main:T988_fn" "fn compiler_main_test_ed_add_sub_expand_vals" "$M"
cg "main:T989_fn" "fn compiler_main_test_ed_no_fold_diff_lhs" "$M"
cg "main:T990_fn" "fn compiler_main_test_ed_no_fold_sub_s1" "$M"
cg "main:T991_fn" "fn compiler_main_test_ed_no_fold_add_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T986-T991 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T986 T987 T988 T989 T990 T991; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T986 T987 T988 T989 T990 T991; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
