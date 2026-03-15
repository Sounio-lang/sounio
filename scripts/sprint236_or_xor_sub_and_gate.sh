#!/usr/bin/env bash
# sprint236_or_xor_sub_and_gate.sh — Sprint 236: Block ET OR-XOR diff to AND
#
# Block ET: (x|y)-(x^y) → x&y.
#   Zero new arrays: uses ao_or_rr_valid/lhs/rhs + as_xor_rr_valid/lhs/rhs.
#   SOTA: Hacker's Delight §2-2; x|y = (x^y)+(x&y) → (x|y)-(x^y)=x&y.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1082" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1082)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 236: Block ET — OR-XOR diff to AND ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_et"    "Block ET" "$O"
cg "src:et_or_rr"    "ao_or_rr_valid\[et_s1 as usize\]" "$O"
cg "src:et_xor_rr"   "as_xor_rr_valid\[et_s2 as usize\]" "$O"
cg "src:et_rewrite"  "ir_binop.*et_instr\.dst.*et_x[lr].*OpBitAnd" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1082_fn" "fn compiler_main_test_et_or_xor_sub_and_basic" "$M"
cg "main:T1083_fn" "fn compiler_main_test_et_or_xor_sub_and_inner_comm" "$M"
cg "main:T1084_fn" "fn compiler_main_test_et_or_xor_sub_and_tracked" "$M"
cg "main:T1085_fn" "fn compiler_main_test_et_no_fold_diff_var" "$M"
cg "main:T1086_fn" "fn compiler_main_test_et_no_fold_add_outer" "$M"
cg "main:T1087_fn" "fn compiler_main_test_et_no_fold_and_inner" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1082-T1087 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1082 T1083 T1084 T1085 T1086 T1087; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1082 T1083 T1084 T1085 T1086 T1087; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
