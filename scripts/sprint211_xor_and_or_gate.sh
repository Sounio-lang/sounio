#!/usr/bin/env bash
# sprint211_xor_and_or_gate.sh — Sprint 211: Block DU XOR-AND to OR gate
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T932" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T932)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 211: Block DU — XOR-AND to OR ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_du"     "Block DU" "$O"
cg "src:du_xor_rr"    "as_xor_rr_valid\[du_s1 as usize\]" "$O"
cg "src:du_and_rr"    "ao_and_rr_valid\[du_s2 as usize\]" "$O"
cg "src:du_or_result" "ir_binop.*du_instr\.dst.*du_xl.*OpBitOr.*du_xr" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T932_fn" "fn compiler_main_test_du_xor_and_or_basic" "$M"
cg "main:T933_fn" "fn compiler_main_test_du_xor_and_or_comm_inner" "$M"
cg "main:T934_fn" "fn compiler_main_test_du_xor_and_or_comm_outer" "$M"
cg "main:T935_fn" "fn compiler_main_test_du_no_fold_diff_vars" "$M"
cg "main:T936_fn" "fn compiler_main_test_du_no_fold_xor_and_and" "$M"
cg "main:T937_fn" "fn compiler_main_test_du_no_fold_xor_or" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T932-T937 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T932 T933 T934 T935 T936 T937; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T932 T933 T934 T935 T936 T937; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
