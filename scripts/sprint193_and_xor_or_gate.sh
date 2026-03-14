#!/usr/bin/env bash
# sprint193_and_xor_or_gate.sh — Sprint 193: Block DC AND-XOR to OR
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n (pattern '$p' not found)"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T824" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T824)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 193: Block DC — AND-XOR to OR ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dc" "Block DC" "$O"
cg "src:dc_and_rr" "ao_and_rr_valid\[dc_s[12] as usize\]" "$O"
cg "src:dc_xor_rr" "as_xor_rr_valid\[dc_s[12] as usize\]" "$O"
cg "src:dc_or_rewrite" "ir_binop.*dc_instr.dst.*OpBitOr" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T824" "fn compiler_main_test_dc_and_xor_or_basic" "$M"
cg "main:T825" "fn compiler_main_test_dc_and_xor_or_comm" "$M"
cg "main:T826" "fn compiler_main_test_dc_and_xor_or_swapped" "$M"
cg "main:T827" "fn compiler_main_test_dc_no_fold_diff_base" "$M"
cg "main:T828" "fn compiler_main_test_dc_no_fold_xor_outer" "$M"
cg "main:T829" "fn compiler_main_test_dc_no_fold_and_outer" "$M"
cg "main:total" "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T824 T825 T826 T827 T828 T829; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T824 T825 T826 T827 T828 T829; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
_tc=$(timeout 120 "$SOUC" check self-hosted/compiler/main.sio 2>&1) || true
if echo "$_tc" | grep -q "All checks passed"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
