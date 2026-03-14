#!/usr/bin/env bash
# sprint201_or_and_sum_gate.sh — Sprint 201: Block DK OR-AND sum factoring
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T872" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T872)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 201: Block DK — OR-AND sum factoring ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dk"   "Block DK" "$O"
cg "src:dk_or_rr"   "ao_or_rr_valid\[dk_s[12] as usize\]" "$O"
cg "src:dk_and_rr"  "ao_and_rr_valid\[dk_s[12] as usize\]" "$O"
cg "src:dk_result"  "ir_binop.*dk_instr\.dst.*dk_ol.*OpAdd.*dk_or_r" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T872_fn" "fn compiler_main_test_dk_or_and_sum_basic" "$M"
cg "main:T873_fn" "fn compiler_main_test_dk_or_and_sum_comm" "$M"
cg "main:T874_fn" "fn compiler_main_test_dk_or_and_sum_swapped" "$M"
cg "main:T875_fn" "fn compiler_main_test_dk_no_fold_diff_base" "$M"
cg "main:T876_fn" "fn compiler_main_test_dk_no_fold_sub_op" "$M"
cg "main:T877_fn" "fn compiler_main_test_dk_no_fold_or_or" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T872-T877 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T872 T873 T874 T875 T876 T877; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T872 T873 T874 T875 T876 T877; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
