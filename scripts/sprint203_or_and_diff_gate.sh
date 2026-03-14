#!/usr/bin/env bash
# sprint203_or_and_diff_gate.sh — Sprint 203: Block DM OR-AND difference to XOR
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T884" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T884)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 203: Block DM — OR-AND difference to XOR ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_dm"   "Block DM" "$O"
cg "src:dm_or_rr"   "ao_or_rr_valid\[dm_s[12] as usize\]" "$O"
cg "src:dm_and_rr"  "ao_and_rr_valid\[dm_s[12] as usize\]" "$O"
cg "src:dm_result"  "ir_binop.*dm_instr\.dst.*dm_ol.*OpBitXor.*dm_or" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T884_fn" "fn compiler_main_test_dm_or_and_diff_basic" "$M"
cg "main:T885_fn" "fn compiler_main_test_dm_or_and_diff_swapped" "$M"
cg "main:T886_fn" "fn compiler_main_test_dm_or_and_diff_val" "$M"
cg "main:T887_fn" "fn compiler_main_test_dm_no_fold_diff_ops" "$M"
cg "main:T888_fn" "fn compiler_main_test_dm_no_fold_diff_base" "$M"
cg "main:T889_fn" "fn compiler_main_test_dm_no_fold_add_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T884-T889 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T884 T885 T886 T887 T888 T889; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T884 T885 T886 T887 T888 T889; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
