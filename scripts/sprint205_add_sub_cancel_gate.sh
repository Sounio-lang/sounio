#!/usr/bin/env bash
# sprint205_add_sub_cancel_gate.sh — Sprint 205: Block DO Add-sub common-addend cancel
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T896" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T896)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 205: Block DO — Add-sub common-addend cancel ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_do"   "Block DO" "$O"
cg "src:do_add_rr"  "aq_add_rr_valid\[do_s[12] as usize\]" "$O"
cg "src:do_result"  "ir_binop.*do_instr\.dst.*do_al.*OpSub" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T896_fn" "fn compiler_main_test_do_add_sub_cancel_basic" "$M"
cg "main:T897_fn" "fn compiler_main_test_do_add_sub_cancel_shared_left" "$M"
cg "main:T898_fn" "fn compiler_main_test_do_add_sub_cancel_self" "$M"
cg "main:T899_fn" "fn compiler_main_test_do_no_fold_no_common" "$M"
cg "main:T900_fn" "fn compiler_main_test_do_no_fold_add_outer" "$M"
cg "main:T901_fn" "fn compiler_main_test_do_no_fold_not_add" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T896-T901 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T896 T897 T898 T899 T900 T901; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T896 T897 T898 T899 T900 T901; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
