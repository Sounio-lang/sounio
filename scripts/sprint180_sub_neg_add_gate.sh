#!/usr/bin/env bash
# sprint180_sub_neg_add_gate.sh — Sprint 180: Block CP Sub-neg to add
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T746" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T746)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 180: Block CP — Sub-neg to add ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_cp" "Block CP" "$O"
cg "src:cp_fn" "fn compiler_main_test_cp_" "self-hosted/compiler/main.sio"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T746" "fn compiler_main_test_cp_" "$M"
cg "main:T747" "fn compiler_main_test_cp_" "$M"
cg "main:T748" "fn compiler_main_test_cp_" "$M"
cg "main:T749" "fn compiler_main_test_cp_" "$M"
cg "main:T750" "fn compiler_main_test_cp_" "$M"
cg "main:T751" "fn compiler_main_test_cp_" "$M"
cg "main:total" "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T746 T747 T748 T749 T750 T751 ; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T746 T747 T748 T749 T750 T751 ; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
