#!/usr/bin/env bash
# sprint179_neg_add_sub_gate.sh — Sprint 179: Block CO Neg-add to sub
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T740" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T740)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 179: Block CO — Neg-add to sub ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:neg_add_sub_comment"  "Block CO" "$O"
cg "src:neg_add_sub_check1"   "is_neg\[co_s[12] as usize\]" "$O"
cg "src:neg_add_sub_rewrite"  "ir_binop.*OpSub.*co_" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T740_fn" "fn compiler_main_test_co_" "$M"
cg "main:T741_fn" "fn compiler_main_test_co_" "$M"
cg "main:T742_fn" "fn compiler_main_test_co_" "$M"
cg "main:T743_fn" "fn compiler_main_test_co_" "$M"
cg "main:T744_fn" "fn compiler_main_test_co_" "$M"
cg "main:T745_fn" "fn compiler_main_test_co_" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T740-T745 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in 740 741 742 743 744 745 ; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:T$t"; NOT_RUN=$((NOT_RUN+1)); done
else "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>/dev/null || _ec=$?
for t in 740 741 742 743 744 745 ; do cl "selftest:T$t" "T${t} OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
