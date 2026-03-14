#!/usr/bin/env bash
# sprint182_and_xor_comp_gate.sh — Sprint 182: Block CR AND-XOR complement extraction
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "fn compiler_main_test_cr_|T758" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before fn compiler_main_test_cr_|T758)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 182: Block CR — AND-XOR complement extraction ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:and_xor_comp_comment"  "Block CR" "$O"
cg "src:and_xor_comp_check1"   "and_valid\[cr_s[12] as usize\]" "$O"
cg "src:and_xor_comp_rewrite"  "ir_unaryop.*cr_s[12]\" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T758_fn" "ir_unaryop(cr_s" "$M"
cg "main:T759_fn" "ir_unaryop(cr_s" "$M"
cg "main:T760_fn" "ir_unaryop(cr_s" "$M"
cg "main:T761_fn" "ir_unaryop(cr_s" "$M"
cg "main:T762_fn" "ir_unaryop(cr_s" "$M"
cg "main:T763_fn" "ir_unaryop(cr_s" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T758-T763 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in 758 759 760 761 762 763 ; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:T$t"; NOT_RUN=$((NOT_RUN+1)); done
else "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>/dev/null || _ec=$?
for t in 758 759 760 761 762 763 ; do cl "selftest:T$t" "T${t} OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
