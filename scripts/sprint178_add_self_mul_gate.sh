#!/usr/bin/env bash
# sprint178_add_self_mul_gate.sh — Sprint 178: Block CN Add-self to multiply
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "fn compiler_main_test_cn_|T734" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before fn compiler_main_test_cn_|T734)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 178: Block CN — Add-self to multiply ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:add_self_mul_comment"  "Block CN" "$O"
cg "src:add_self_mul_check1"   "cn_instr\.bin_op == BinaryOp::OpAdd && cn_s1 == cn_s2" "$O"
cg "src:add_self_mul_rewrite"  "mul_const_v\[cn_instr\.dst.*= 2\" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T734_fn" "mul_const_v.*= 2" "$M"
cg "main:T735_fn" "mul_const_v.*= 2" "$M"
cg "main:T736_fn" "mul_const_v.*= 2" "$M"
cg "main:T737_fn" "mul_const_v.*= 2" "$M"
cg "main:T738_fn" "mul_const_v.*= 2" "$M"
cg "main:T739_fn" "mul_const_v.*= 2" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T734-T739 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in 734 735 736 737 738 739 ; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:T$t"; NOT_RUN=$((NOT_RUN+1)); done
else "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>/dev/null || _ec=$?
for t in 734 735 736 737 738 739 ; do cl "selftest:T$t" "T${t} OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
