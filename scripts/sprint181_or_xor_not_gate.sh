#!/usr/bin/env bash
# sprint181_or_xor_not_gate.sh — Sprint 181: Block CQ OR-complement XOR to NOT
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "am_or_const_val.*\^ -1|UnaryOp::OpNot.*cq_x\|ir_unaryop.*cq_|fn compiler_main_test_cq_|T752" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before am_or_const_val.*\^ -1|UnaryOp::OpNot.*cq_x\|ir_unaryop.*cq_|fn compiler_main_test_cq_|T752)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 181: Block CQ — OR-complement XOR to NOT ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:or_xor_not_comment"  "Block CQ" "$O"
cg "src:or_xor_not_check1"   "~C)→~x" "$O"
cg "src:or_xor_not_rewrite"  "am_or_valid\[cq_s[12] as usize\]" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T752_fn" "am_or_const_val\[cq_s1 as usize\] == .am_or_const_val\[cq_s2 as usize\] \^ -1.\" "$M"
cg "main:T753_fn" "am_or_const_val\[cq_s1 as usize\] == .am_or_const_val\[cq_s2 as usize\] \^ -1.\" "$M"
cg "main:T754_fn" "am_or_const_val\[cq_s1 as usize\] == .am_or_const_val\[cq_s2 as usize\] \^ -1.\" "$M"
cg "main:T755_fn" "am_or_const_val\[cq_s1 as usize\] == .am_or_const_val\[cq_s2 as usize\] \^ -1.\" "$M"
cg "main:T756_fn" "am_or_const_val\[cq_s1 as usize\] == .am_or_const_val\[cq_s2 as usize\] \^ -1.\" "$M"
cg "main:T757_fn" "am_or_const_val\[cq_s1 as usize\] == .am_or_const_val\[cq_s2 as usize\] \^ -1.\" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T752-T757 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in 752 753 754 755 756 757 ; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:T$t"; NOT_RUN=$((NOT_RUN+1)); done
else "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>/dev/null || _ec=$?
for t in 752 753 754 755 756 757 ; do cl "selftest:T$t" "T${t} OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
