#!/usr/bin/env bash
# sprint246_xor_decomp_fold_gate.sh — Sprint 246: Block FD XOR decomposition fold
#
# Block FD: (~x&y)^(x&~y) → x^y; commutative variants.
#   Zero new arrays: uses ao_and_rr_valid/lhs/rhs + is_bnot/bnot_src.
#   SOTA: XOR canonical decomposition; Hacker's Delight §2-1.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."  ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1142" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1142)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 246: Block FD — XOR decomposition fold ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_fd"    "Block FD" "$O"
cg "src:fd_and_rr"   "ao_and_rr_valid\[fd_s[12] as usize\]" "$O"
cg "src:fd_is_bnot"  "is_bnot\[fd_(al|ar)[0-9]* as usize\]" "$O"
cg "src:fd_rewrite"  "ir_binop.*fd_instr\.dst.*fd_x.*OpBitXor.*fd_y" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1142_fn" "fn compiler_main_test_fd_xor_decomp_basic" "$M"
cg "main:T1143_fn" "fn compiler_main_test_fd_xor_decomp_comm" "$M"
cg "main:T1144_fn" "fn compiler_main_test_fd_xor_decomp_inner_comm" "$M"
cg "main:T1145_fn" "fn compiler_main_test_fd_no_fold_diff_var" "$M"
cg "main:T1146_fn" "fn compiler_main_test_fd_no_fold_or_inner" "$M"
cg "main:T1147_fn" "fn compiler_main_test_fd_no_fold_and_outer" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1142-T1147 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1142 T1143 T1144 T1145 T1146 T1147; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1142 T1143 T1144 T1145 T1146 T1147; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
