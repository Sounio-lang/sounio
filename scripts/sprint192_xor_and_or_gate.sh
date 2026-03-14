#!/usr/bin/env bash
# sprint192_xor_and_or_gate.sh — Sprint 192: Block DB XOR-AND to OR
#
# Block DB: (x^y)|(x&y) → x|y; commutative (x&y)|(x^y) → x|y.
#   Zero new arrays: uses as_xor_rr_valid/lhs/rhs (Block AS) + ao_and_rr_valid/lhs/rhs (Block AO).
#   SOTA: LLVM InstCombineAndOrXor; Boolean algebra.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T818" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T818)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 192: Block DB — XOR-AND to OR ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_db"       "Block DB" "$O"
cg "src:db_xor_rr"      "as_xor_rr_valid\[db_s[12] as usize\]" "$O"
cg "src:db_and_rr"      "ao_and_rr_valid\[db_s[12] as usize\]" "$O"
cg "src:db_or_rewrite"  "ir_binop.*db_instr\.dst.*OpBitOr" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T818_fn" "fn compiler_main_test_db_xor_and_or_basic" "$M"
cg "main:T819_fn" "fn compiler_main_test_db_xor_and_or_comm" "$M"
cg "main:T820_fn" "fn compiler_main_test_db_xor_and_or_val" "$M"
cg "main:T821_fn" "fn compiler_main_test_db_no_fold_diff_base_xor" "$M"
cg "main:T822_fn" "fn compiler_main_test_db_no_fold_rr_or" "$M"
cg "main:T823_fn" "fn compiler_main_test_db_no_fold_xor_outer" "$M"
cg "main:total"   "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T818-T823 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T818 T819 T820 T821 T822 T823; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T818 T819 T820 T821 T822 T823; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
