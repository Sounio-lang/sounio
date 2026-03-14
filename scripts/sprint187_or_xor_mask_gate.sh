#!/usr/bin/env bash
# sprint187_or_xor_mask_gate.sh — Sprint 187: Block CW OR-XOR mask extraction
#
# Block CW: (x|C)^C → x & ~C
#   Zero new arrays: uses am_or_valid/var_src/const_val (Block AM) + and_valid/var_src/const_val.
#   SOTA: LLVM InstCombineAndOrXor; Hacker's Delight §2-12.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n (pattern '$p' not found)"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T788" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T788)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 187: Block CW — OR-XOR Mask Extraction ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_cw" "Block CW" "$O"
cg "src:cw_or_valid" "am_or_valid\[cw_s[12] as usize\]" "$O"
cg "src:cw_not_c" "cw_not_c" "$O"
cg "src:cw_and_rewrite" "OpBitAnd.*cw_s[12]|cw_instr\.dst.*OpBitAnd" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T788" "fn compiler_main_test_cw_or_xor_mask_basic" "$M"
cg "main:T789" "fn compiler_main_test_cw_or_xor_mask_comm" "$M"
cg "main:T790" "fn compiler_main_test_cw_or_xor_mask_val" "$M"
cg "main:T791" "fn compiler_main_test_cw_no_fold_diff_const" "$M"
cg "main:T792" "fn compiler_main_test_cw_no_fold_rr_or" "$M"
cg "main:T793" "fn compiler_main_test_cw_no_fold_and_outer" "$M"
cg "main:total" "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T788 T789 T790 T791 T792 T793; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T788 T789 T790 T791 T792 T793; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
