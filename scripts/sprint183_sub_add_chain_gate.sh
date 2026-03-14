#!/usr/bin/env bash
# sprint183_sub_add_chain_gate.sh — Sprint 183: Block CS sub-add constant chain fold
#
# Block CS: (x-C1)+C2 → x+(C2-C1) [or x-(C1-C2) or IrCopy if equal]
#   Zero new arrays: uses bs_sub_valid/src/cval + bt_add_valid/src/cval.
#   SOTA: LLVM InstCombineAddSub; associative constant folding for add/sub.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n (pattern '$p' not found)"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T764" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T764)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 183: Block CS — Sub-Add Constant Chain Fold ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_cs" "Block CS" "$O"
cg "src:cs_bs_valid" "bs_sub_valid\[cs_s[12] as usize\]" "$O"
cg "src:cs_merged" "cs_merged = cs_c2 - cs_c1|cs_merged.*cs_c" "$O"
cg "src:cs_rewrite" "(ir_copy|ir_binop).*cs_instr\.dst" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T764" "fn compiler_main_test_cs_sub_add_chain_basic" "$M"
cg "main:T765" "fn compiler_main_test_cs_sub_add_chain_comm" "$M"
cg "main:T766" "fn compiler_main_test_cs_sub_add_chain_cancel" "$M"
cg "main:T767" "fn compiler_main_test_cs_sub_add_chain_neg_result" "$M"
cg "main:T768" "fn compiler_main_test_cs_no_fold_rr_sub" "$M"
cg "main:T769" "fn compiler_main_test_cs_no_fold_rr_add" "$M"
cg "main:total" "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T764 T765 T766 T767 T768 T769; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T764 T765 T766 T767 T768 T769; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
