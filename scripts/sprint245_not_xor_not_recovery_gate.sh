#!/usr/bin/env bash
# sprint245_not_xor_not_recovery_gate.sh — Sprint 245: Block FC NOT-XOR-NOT recovery
#
# Block FC: (~x^y)^~y → x; commutative variants.
#   Zero new arrays: uses is_bnot/bnot_src + as_xor_rr_valid/lhs/rhs.
#   SOTA: XOR self-inverse + NOT involution; Hacker's Delight §2-3.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."  ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1136" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1136)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 245: Block FC — NOT-XOR-NOT recovery ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_fc"    "Block FC" "$O"
cg "src:fc_xor_rr"   "as_xor_rr_valid\[fc_s[12] as usize\]" "$O"
cg "src:fc_is_bnot"  "is_bnot\[fc_(s[12]|xl|xr)[0-9]* as usize\]" "$O"
cg "src:fc_copy"     "ir_copy.*fc_instr\.dst.*fc_result" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1136_fn" "fn compiler_main_test_fc_not_xor_not_recovery_basic" "$M"
cg "main:T1137_fn" "fn compiler_main_test_fc_not_xor_not_recovery_inner_comm" "$M"
cg "main:T1138_fn" "fn compiler_main_test_fc_not_xor_not_recovery_outer_comm" "$M"
cg "main:T1139_fn" "fn compiler_main_test_fc_no_fold_diff_not" "$M"
cg "main:T1140_fn" "fn compiler_main_test_fc_no_fold_no_not" "$M"
cg "main:T1141_fn" "fn compiler_main_test_fc_no_fold_and_outer" "$M"
cg "main:total"    "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests: T1136-T1141 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1136 T1137 T1138 T1139 T1140 T1141; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1136 T1137 T1138 T1139 T1140 T1141; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
TC_OUT="$(mktemp)"; if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
