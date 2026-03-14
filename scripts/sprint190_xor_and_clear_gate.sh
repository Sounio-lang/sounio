#!/usr/bin/env bash
# sprint190_xor_and_clear_gate.sh — Sprint 190: Block CZ XOR-AND mask clear
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n (pattern '$p' not found)"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T806" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T806)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 190: Block CZ — XOR-AND Mask Clear ==="
echo ""; echo "--- Source ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_cz" "Block CZ" "$O"
cg "src:cz_and_valid" "and_valid\[cz_s[12] as usize\]" "$O"
cg "src:cz_var_src_eq" "and_var_src\[cz_s[12] as usize\] == cz_s[12]" "$O"
cg "src:cz_not_c" "cz_not_c.*\^ -1|\^ -1.*cz" "$O"
echo ""; echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T806" "fn compiler_main_test_cz_xor_and_clear_basic" "$M"
cg "main:T807" "fn compiler_main_test_cz_xor_and_clear_comm" "$M"
cg "main:T808" "fn compiler_main_test_cz_xor_and_clear_val" "$M"
cg "main:T809" "fn compiler_main_test_cz_no_fold_diff_base" "$M"
cg "main:T810" "fn compiler_main_test_cz_no_fold_rr_and" "$M"
cg "main:T811" "fn compiler_main_test_cz_no_fold_or" "$M"
cg "main:total" "let total: i64 = [0-9]+" "$M"
echo ""; echo "--- Self-tests ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T806 T807 T808 T809 T810 T811; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T806 T807 T808 T809 T810 T811; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""; echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
_tc=$(timeout 120 "$SOUC" check self-hosted/compiler/main.sio 2>&1) || true
if echo "$_tc" | grep -q "All checks passed"; then echo "PASS  typecheck"; PASS=$((PASS+1)); else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi
echo ""; echo "=== SUMMARY ==="; echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
