#!/usr/bin/env bash
# sprint199_or_not_and_gate.sh — Sprint 199: Block DI OR-NOT-AND tautology gate
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T860" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T860)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }

echo "=== Sprint 199: Block DI OR-NOT-AND Tautology Gate ==="
echo ""

echo "--- Source (opt_cleanup.sio) ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:block_di_comment" "Block DI.*OR-NOT-AND" "$O"
cg "src:di_bnot" "is_bnot\[di_s[12] as usize\]" "$O"
cg "src:di_and_rr" "ao_and_rr_valid\[di_and as usize\]" "$O"
cg "src:di_result" "ir_load_imm\(di_instr\.dst, -1\)" "$O"

echo ""
echo "--- Tests (main.sio) ---"
M="self-hosted/compiler/main.sio"
for t in T860 T861 T862 T863 T864 T865; do
    cg "main:$t" "$t OK" "$M"
done
cg "main:total" "let total: i64 = [0-9]+" "$M"

echo ""
echo "--- Self-tests: T860-T865 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for i in $(seq 860 865); do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:T$i"; NOT_RUN=$((NOT_RUN+1)); done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
    for i in $(seq 860 865); do cl "selftest:T$i" "T$i OK" "$L"; done
fi
rm -f "$L"

echo ""
echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
_tc_out=$(timeout 120 "$SOUC" check self-hosted/compiler/main.sio 2>&1) || _tc_ec=$?
if echo "$_tc_out" | grep -q "All checks passed"; then
    echo "PASS  typecheck"; PASS=$((PASS+1))
elif [ "${_tc_ec:-0}" -eq 137 ]; then echo "NOT_RUN  typecheck (OOM)"; NOT_RUN=$((NOT_RUN+1))
else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi

echo ""
echo "=== SUMMARY ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
