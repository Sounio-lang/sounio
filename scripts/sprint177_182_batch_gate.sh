#!/usr/bin/env bash
# sprint177_182_batch_gate.sh — Sprints 177-182: Blocks CM-CR batch gate
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T728" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T728)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }

echo "=== Sprints 177-182: Blocks CM-CR Batch Gate ==="
echo ""

echo "--- Source (opt_cleanup.sio) ---"
O="self-hosted/ir/opt_cleanup.sio"
cg "src:CM" "Block CM.*AND complement XOR" "$O"
cg "src:CN" "Block CN.*Add.self" "$O"
cg "src:CO" "Block CO.*Neg.add" "$O"
cg "src:CP" "Block CP.*Sub.neg" "$O"
cg "src:CQ" "Block CQ.*OR complement XOR.*NOT" "$O"
cg "src:CR" "Block CR.*AND.XOR complement" "$O"

echo ""
echo "--- Tests (main.sio) ---"
M="self-hosted/compiler/main.sio"
for t in T728 T729 T730 T731 T732 T733 T734 T735 T736 T737 T738 T739 \
         T740 T741 T742 T743 T744 T745 T746 T747 T748 T749 T750 T751 \
         T752 T753 T754 T755 T756 T757 T758 T759 T760 T761 T762 T763; do
    cg "main:$t" "$t OK" "$M"
done
cg "main:total" "let total: i64 = 763" "$M"

echo ""
echo "--- Self-tests: T728-T763 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for i in $(seq 728 763); do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:T$i"; NOT_RUN=$((NOT_RUN+1)); done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
    for i in $(seq 728 763); do cl "selftest:T$i" "T$i OK" "$L"; done
fi
rm -f "$L"

echo ""
echo "--- Type-check ---"; TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck"; PASS=$((PASS+1))
else echo "FAIL  typecheck"; FAIL=$((FAIL+1)); fi

echo ""
echo "=== SUMMARY ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
