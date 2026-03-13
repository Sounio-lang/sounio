#!/usr/bin/env bash
set -eo pipefail
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; TOTAL=6

grepcheck() { local name=$1 file=$2 pat=$3; if grep -q "$pat" "$file"; then echo "  PASS: $name"; PASS=$((PASS+1)); else echo "  FAIL: $name"; FAIL=$((FAIL+1)); fi; }

echo "=== Sprint 121 — Diagnostic Enrichment Gate ==="
echo -n "T1_contract_check: "; if $SOUC check self-hosted/interop/contract.sio >/dev/null 2>&1; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL"; FAIL=$((FAIL+1)); fi
grepcheck "T2_fix_suggestion" self-hosted/interop/contract.sio "fix_suggestion"
grepcheck "T3_DIAG_E001" self-hosted/interop/contract.sio "DIAG_E001_UNDEFINED_VAR"
grepcheck "T4_DIAG_E010" self-hosted/interop/contract.sio "DIAG_E010_EFFECT_MISSING"
grepcheck "T5_diag_suggest_fix" self-hosted/interop/contract.sio "diag_suggest_fix"
COUNT=$(grep -c "DIAG_E0" self-hosted/interop/contract.sio)
if [ "$COUNT" -ge 10 ]; then echo "  PASS: T6_at_least_10_codes ($COUNT)"; PASS=$((PASS+1)); else echo "  FAIL: T6_at_least_10_codes ($COUNT)"; FAIL=$((FAIL+1)); fi

echo ""
echo "PASS=$PASS FAIL=$FAIL TOTAL=$TOTAL"
