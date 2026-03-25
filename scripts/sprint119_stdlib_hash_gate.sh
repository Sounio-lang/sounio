#!/usr/bin/env bash
set -eo pipefail
SOUC=./bin/souc
PASS=0; FAIL=0; TOTAL=4

grepcheck() { local name=$1 file=$2 pat=$3; if grep -q "$pat" "$file"; then echo "  PASS: $name"; PASS=$((PASS+1)); else echo "  FAIL: $name"; FAIL=$((FAIL+1)); fi; }

echo "=== Sprint 119 — Stdlib Hash Gate ==="
echo -n "T1_contract_check: "; if $SOUC check self-hosted/interop/contract.sio >/dev/null 2>&1; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL"; FAIL=$((FAIL+1)); fi
grepcheck "T2_stdlib_hash_field" self-hosted/interop/contract.sio "stdlib_hash"
grepcheck "T3_compute_stdlib_hash" self-hosted/interop/contract.sio "compute_stdlib_hash"
grepcheck "T4_fnv1a_hash_string" self-hosted/interop/contract.sio "fnv1a_hash_string"

echo ""
echo "PASS=$PASS FAIL=$FAIL TOTAL=$TOTAL"
