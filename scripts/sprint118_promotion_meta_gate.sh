#!/usr/bin/env bash
set -eo pipefail
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; TOTAL=6

grepcheck() { local name=$1 file=$2 pat=$3; if grep -q "$pat" "$file"; then echo "  PASS: $name"; PASS=$((PASS+1)); else echo "  FAIL: $name"; FAIL=$((FAIL+1)); fi; }

echo "=== Sprint 118 — Promotion Metadata Gate ==="
echo -n "T1_contract_check: "; if $SOUC check self-hosted/interop/contract.sio >/dev/null 2>&1; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL"; FAIL=$((FAIL+1)); fi
echo -n "T2_file_write_check: "; if $SOUC check self-hosted/io/file_write.sio >/dev/null 2>&1; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL"; FAIL=$((FAIL+1)); fi
grepcheck "T3_PromotionRecord" self-hosted/interop/contract.sio "PromotionRecord"
grepcheck "T4_PromotionSummary" self-hosted/interop/contract.sio "PromotionSummary"
grepcheck "T5_promotion_summary_add" self-hosted/interop/contract.sio "promotion_summary_add"
grepcheck "T6_write_promotions_json" self-hosted/io/file_write.sio "io_write_promotions_json"

echo ""
echo "PASS=$PASS FAIL=$FAIL TOTAL=$TOTAL"
