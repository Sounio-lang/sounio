#!/usr/bin/env bash
# Sprint 114 — Structured Diagnostics gate
set -eo pipefail

SOUC=./bin/souc
CONTRACT=self-hosted/interop/contract.sio
FILE_WRITE=self-hosted/io/file_write.sio
PASS=0; FAIL=0; TOTAL=0

run_check() {
    local name="$1" file="$2"
    TOTAL=$((TOTAL+1))
    if timeout 30 $SOUC check "$file" 2>&1 | grep -q "All checks passed"; then
        echo "  PASS: $name"; PASS=$((PASS+1))
    else
        echo "  FAIL: $name"; FAIL=$((FAIL+1))
    fi
}

check_grep() {
    local name="$1" file="$2" pattern="$3"
    TOTAL=$((TOTAL+1))
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "  PASS: $name"; PASS=$((PASS+1))
    else
        echo "  FAIL: $name"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 114 — Structured Diagnostics Gate ==="
echo ""

echo "[typecheck]"
run_check "T1_contract_check" "$CONTRACT"
run_check "T2_file_write_check" "$FILE_WRITE"

echo ""
echo "[structural]"
check_grep "T3_KernelDiagnostic_struct"   "$CONTRACT" "struct KernelDiagnostic {"
check_grep "T4_KernelDiagnostics_struct"  "$CONTRACT" "struct KernelDiagnostics {"
check_grep "T5_diagnostics_add_fn"        "$CONTRACT" "fn kernel_diagnostics_add("
check_grep "T6_write_diagnostics_fn"      "$FILE_WRITE" "fn io_write_diagnostics_json("

echo ""
echo "========================================="
echo "Sprint 114 — Structured Diagnostics"
echo "PASS=$PASS  FAIL=$FAIL  TOTAL=$TOTAL"
echo "========================================="

if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
