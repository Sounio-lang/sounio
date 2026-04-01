#!/usr/bin/env bash
set -eo pipefail

SOUC=./bin/souc
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=6

pass() { echo "PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

# T1: souc check contract.sio
if $SOUC check self-hosted/interop/contract.sio >/dev/null 2>&1; then
    pass "T1: souc check contract.sio"
else
    fail "T1: souc check contract.sio"
fi

# T2: souc check file_write.sio
if $SOUC check self-hosted/io/file_write.sio >/dev/null 2>&1; then
    pass "T2: souc check file_write.sio"
else
    fail "T2: souc check file_write.sio"
fi

# T3: ExportMetadata struct exists
if grep -q 'struct ExportMetadata' self-hosted/interop/contract.sio; then
    pass "T3: ExportMetadata struct exists"
else
    fail "T3: ExportMetadata struct exists"
fi

# T4: export_metadata_add function exists
if grep -q 'fn export_metadata_add' self-hosted/interop/contract.sio; then
    pass "T4: export_metadata_add function exists"
else
    fail "T4: export_metadata_add function exists"
fi

# T5: io_write_export_metadata_json function exists
if grep -q 'fn io_write_export_metadata_json' self-hosted/io/file_write.sio; then
    pass "T5: io_write_export_metadata_json function exists"
else
    fail "T5: io_write_export_metadata_json function exists"
fi

# T6: export_count field in ExportMetadata
if grep -A5 'struct ExportMetadata' self-hosted/interop/contract.sio | grep -q 'count: i64'; then
    pass "T6: export_count field exists in ExportMetadata"
else
    fail "T6: export_count field exists in ExportMetadata"
fi

echo ""
echo "=== Sprint 112 Export Metadata Gate ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
