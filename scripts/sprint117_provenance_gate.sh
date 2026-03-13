#!/usr/bin/env bash
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=6

echo "=== Sprint 117: Compile Provenance Gate ==="
echo ""

# T1: souc check contract.sio
echo -n "T1 contract.sio check ... "
if $SOUC check self-hosted/interop/contract.sio >/dev/null 2>&1; then
    echo "PASS"; PASS=$((PASS+1))
else
    echo "FAIL"; FAIL=$((FAIL+1))
fi

# T2: souc check file_write.sio
echo -n "T2 file_write.sio check ... "
if $SOUC check self-hosted/io/file_write.sio >/dev/null 2>&1; then
    echo "PASS"; PASS=$((PASS+1))
else
    echo "FAIL"; FAIL=$((FAIL+1))
fi

# T3: KernelProvenance struct exists
echo -n "T3 KernelProvenance struct ... "
if grep -q 'struct KernelProvenance' self-hosted/interop/contract.sio; then
    echo "PASS"; PASS=$((PASS+1))
else
    echo "FAIL"; FAIL=$((FAIL+1))
fi

# T4: fnv1a_hash_string function exists
echo -n "T4 fnv1a_hash_string fn ... "
if grep -q 'fn fnv1a_hash_string' self-hosted/interop/contract.sio; then
    echo "PASS"; PASS=$((PASS+1))
else
    echo "FAIL"; FAIL=$((FAIL+1))
fi

# T5: io_write_provenance_json function exists
echo -n "T5 io_write_provenance_json fn ... "
if grep -q 'fn io_write_provenance_json' self-hosted/io/file_write.sio; then
    echo "PASS"; PASS=$((PASS+1))
else
    echo "FAIL"; FAIL=$((FAIL+1))
fi

# T6: source_hash field in KernelProvenance
echo -n "T6 source_hash field ... "
if grep -A5 'struct KernelProvenance' self-hosted/interop/contract.sio | grep -q 'source_hash'; then
    echo "PASS"; PASS=$((PASS+1))
else
    echo "FAIL"; FAIL=$((FAIL+1))
fi

echo ""
echo "=== Results: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN TOTAL=$TOTAL ==="
exit $FAIL
