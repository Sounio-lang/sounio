#!/usr/bin/env bash
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=6

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }
not_run() { echo "  NOT_RUN: $1"; NOT_RUN=$((NOT_RUN + 1)); }

echo "=== Sprint 120: GPU Capability Probe Gate ==="
echo ""

# T1: souc check contract.sio
echo "T1: souc check contract.sio"
if $SOUC check self-hosted/interop/contract.sio 2>&1 | grep -q "All checks passed"; then
    pass "contract.sio check passes"
else
    fail "contract.sio check failed"
fi

# T2: souc check protocol.sio
echo "T2: souc check protocol.sio"
if $SOUC check self-hosted/interop/protocol.sio 2>&1 | grep -q "All checks passed"; then
    pass "protocol.sio check passes"
else
    fail "protocol.sio check failed"
fi

# T3: souc check server.sio
echo "T3: souc check server.sio"
if $SOUC check self-hosted/interop/server.sio 2>&1 | grep -q "All checks passed"; then
    pass "server.sio check passes"
else
    fail "server.sio check failed"
fi

# T4: GpuCapability struct exists
echo "T4: GpuCapability struct exists in contract.sio"
if grep -q "struct GpuCapability" self-hosted/interop/contract.sio; then
    pass "GpuCapability struct found"
else
    fail "GpuCapability struct not found"
fi

# T5: IPC_MSG_GPU_CAPS constant exists
echo "T5: IPC_MSG_GPU_CAPS constant exists in protocol.sio"
if grep -q "fn IPC_MSG_GPU_CAPS" self-hosted/interop/protocol.sio; then
    pass "IPC_MSG_GPU_CAPS constant found"
else
    fail "IPC_MSG_GPU_CAPS constant not found"
fi

# T6: io_write_gpu_caps_json function exists
echo "T6: io_write_gpu_caps_json function exists in file_write.sio"
if grep -q "fn io_write_gpu_caps_json" self-hosted/io/file_write.sio; then
    pass "io_write_gpu_caps_json function found"
else
    fail "io_write_gpu_caps_json function not found"
fi

echo ""
echo "=== Results: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN TOTAL=$TOTAL ==="
exit $FAIL
