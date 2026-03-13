#!/usr/bin/env bash
set -eo pipefail
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; TOTAL=8

run() { local name=$1; shift; if "$@" >/dev/null 2>&1; then echo "  PASS: $name"; PASS=$((PASS+1)); else echo "  FAIL: $name"; FAIL=$((FAIL+1)); fi; }
grepcheck() { local name=$1 file=$2 pat=$3; if grep -q "$pat" "$file"; then echo "  PASS: $name"; PASS=$((PASS+1)); else echo "  FAIL: $name"; FAIL=$((FAIL+1)); fi; }

echo "=== Sprint 116+124 — SNIO Lifecycle Gate ==="
run "T1_protocol_check" $SOUC check self-hosted/interop/protocol.sio
run "T2_server_check" $SOUC check self-hosted/interop/server.sio
grepcheck "T3_IPC_MSG_DIAGNOSTICS" self-hosted/interop/protocol.sio "IPC_MSG_DIAGNOSTICS"
grepcheck "T4_IPC_MSG_HEALTH" self-hosted/interop/protocol.sio "IPC_MSG_HEALTH"
grepcheck "T5_IPC_MSG_INIT" self-hosted/interop/protocol.sio "IPC_MSG_INIT"
grepcheck "T6_IPC_MSG_STATS" self-hosted/interop/protocol.sio "IPC_MSG_STATS"
grepcheck "T7_IpcSessionState" self-hosted/interop/server.sio "IpcSessionState"
grepcheck "T8_ipc_handle_health" self-hosted/interop/server.sio "ipc_handle_health"

echo ""
echo "PASS=$PASS FAIL=$FAIL TOTAL=$TOTAL"
