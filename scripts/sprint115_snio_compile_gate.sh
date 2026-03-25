#!/usr/bin/env bash
set -eo pipefail
SOUC=./bin/souc
PASS=0; FAIL=0; TOTAL=4

grepcheck() { local name=$1 file=$2 pat=$3; if grep -q "$pat" "$file"; then echo "  PASS: $name"; PASS=$((PASS+1)); else echo "  FAIL: $name"; FAIL=$((FAIL+1)); fi; }

echo "=== Sprint 115 — SNIO COMPILE Message Gate ==="
echo -n "T1_protocol_check: "; if $SOUC check self-hosted/interop/protocol.sio >/dev/null 2>&1; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL"; FAIL=$((FAIL+1)); fi
echo -n "T2_server_check: "; if $SOUC check self-hosted/interop/server.sio >/dev/null 2>&1; then echo "PASS"; PASS=$((PASS+1)); else echo "FAIL"; FAIL=$((FAIL+1)); fi
grepcheck "T3_IPC_MSG_COMPILE" self-hosted/interop/protocol.sio "IPC_MSG_COMPILE"
grepcheck "T4_ipc_handle_compile" self-hosted/interop/server.sio "ipc_handle_compile"

echo ""
echo "PASS=$PASS FAIL=$FAIL TOTAL=$TOTAL"
