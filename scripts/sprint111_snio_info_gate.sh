#!/usr/bin/env bash
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; NOT_RUN=0

run_test() {
    local name="$1"; shift
    if eval "$@" >/dev/null 2>&1; then
        echo "PASS: $name"; PASS=$((PASS+1))
    else
        echo "FAIL: $name"; FAIL=$((FAIL+1))
    fi
}

# T1: protocol.sio check passes
run_test "T1_check_protocol" "$SOUC check self-hosted/interop/protocol.sio"

# T2: server.sio check passes
run_test "T2_check_server" "$SOUC check self-hosted/interop/server.sio"

# T3: IPC_MSG_INFO constant exists
run_test "T3_msg_info_constant" "grep -q 'IPC_MSG_INFO' self-hosted/interop/protocol.sio"

# T4: IPC_MSG_CAPABILITIES constant exists
run_test "T4_msg_caps_constant" "grep -q 'IPC_MSG_CAPABILITIES' self-hosted/interop/protocol.sio"

# T5: capability bitmask constants
run_test "T5_cap_ffi_constant" "grep -q 'IPC_CAP_FFI' self-hosted/interop/protocol.sio"

# T6: info handler in server
run_test "T6_info_handler" "grep -q 'ipc_handle_info' self-hosted/interop/server.sio"

# T7: capabilities handler in server
run_test "T7_caps_handler" "grep -q 'ipc_handle_capabilities' self-hosted/interop/server.sio"

# T8: serve loop dispatches INFO
run_test "T8_serve_info_dispatch" "grep -q 'IPC_MSG_INFO' self-hosted/interop/server.sio"

echo ""
echo "Sprint 111 gate: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN (total=$((PASS+FAIL+NOT_RUN)))"
