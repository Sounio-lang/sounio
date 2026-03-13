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

# T1: contract.sio check passes
run_test "T1_check_contract" "$SOUC check self-hosted/interop/contract.sio"

# T2: file_write.sio check passes
run_test "T2_check_file_write" "$SOUC check self-hosted/io/file_write.sio"

# T3: main.sio check passes
run_test "T3_check_main" "timeout 120 $SOUC check self-hosted/compiler/main.sio"

# T4: KernelRequest struct exists
run_test "T4_kernel_request_struct" "grep -q 'struct KernelRequest' self-hosted/interop/contract.sio"

# T5: KernelResponse struct exists
run_test "T5_kernel_response_struct" "grep -q 'struct KernelResponse' self-hosted/interop/contract.sio"

# T6: kernel_response_ok function exists
run_test "T6_kernel_response_ok" "grep -q 'fn kernel_response_ok' self-hosted/interop/contract.sio"

# T7: io_write_kernel_response_json function exists
run_test "T7_response_json_fn" "grep -q 'io_write_kernel_response_json' self-hosted/io/file_write.sio"

# T8: --kernel-compile mode dispatch in main
run_test "T8_kernel_compile_dispatch" "grep -q 'kernel-compile' self-hosted/compiler/main.sio"

echo ""
echo "Sprint 113 gate: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN (total=$((PASS+FAIL+NOT_RUN)))"
