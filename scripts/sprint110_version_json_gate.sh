#!/usr/bin/env bash
set -eo pipefail

SOUC=./bin/souc
PASS=0; FAIL=0; NOT_RUN=0

run_test() {
    local name="$1"; shift
    if eval "$@" >/dev/null 2>&1; then
        echo "PASS: $name"; PASS=$((PASS+1))
    else
        echo "FAIL: $name"; FAIL=$((FAIL+1))
    fi
}

# T1: souc check ir.sio passes
run_test "T1_check_ir" "$SOUC check self-hosted/ir/ir.sio"

# T2: souc check file_write.sio passes
run_test "T2_check_file_write" "$SOUC check self-hosted/io/file_write.sio"

# T3: souc check main.sio passes
run_test "T3_check_main" "timeout 120 $SOUC check self-hosted/compiler/main.sio"

# T4: --version-json produces JSON output
run_test "T4_version_json_output" "timeout 30 $SOUC run self-hosted/compiler/main.sio -- --version-json 2>/dev/null | grep -q abi_version"

# T5: version constants grep in ir.sio
run_test "T5_abi_version_constant" "grep -q 'SOUNIO_ABI_VERSION' self-hosted/ir/ir.sio"

# T6: io_write_version_json function exists
run_test "T6_io_write_version_json_exists" "grep -q 'io_write_version_json' self-hosted/io/file_write.sio"

echo ""
echo "Sprint 110 gate: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN (total=$((PASS+FAIL+NOT_RUN)))"
