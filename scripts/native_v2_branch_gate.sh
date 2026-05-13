#!/usr/bin/env bash
set -euo pipefail
SOUC=./bin/souc
OUT=/tmp/native_v2_branch_smoke
$SOUC run self-hosted/compiler/native_compile_driver.sio \
    -- tests/run-pass/native_v2_branch_smoke.sio -o $OUT
ACTUAL=$($OUT)
EXPECTED=$'55\n-1'
if [ "$ACTUAL" = "$EXPECTED" ]; then
    echo "PASS"
    exit 0
else
    echo "FAIL: expected '$EXPECTED' got '$ACTUAL'"
    exit 1
fi
