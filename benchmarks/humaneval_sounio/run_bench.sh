#!/usr/bin/env bash
# HumanEval-Sounio benchmark runner
# Usage: bash benchmarks/humaneval_sounio/run_bench.sh

SOUC="${SOUC:-./artifacts/omega/souc-bin/souc-linux-x86_64-jit}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$(pwd)/stdlib}"
PASS=0 FAIL=0
for f in benchmarks/humaneval_sounio/*.sio; do
    name=$(basename "$f")
    if timeout 30 $SOUC run "$f" >/dev/null 2>&1; then
        PASS=$((PASS+1))
        echo "PASS: $name"
    else
        FAIL=$((FAIL+1))
        echo "FAIL: $name"
    fi
done
echo ""
echo "HumanEval-Sounio: $PASS passed, $FAIL failed out of $((PASS+FAIL)) total"
