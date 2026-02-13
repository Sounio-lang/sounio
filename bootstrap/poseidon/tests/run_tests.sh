#!/bin/bash
# Test runner for Poseidon VM

set -e

POSEIDON="../poseidon"
PASS=0
FAIL=0

echo "Generating test SOIR files..."
python3 gen_test_soir.py

run_test() {
    local name=$1
    local expected=$2
    local file="${name}.soir"

    echo -n "Testing $name... "

    if [ ! -f "$file" ]; then
        echo "FAIL (file not found)"
        FAIL=$((FAIL + 1))
        return
    fi

    set +e
    $POSEIDON "$file" > /dev/null 2>&1
    local actual=$?
    set -e

    if [ "$actual" -eq "$expected" ]; then
        echo "PASS (exit code $actual)"
        PASS=$((PASS + 1))
    else
        echo "FAIL (expected $expected, got $actual)"
        FAIL=$((FAIL + 1))
    fi
}

echo ""
echo "Running tests..."
echo "================"

run_test "add" 42
run_test "call" 42
run_test "branch" 1
run_test "loop" 10
run_test "return" 42

echo ""
echo "================"
echo "Results: $PASS passed, $FAIL failed"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed."
    exit 1
fi
