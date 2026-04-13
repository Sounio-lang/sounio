#!/bin/bash
# Day 1 verification script for Sounio Jupyter Kernel

set -e

echo "==============================================="
echo "Day 1 Deliverable Verification"
echo "==============================================="
echo

# Check file structure
echo "1. Checking file structure..."
required_files=(
    "pyproject.toml"
    "README.md"
    "DAY1_DELIVERABLE.md"
    "kernel.json"
    "sounio_kernel/__init__.py"
    "sounio_kernel/kernel.py"
    "sounio_kernel/executor.py"
    "sounio_kernel/display.py"
    "sounio_kernel/magics.py"
    "tests/__init__.py"
    "tests/test_executor.py"
    "tests/test_display.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ MISSING: $file"
        exit 1
    fi
done
echo

# Check Python syntax
echo "2. Checking Python syntax..."
python3 -m py_compile \
    sounio_kernel/__init__.py \
    sounio_kernel/kernel.py \
    sounio_kernel/executor.py \
    sounio_kernel/display.py \
    sounio_kernel/magics.py && echo "  ✓ All Python modules compile"
echo

# Check souc binary
echo "3. Checking souc binary..."
if [ -x "/workspace/sounio/bin/souc" ]; then
    echo "  ✓ souc binary found and executable"
else
    echo "  ✗ souc binary not found or not executable"
    exit 1
fi
echo

# Check stdlib path
echo "4. Checking Sounio stdlib..."
if [ -d "/workspace/sounio/stdlib" ]; then
    echo "  ✓ stdlib path found"
else
    echo "  ✗ stdlib path not found"
    exit 1
fi
echo

# Run tests
echo "5. Running test suite..."
echo "  Running executor tests..."
python3 tests/test_executor.py > /tmp/test_executor.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ test_executor.py PASS"
else
    echo "  ✗ test_executor.py FAIL"
    cat /tmp/test_executor.log
    exit 1
fi

echo "  Running display tests..."
python3 tests/test_display.py > /tmp/test_display.log 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ test_display.py PASS"
else
    echo "  ✗ test_display.py FAIL"
    cat /tmp/test_display.log
    exit 1
fi
echo

# Test souc execution
echo "6. Testing souc execution..."
SOUC="/workspace/sounio/bin/souc"
cat > /tmp/test_kernel.sio << 'EOF'
fn main() with IO {
    print(1 + 1)
}
EOF

output=$($SOUC run /tmp/test_kernel.sio 2>&1)
if [ "$output" = "2" ]; then
    echo "  ✓ souc execution works (output: $output)"
else
    echo "  ✗ souc execution failed (output: $output)"
    exit 1
fi
echo

# Summary
echo "==============================================="
echo "✓ All Day 1 checks passed!"
echo "==============================================="
echo
echo "Next steps:"
echo "  1. Install with: pip install -e ."
echo "  2. Register kernel: jupyter kernelspec install kernelspec/ --user"
echo "  3. Verify: jupyter kernelspec list | grep sounio"
echo
echo "For Day 2 implementation, see DAY1_DELIVERABLE.md"
