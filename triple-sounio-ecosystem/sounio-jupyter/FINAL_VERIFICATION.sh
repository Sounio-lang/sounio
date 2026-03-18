#!/bin/bash
# Final verification script for sounio-jupyter implementation

set -e

echo "=========================================="
echo "Sounio Jupyter Kernel - Final Verification"
echo "=========================================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python3 --version

# Check syntax of all Python files
echo "✓ Checking Python syntax..."
for file in sounio_kernel/*.py tests/*.py; do
    python3 -m py_compile "$file" 2>/dev/null || {
        echo "✗ Syntax error in $file"
        exit 1
    }
    echo "  ✓ $file"
done

# Check JSON files
echo "✓ Validating JSON..."
python3 -m json.tool kernel.json > /dev/null && echo "  ✓ kernel.json"

# Check TOML file
echo "✓ Validating TOML..."
python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))" 2>/dev/null && echo "  ✓ pyproject.toml"

# Count lines of code
echo "✓ Counting implementation..."
kernel_lines=$(find sounio_kernel -name "*.py" | xargs wc -l | tail -1 | awk '{print $1}')
test_lines=$(find tests -name "*.py" | xargs wc -l | tail -1 | awk '{print $1}')
echo "  - Core implementation: $kernel_lines lines"
echo "  - Tests: $test_lines lines"

# List all files
echo "✓ File structure:"
find . -type f \( -name "*.py" -o -name "*.json" -o -name "*.toml" \) ! -path "*/__pycache__/*" | sort | sed 's/^/  /'

echo ""
echo "=========================================="
echo "Running Tests (61 tests)"
echo "=========================================="
echo ""

python3 -m pytest tests/ -v --tb=short

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""
python3 -m pytest tests/ --tb=no -q

echo ""
echo "=========================================="
echo "Verification Complete ✓"
echo "=========================================="
echo ""
echo "Installation instructions:"
echo "  pip install -e ."
echo "  jupyter kernelspec install ./kernel_spec --user"
echo "  jupyter notebook"
echo ""
