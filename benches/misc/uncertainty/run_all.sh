#!/bin/bash
# Run all uncertainty propagation benchmarks
# Usage: ./run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPILER="${SCRIPT_DIR}/../../compiler/target/release/souc"

echo "========================================"
echo "Uncertainty Propagation Benchmark Suite"
echo "========================================"
echo ""
echo "Hardware: $(uname -m)"
echo "OS: $(uname -s)"
echo "Date: $(date -Iseconds)"
echo ""

# Check dependencies
echo "Checking dependencies..."

if ! command -v python3 &> /dev/null; then
    echo "WARNING: python3 not found, skipping Python benchmarks"
    SKIP_PYTHON=1
fi

if ! command -v julia &> /dev/null; then
    echo "WARNING: julia not found, skipping Julia benchmarks"
    SKIP_JULIA=1
fi

if [ ! -f "$COMPILER" ]; then
    echo "WARNING: Sounio compiler not found at $COMPILER"
    echo "Building compiler..."
    (cd "${SCRIPT_DIR}/../../compiler" && cargo build --release --features jit)
fi

echo ""

# Python benchmarks
if [ -z "$SKIP_PYTHON" ]; then
    echo "========================================"
    echo "Running Python benchmarks..."
    echo "========================================"
    python3 "${SCRIPT_DIR}/python_uncertainty.py"
    echo ""
fi

# Julia benchmarks
if [ -z "$SKIP_JULIA" ]; then
    echo "========================================"
    echo "Running Julia benchmarks..."
    echo "========================================"
    julia "${SCRIPT_DIR}/julia_uncertainty.jl"
    echo ""
fi

# Sounio benchmarks
echo "========================================"
echo "Running Sounio benchmarks..."
echo "========================================"
"$COMPILER" run "${SCRIPT_DIR}/sounio_uncertainty.sio"
echo ""

echo "========================================"
echo "Benchmark suite complete!"
echo "========================================"
