#!/bin/bash
# Test script to verify lean driver parser functionality
#
# This script tests whether the parser correctly extracts identifiers
# from tokens and builds an AST with proper name fields.
#
# Usage: bash test_lean_driver_parser.sh
#
# Expected output on success:
# - AST for hello.sio showing fn main() with non-empty name field
# - Token count: 15 (from previous exploration)
# - No "parse error" messages

set -e

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit

if [ ! -f "$SOUC" ]; then
    echo "ERROR: SOUC binary not found at $SOUC"
    exit 1
fi

echo "=== Testing Parser with hello.sio ==="
echo ""
echo "Step 1: Parse hello.sio and show AST (parser only, no JIT compilation overhead)"
timeout 30 "$SOUC" check examples/hello.sio --show-ast 2>&1 | head -100

echo ""
echo "Step 2: Check type information (verify identifiers were extracted)"
timeout 30 "$SOUC" check examples/hello.sio --show-types 2>&1 | head -50

echo ""
echo "=== End of test ==="
echo ""
echo "If you see 'fn main' in the output above, the parser successfully extracted identifiers."
echo "If AST shows empty name fields, Gap 2 (token text population) may be the issue."
