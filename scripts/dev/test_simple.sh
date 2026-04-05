#!/bin/bash
# Test the simple test file

cd /home/demetrios/work/sounio

# Set stdlib path
export SOUNIO_STDLIB_PATH="./stdlib"

# Run simple test
./souc run tests/frontend/simple_lexer_test.sio

echo "Exit code: $?"
