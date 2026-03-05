#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export SOUNIO_STDLIB_PATH="./stdlib"

# Create fixed lexer_integration_fixed.sio
cat > tests/frontend/lexer_integration_fixed.sio << 'FIXED_LEXER_INT'
[COMPLETE lexer_integration_fixed.sio CODE FROM EARLIER]
