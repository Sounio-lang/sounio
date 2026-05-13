#!/usr/bin/env bash
# Run the Metal MSL emitter gate test standalone.
# This does NOT require macOS or Apple GPU — it validates MSL codegen only.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC_BIN="${SOUNIO_TEST_SOUC_BIN:-$ROOT_DIR/bin/souc-linux-x86_64}"
TEST_SRC="$ROOT_DIR/self-hosted/gpu/test_metal_emitters.sio"
TEST_ELF="$(mktemp /tmp/metal_emitter_test.XXXXXX.elf)"
trap 'rm -f "$TEST_ELF"' EXIT

echo "=== Compiling Metal emitter gate test ==="
"$SOUC_BIN" "$TEST_SRC" "$TEST_ELF"
chmod +x "$TEST_ELF"

echo "=== Running Metal emitter gate test ==="
"$TEST_ELF"
