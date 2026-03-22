#!/bin/bash
# Build the self-hosted native Sounio compiler for CI use.
# Produces /tmp/souc-native.elf via the bootstrap chain.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

BOOT4="$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-jit"
LEAN="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
OUT="${1:-/tmp/souc-native.elf}"

# If boot4 JIT binary exists, use it to bootstrap
if [ -x "$BOOT4" ]; then
    echo "Bootstrapping native compiler via JIT..."
    "$BOOT4" run "$LEAN" -- "$LEAN" "$OUT" 2>/dev/null || true
    if [ -f "$OUT" ] && [ -s "$OUT" ]; then
        chmod +x "$OUT"
        echo "Native compiler built: $OUT ($(stat -c%s "$OUT") bytes)"
        exit 0
    fi
fi

# Fallback: use pre-built boot4.elf if available
BOOT4_ELF="$ROOT_DIR/artifacts/bootstrap/final_boot4.elf"
if [ -x "$BOOT4_ELF" ]; then
    echo "Bootstrapping native compiler via boot4.elf..."
    "$BOOT4_ELF" "$LEAN" /tmp/souc-s1.elf
    chmod +x /tmp/souc-s1.elf
    /tmp/souc-s1.elf "$LEAN" "$OUT"
    chmod +x "$OUT"
    echo "Native compiler built: $OUT ($(stat -c%s "$OUT") bytes)"
    exit 0
fi

echo "error: no bootstrap binary available (need JIT or boot4.elf)"
exit 1
