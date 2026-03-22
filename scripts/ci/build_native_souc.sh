#!/bin/bash
# Build the self-hosted native Sounio compiler for CI use.
# Produces /tmp/souc-native.elf via the bootstrap chain.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LEAN="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
OUT="${1:-/tmp/souc-native.elf}"

# Strategy 0: Use pre-built native compiler (v1.0.0 self-hosted fixed-point)
NATIVE_PREBUILT="$ROOT_DIR/artifacts/bootstrap/souc-native-v1.0.0.elf"
if [ -x "$NATIVE_PREBUILT" ]; then
    echo "Using pre-built native compiler..."
    cp "$NATIVE_PREBUILT" "$OUT"
    chmod +x "$OUT"
    echo "Native compiler ready: $OUT ($(stat -c%s "$OUT") bytes)"
    exit 0
fi

# Strategy 1: Use boot4.elf (pure native, no dependencies)
for BOOT4_ELF in \
    "$ROOT_DIR/artifacts/bootstrap/boot4.elf" \
    "$ROOT_DIR/artifacts/bootstrap/final_boot4.elf" \
    "$ROOT_DIR/bootstrap/stage0"; do
    if [ -x "$BOOT4_ELF" ]; then
        echo "Bootstrapping native compiler via $(basename $BOOT4_ELF)..."
        "$BOOT4_ELF" "$LEAN" /tmp/souc-s1.elf 2>/dev/null
        if [ -f /tmp/souc-s1.elf ] && [ -s /tmp/souc-s1.elf ]; then
            chmod +x /tmp/souc-s1.elf
            /tmp/souc-s1.elf "$LEAN" "$OUT" 2>/dev/null
            if [ -f "$OUT" ] && [ -s "$OUT" ]; then
                chmod +x "$OUT"
                echo "Native compiler built: $OUT ($(stat -c%s "$OUT") bytes)"
                exit 0
            fi
        fi
        echo "warn: $BOOT4_ELF bootstrap failed, trying next..."
    fi
done

# Strategy 2: Use JIT binary
JIT="$ROOT_DIR/artifacts/omega/souc-bin/souc-linux-x86_64-jit"
if [ -x "$JIT" ]; then
    echo "Bootstrapping native compiler via JIT..."
    "$JIT" run "$LEAN" -- "$LEAN" "$OUT" 2>/dev/null || true
    if [ -f "$OUT" ] && [ -s "$OUT" ]; then
        chmod +x "$OUT"
        echo "Native compiler built: $OUT ($(stat -c%s "$OUT") bytes)"
        exit 0
    fi
fi

echo "error: no bootstrap binary available"
exit 1
