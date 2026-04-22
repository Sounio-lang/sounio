#!/bin/bash
# Build the self-hosted native Sounio compiler for CI use.
# Produces a host-native compiler artifact, preferring checked-in binaries when available.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MACOS_CODESIGN_HELPER="$ROOT_DIR/scripts/lib/macos_codesign.sh"
if [[ -f "$MACOS_CODESIGN_HELPER" ]]; then
    # shellcheck source=/dev/null
    source "$MACOS_CODESIGN_HELPER"
fi

host_platform() {
    local host_os="${SOUNIO_HOST_OS_OVERRIDE:-$(uname -s 2>/dev/null || echo unknown)}"
    local host_arch="${SOUNIO_HOST_ARCH_OVERRIDE:-$(uname -m 2>/dev/null || echo unknown)}"
    printf '%s:%s\n' "$host_os" "$host_arch"
}

portable_size() {
    local path="$1"
    stat -c%s "$path" 2>/dev/null || stat -f%z "$path"
}

LEAN="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
OUT="${1:-/tmp/souc-native}"
FORCE_SOURCE_BOOTSTRAP="${SOUNIO_FORCE_SOURCE_BOOTSTRAP:-0}"

HOST_PLATFORM="$(host_platform)"
HOST_PREBUILTS=()

case "$HOST_PLATFORM" in
    Linux:x86_64|Linux:amd64)
        HOST_PREBUILTS=(
            "$ROOT_DIR/bin/souc-linux-x86_64"
            "$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64"
            "$ROOT_DIR/artifacts/bootstrap/souc-native-v1.0.0.elf"
        )
        ;;
    Darwin:arm64|Darwin:aarch64)
        HOST_PREBUILTS=(
            "$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-arm64-macos"
        )
        ;;
    Darwin:x86_64|Darwin:amd64)
        HOST_PREBUILTS=(
            "$ROOT_DIR/artifacts/self-hosted/souc-self-hosted-x86_64-macos"
        )
        ;;
    *)
        HOST_PREBUILTS=()
        ;;
esac

# Strategy 0: Use checked-in self-hosted native compiler
if [ "$FORCE_SOURCE_BOOTSTRAP" != "1" ]; then
    for NATIVE_PREBUILT in "${HOST_PREBUILTS[@]}"; do
        if [ -x "$NATIVE_PREBUILT" ]; then
            echo "Using native compiler artifact for $HOST_PLATFORM..."
            cp "$NATIVE_PREBUILT" "$OUT"
            chmod +x "$OUT"
            if declare -F sounio_ad_hoc_codesign >/dev/null 2>&1; then
                sounio_ad_hoc_codesign "$OUT"
            fi
            echo "Native compiler ready: $OUT ($(portable_size "$OUT") bytes)"
            exit 0
        fi
    done
else
    echo "Skipping checked-in native compiler artifacts (SOUNIO_FORCE_SOURCE_BOOTSTRAP=1)"
fi

# Strategy 1: Use boot4.elf (pure native, no dependencies)
case "$HOST_PLATFORM" in
    Darwin:*)
        echo "error: no source bootstrap path is available for $HOST_PLATFORM in this checkout" >&2
        echo "hint: use the checked-in Mach-O artifact under artifacts/self-hosted/" >&2
        exit 1
        ;;
esac

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
                if declare -F sounio_ad_hoc_codesign >/dev/null 2>&1; then
                    sounio_ad_hoc_codesign "$OUT"
                fi
                echo "Native compiler built: $OUT ($(portable_size "$OUT") bytes)"
                exit 0
            fi
        fi
        echo "warn: $BOOT4_ELF bootstrap failed, trying next..."
    fi
done

echo "error: no bootstrap binary available"
exit 1
