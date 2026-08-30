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

host_target() {
    case "$1" in
        Linux:x86_64|Linux:amd64) printf '%s\n' "x86_64-linux" ;;
        Linux:arm64|Linux:aarch64) printf '%s\n' "aarch64-linux" ;;
        Darwin:arm64|Darwin:aarch64) printf '%s\n' "aarch64-macos" ;;
        Darwin:x86_64|Darwin:amd64) printf '%s\n' "x86_64-macos" ;;
        *) return 1 ;;
    esac
}

portable_size() {
    local path="$1"
    stat -c%s "$path" 2>/dev/null || stat -f%z "$path"
}

run_bootstrap_step() {
    local compiler_bin="$1"
    local src_path="$2"
    local out_path="$3"
    local target="${4:-}"

    if declare -F sounio_ad_hoc_codesign >/dev/null 2>&1; then
        sounio_ad_hoc_codesign "$compiler_bin"
    fi

    set +e
    if [[ -n "$target" ]]; then
        "$compiler_bin" "$src_path" "$out_path" --target "$target" >/dev/null 2>&1
    else
        "$compiler_bin" "$src_path" "$out_path" >/dev/null 2>&1
    fi
    local rc=$?
    set -e

    if [[ $rc -ne 0 ]] || [[ ! -s "$out_path" ]]; then
        return 1
    fi

    chmod +x "$out_path"
    return 0
}

bootstrap_direct_lean() {
    local seed_bin="$1"
    local stage1_bin
    stage1_bin="$(mktemp /tmp/sounio-lean-stage1.XXXXXX)"

    rm -f "$stage1_bin"
    if ! run_bootstrap_step "$seed_bin" "$LEAN" "$stage1_bin" "$HOST_TARGET"; then
        rm -f "$stage1_bin"
        return 1
    fi
    if ! run_bootstrap_step "$stage1_bin" "$LEAN" "$OUT" "$HOST_TARGET"; then
        rm -f "$stage1_bin"
        return 1
    fi

    rm -f "$stage1_bin"
    return 0
}

bootstrap_via_boot4_chain() {
    local seed_bin="$1"
    local boot4_src="$ROOT_DIR/bootstrap/boot4.sio"
    local stage1_bin
    local stage2_bin

    if [[ ! -f "$boot4_src" ]]; then
        return 1
    fi

    stage1_bin="$(mktemp /tmp/sounio-boot4-stage1.XXXXXX)"
    stage2_bin="$(mktemp /tmp/sounio-boot4-stage2.XXXXXX)"

    rm -f "$stage1_bin" "$stage2_bin"

    if ! run_bootstrap_step "$seed_bin" "$boot4_src" "$stage1_bin"; then
        rm -f "$stage1_bin" "$stage2_bin"
        return 1
    fi
    if ! run_bootstrap_step "$stage1_bin" "$boot4_src" "$stage2_bin"; then
        rm -f "$stage1_bin" "$stage2_bin"
        return 1
    fi
    if ! run_bootstrap_step "$stage2_bin" "$LEAN" "$OUT"; then
        rm -f "$stage1_bin" "$stage2_bin"
        return 1
    fi

    rm -f "$stage1_bin" "$stage2_bin"
    return 0
}

LEAN="$ROOT_DIR/self-hosted/compiler/lean_single.sio"
OUT="${1:-/tmp/souc-native}"
if [[ "$OUT" == -* ]]; then
    echo "error: output path must not start with '-': $OUT" >&2
    exit 2
fi
FORCE_SOURCE_BOOTSTRAP="${SOUNIO_FORCE_SOURCE_BOOTSTRAP:-0}"
NATIVE_SEED_OVERRIDE="${SOUNIO_NATIVE_SEED:-}"
if [[ -n "$NATIVE_SEED_OVERRIDE" ]]; then
    if [[ ! -x "$NATIVE_SEED_OVERRIDE" ]]; then
        echo "error: SOUNIO_NATIVE_SEED is not executable: $NATIVE_SEED_OVERRIDE" >&2
        exit 2
    fi
    FORCE_SOURCE_BOOTSTRAP=1
fi
LEAN_BYTES="$(portable_size "$LEAN")"
BOOT4_DIRECT_SRC_CAP=1048576

HOST_PLATFORM="$(host_platform)"
HOST_TARGET="$(host_target "$HOST_PLATFORM")"
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

# The canonical lean_single fixed point is newer than the compatibility host
# artifact on Linux x86_64. Prefer it for staged source bootstrap, while
# preserving the host artifact as a fallback seed.
SOURCE_BOOTSTRAP_SEEDS=("${HOST_PREBUILTS[@]}")
case "$HOST_PLATFORM" in
    Linux:x86_64|Linux:amd64)
        if [[ -x "$ROOT_DIR/bin/souc-lean-single-x86_64" ]]; then
            SOURCE_BOOTSTRAP_SEEDS=("$ROOT_DIR/bin/souc-lean-single-x86_64" "${SOURCE_BOOTSTRAP_SEEDS[@]}")
        fi
        ;;
esac
if [[ -n "$NATIVE_SEED_OVERRIDE" ]]; then
    SOURCE_BOOTSTRAP_SEEDS=("$NATIVE_SEED_OVERRIDE" "${SOURCE_BOOTSTRAP_SEEDS[@]}")
    echo "Using explicit source-bootstrap seed: $NATIVE_SEED_OVERRIDE"
fi

# Strategy 0: Use checked-in self-hosted native compiler (fast path -- copy)
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
    echo "Skipping checked-in native compiler artifact copy (SOUNIO_FORCE_SOURCE_BOOTSTRAP=1)"
fi

# Strategy 0.5: Staged bootstrap from an explicit or checked-in seed.
# boot4.elf is stale and miscompiles current lean_single.sio; use the checked-in
# artifact as a correct seed for a staged bootstrap instead.
for NATIVE_PREBUILT in "${SOURCE_BOOTSTRAP_SEEDS[@]}"; do
    if [ -x "$NATIVE_PREBUILT" ]; then
        echo "Staged bootstrap from seed: $NATIVE_PREBUILT"
        if bootstrap_direct_lean "$NATIVE_PREBUILT"; then
            if declare -F sounio_ad_hoc_codesign >/dev/null 2>&1; then
                sounio_ad_hoc_codesign "$OUT"
            fi
            echo "Native compiler built: $OUT ($(portable_size "$OUT") bytes)"
            exit 0
        fi
        echo "warn: staged bootstrap from $NATIVE_PREBUILT failed, falling back..."
    fi
done

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
        if [[ "$BOOT4_ELF" != "$ROOT_DIR/bootstrap/stage0" && "$LEAN_BYTES" -gt "$BOOT4_DIRECT_SRC_CAP" ]]; then
            echo "Bootstrapping native compiler via $(basename "$BOOT4_ELF") -> boot4.sio -> boot4.sio -> lean_single.sio..."
            if bootstrap_via_boot4_chain "$BOOT4_ELF"; then
                if declare -F sounio_ad_hoc_codesign >/dev/null 2>&1; then
                    sounio_ad_hoc_codesign "$OUT"
                fi
                echo "Native compiler built: $OUT ($(portable_size "$OUT") bytes)"
                exit 0
            fi
        else
            echo "Bootstrapping native compiler via $(basename "$BOOT4_ELF")..."
            if bootstrap_direct_lean "$BOOT4_ELF"; then
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
