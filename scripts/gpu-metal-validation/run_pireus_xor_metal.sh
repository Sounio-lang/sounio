#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MSL="${1:?usage: run_pireus_xor_metal.sh <generated.metal> [kernel-name]}"
KERNEL_NAME="${2:-sedenion_xor_product}"
WORK_DIR="${SOUNIO_PIREUS_METAL_WORK_DIR:-$(mktemp -d /tmp/sounio-pireus-metal.XXXXXX)}"
AIR="$WORK_DIR/pireus_xor.air"
LIB="$WORK_DIR/pireus_xor.metallib"
RUNNER="$WORK_DIR/pireus_xor_metal_runner"

mkdir -p "$WORK_DIR"
xcrun -sdk macosx metal -std=metal3.1 -c "$MSL" -o "$AIR"
xcrun metallib -o "$LIB" "$AIR"
swiftc -O "$ROOT_DIR/scripts/gpu-metal-validation/pireus_xor_metal_runner.swift" \
  -o "$RUNNER" -framework Metal -framework Foundation
if command -v codesign >/dev/null 2>&1; then
  codesign -s - "$RUNNER"
fi
"$RUNNER" "$LIB" "$KERNEL_NAME"
