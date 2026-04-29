#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${SOUNIO_METAL_ALGEBRA_WORK_DIR:-$(mktemp -d /tmp/sounio-metal-algebra.XXXXXX)}"
MSL_OSSM="$ROOT_DIR/tests/gpu/metal_first/ossm_oct_step_f64_checked.metal"
MSL_SED="$ROOT_DIR/tests/gpu/metal_first/sedenion_cd_step_f64_checked.metal"
RUNNER_SRC="$ROOT_DIR/scripts/gpu-metal-validation/algebra_metal_runner.swift"
LIB="$WORK_DIR/algebra_kernels.metallib"
RUNNER="$WORK_DIR/algebra_metal_runner"

mkdir -p "$WORK_DIR"

echo "=== Compiling Metal algebra shaders ==="
xcrun metal -std=metal3.1 -o "$WORK_DIR/ossm_oct_step_f64_checked.air" "$MSL_OSSM"
xcrun metal -std=metal3.1 -o "$WORK_DIR/sedenion_cd_step_f64_checked.air" "$MSL_SED"
xcrun metallib -o "$LIB" "$WORK_DIR/ossm_oct_step_f64_checked.air" "$WORK_DIR/sedenion_cd_step_f64_checked.air"
echo "  metallib: $(wc -c < "$LIB") bytes"

echo "=== Compiling Swift algebra runner ==="
swiftc -O "$RUNNER_SRC" -o "$RUNNER" -framework Metal -framework Foundation
if command -v codesign >/dev/null 2>&1; then
  codesign -s - "$RUNNER"
fi

echo "=== Running Metal algebra dispatch ==="
"$RUNNER" "$LIB"
