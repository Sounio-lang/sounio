#!/usr/bin/env bash
set -euo pipefail

# Compile a HIP C++ source file into a real AMD HSACO code object.
# Usage: hip_compile_hsaco.sh <input.hip.cpp> <output.hsaco> [gfx_target]

INPUT="${1:?usage: hip_compile_hsaco.sh <input.hip.cpp> <output.hsaco> [gfx_target]}"
OUTPUT="${2:?usage: hip_compile_hsaco.sh <input.hip.cpp> <output.hsaco> [gfx_target]}"
GFX_TARGET="${3:-gfx1100}"   # Radeon R7900 AI PRO = RDNA3 Navi 31

HIPCC="${HIPCC:-hipcc}"

# --genco emits a code object directly from HIP source.
# This is the canonical ROCm path for ahead-of-time kernel compilation.
"$HIPCC" -O3 -offload-arch="$GFX_TARGET" --genco "$INPUT" -o "$OUTPUT"

echo "HSACO generated: $OUTPUT (target=$GFX_TARGET)"
