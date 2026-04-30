#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

MSL="sedenion_f3.metal"
LIB="sedenion_f3.metallib"
RUNNER="sed_f3_metal_runner"

echo "=== Compiling Metal shader ==="
xcrun metal -std=metal3.1 -o "${MSL%.metal}.air" "$MSL"
xcrun metallib -o "$LIB" "${MSL%.metal}.air"
echo "  metallib: $(wc -c < "$LIB") bytes"
xcrun metal-nm "$LIB" | grep sed_f3

echo "=== Compiling Swift runner ==="
swiftc -O sed_f3_metal_runner.swift -o "$RUNNER" -framework Metal -framework Foundation
codesign -s - "$RUNNER"

echo "=== Running GPU dispatch ==="
"./$RUNNER" "$LIB"
