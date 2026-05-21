#!/usr/bin/env bash
# scripts/ci/graphics_scaffold_gate.sh
#
# Acceptance gate for the native graphics library scaffold (Option C).
# Checks that all graphics module files compile and the smoke test passes.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${ROOT_DIR}/bin/souc"

echo "=== Graphics Scaffold Gate ==="

# 1. Check all module files compile
echo "--> check graphics::drawing"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/drawing.sio"

echo "--> check graphics::surface"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/surface.sio"

echo "--> check graphics::export"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/export.sio"

echo "--> check graphics::plot"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/plot.sio"

# 2. Smoke test compiles and runs
echo "--> compile + run graphics_smoke"
${SOUC} compile "${ROOT_DIR}/tests/run-pass/graphics_smoke.sio" -o /tmp/graphics_smoke
/tmp/graphics_smoke

echo "=== Graphics Scaffold Gate: PASS ==="
