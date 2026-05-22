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

echo "--> check graphics::scatter"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/scatter.sio"

echo "--> check graphics::heatmap"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/heatmap.sio"

echo "--> check graphics::epistemic"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/epistemic.sio"

echo "--> check graphics::svg"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/svg.sio"

echo "--> check graphics::tile"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/tile.sio"

echo "--> check graphics::png"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/png.sio"

echo "--> check graphics::plot_png"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/plot_png.sio"

echo "--> check graphics::animate"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/animate.sio"

echo "--> check graphics::tiled_plot"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/tiled_plot.sio"

echo "--> check graphics::bar"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/bar.sio"

echo "--> check graphics::histogram"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/histogram.sio"

echo "--> check graphics::grid"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/grid.sio"

echo "--> check graphics::errorbar"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/errorbar.sio"

if [[ -f "${ROOT_DIR}/stdlib/graphics/text.sio" ]]; then
  echo "--> check graphics::text"
  ${SOUC} check "${ROOT_DIR}/stdlib/graphics/text.sio"
fi

echo "--> check graphics::extra_png"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/extra_png.sio"

echo "--> check graphics::area"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/area.sio"

echo "--> check graphics::boxplot"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/boxplot.sio"

echo "--> check graphics::pie"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/pie.sio"

echo "--> check graphics::filter"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/filter.sio"

echo "--> check graphics::multi_series"
${SOUC} check "${ROOT_DIR}/stdlib/graphics/multi_series.sio"

if [[ -f "${ROOT_DIR}/stdlib/graphics/quality.sio" ]]; then
  echo "--> check graphics::quality"
  ${SOUC} check "${ROOT_DIR}/stdlib/graphics/quality.sio"
fi

# 2. Smoke test compiles and runs
echo "--> compile + run graphics_smoke"
${SOUC} compile "${ROOT_DIR}/tests/run-pass/graphics_smoke.sio" -o /tmp/graphics_smoke
/tmp/graphics_smoke

echo "=== Graphics Scaffold Gate: PASS ==="
