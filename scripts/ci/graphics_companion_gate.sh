#!/usr/bin/env bash
# Companion acceptance gate for the native graphics scaffold.
#
# This intentionally leaves the Phase 1 scaffold files untouched. It runs the
# owner gate first, then adds edge-case coverage for clipping and constant plots.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOUC="${ROOT_DIR}/bin/souc"

# Graphics value-types are large and passed by value on the stack: a Surface is
# ~2 MB (four [i64; 65536] channel arrays) and a TiledSurface embeds 9 of them
# (~18 MB), exceeding the default ~8-12 MB stack and SIGSEGVing at runtime (the
# binaries are logically correct — they pass with a larger stack). Raise the
# stack soft limit before running them; this is inherited by the owner gate and
# every smoke binary below. Guarded so the gate still runs where the limit cannot
# be changed. FOLLOW-UP: shrink Surface channel storage (tracking issue #187).
ulimit -S -s unlimited 2>/dev/null || ulimit -S -s 262144 2>/dev/null || true

echo "=== Graphics Companion Gate ==="

"${ROOT_DIR}/scripts/ci/graphics_scaffold_gate.sh"

echo "--> check graphics_companion_smoke"
"${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_companion_smoke.sio"

echo "--> compile + run graphics_companion_smoke"
"${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_companion_smoke.sio" -o /tmp/graphics_companion_smoke
/tmp/graphics_companion_smoke

echo "--> check graphics_gallery_smoke"
"${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_gallery_smoke.sio"

echo "--> compile + run graphics_gallery_smoke"
"${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_gallery_smoke.sio" -o /tmp/graphics_gallery_smoke
/tmp/graphics_gallery_smoke

echo "--> check graphics_degenerate_smoke"
"${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_degenerate_smoke.sio"

echo "--> compile + run graphics_degenerate_smoke"
"${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_degenerate_smoke.sio" -o /tmp/graphics_degenerate_smoke
/tmp/graphics_degenerate_smoke

if [[ -f "${ROOT_DIR}/stdlib/graphics/raster.sio" ]]; then
  echo "--> check graphics::raster"
  "${SOUC}" check "${ROOT_DIR}/stdlib/graphics/raster.sio"

  echo "--> check graphics_raster_quality_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_raster_quality_smoke.sio"

  echo "--> compile + run graphics_raster_quality_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_raster_quality_smoke.sio" -o /tmp/graphics_raster_quality_smoke
  /tmp/graphics_raster_quality_smoke
fi

if [[ -f "${ROOT_DIR}/stdlib/graphics/quality.sio" ]]; then
  echo "--> check graphics::quality"
  "${SOUC}" check "${ROOT_DIR}/stdlib/graphics/quality.sio"

  echo "--> check graphics_quality_plot_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_plot_smoke.sio"

  echo "--> compile + run graphics_quality_plot_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_plot_smoke.sio" -o /tmp/graphics_quality_plot_smoke
  /tmp/graphics_quality_plot_smoke

  echo "--> check graphics_quality_heatmap_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_heatmap_smoke.sio"

  echo "--> compile + run graphics_quality_heatmap_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_heatmap_smoke.sio" -o /tmp/graphics_quality_heatmap_smoke
  /tmp/graphics_quality_heatmap_smoke

  echo "--> check graphics_quality_contour_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_contour_smoke.sio"

  echo "--> compile + run graphics_quality_contour_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_contour_smoke.sio" -o /tmp/graphics_quality_contour_smoke
  /tmp/graphics_quality_contour_smoke

  echo "--> check graphics_quality_colorbar_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_colorbar_smoke.sio"

  echo "--> compile + run graphics_quality_colorbar_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_colorbar_smoke.sio" -o /tmp/graphics_quality_colorbar_smoke
  /tmp/graphics_quality_colorbar_smoke

  echo "--> check graphics_quality_band_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_band_smoke.sio"

  echo "--> compile + run graphics_quality_band_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_band_smoke.sio" -o /tmp/graphics_quality_band_smoke
  /tmp/graphics_quality_band_smoke

  echo "--> check graphics_quality_marker_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_marker_smoke.sio"

  echo "--> compile + run graphics_quality_marker_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_marker_smoke.sio" -o /tmp/graphics_quality_marker_smoke
  /tmp/graphics_quality_marker_smoke

  echo "--> check graphics_quality_error_bar_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_error_bar_smoke.sio"

  echo "--> compile + run graphics_quality_error_bar_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_error_bar_smoke.sio" -o /tmp/graphics_quality_error_bar_smoke
  /tmp/graphics_quality_error_bar_smoke

  echo "--> check graphics_quality_histogram_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_histogram_smoke.sio"

  echo "--> compile + run graphics_quality_histogram_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_histogram_smoke.sio" -o /tmp/graphics_quality_histogram_smoke
  /tmp/graphics_quality_histogram_smoke

  echo "--> check graphics_quality_ecdf_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_ecdf_smoke.sio"

  echo "--> compile + run graphics_quality_ecdf_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_ecdf_smoke.sio" -o /tmp/graphics_quality_ecdf_smoke
  /tmp/graphics_quality_ecdf_smoke

  echo "--> check graphics_quality_boxplot_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_boxplot_smoke.sio"

  echo "--> compile + run graphics_quality_boxplot_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_boxplot_smoke.sio" -o /tmp/graphics_quality_boxplot_smoke
  /tmp/graphics_quality_boxplot_smoke

  echo "--> check graphics_quality_violin_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_violin_smoke.sio"

  echo "--> compile + run graphics_quality_violin_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_violin_smoke.sio" -o /tmp/graphics_quality_violin_smoke
  /tmp/graphics_quality_violin_smoke

  echo "--> check graphics_quality_hexbin_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_quality_hexbin_smoke.sio"

  echo "--> compile + run graphics_quality_hexbin_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_quality_hexbin_smoke.sio" -o /tmp/graphics_quality_hexbin_smoke
  /tmp/graphics_quality_hexbin_smoke
fi

if [[ -f "${ROOT_DIR}/stdlib/graphics/svg.sio" ]]; then
  echo "--> check graphics::svg"
  "${SOUC}" check "${ROOT_DIR}/stdlib/graphics/svg.sio"

  echo "--> check graphics_svg_export_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_svg_export_smoke.sio"

  echo "--> compile + verify graphics_svg_export_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_svg_export_smoke.sio" -o /tmp/graphics_svg_export_smoke
  /tmp/graphics_svg_export_smoke > /tmp/graphics_svg_export_smoke.svg
  grep -q '<!-- line -->' /tmp/graphics_svg_export_smoke.svg
  grep -q '<!-- scatter -->' /tmp/graphics_svg_export_smoke.svg
  grep -q '<!-- heatmap -->' /tmp/graphics_svg_export_smoke.svg
  grep -q '<!-- epistemic -->' /tmp/graphics_svg_export_smoke.svg
  grep -q '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">' /tmp/graphics_svg_export_smoke.svg
  grep -q '<rect x="0" y="0" width="64" height="64" fill="#ffffff"/>' /tmp/graphics_svg_export_smoke.svg
  grep -q '<polyline fill="none" stroke="#ff0000" stroke-width="2" points="8,56 56,8"/>' /tmp/graphics_svg_export_smoke.svg
  grep -q '<circle cx="8" cy="56" r="3" fill="#0000ff"/>' /tmp/graphics_svg_export_smoke.svg
  grep -q '<circle cx="56" cy="8" r="3" fill="#0000ff"/>' /tmp/graphics_svg_export_smoke.svg
  grep -q '<rect x="8" y="32" width="24" height="24" fill="#0000ff"/>' /tmp/graphics_svg_export_smoke.svg
  grep -q '<rect x="32" y="32" width="24" height="24" fill="#ff0000"/>' /tmp/graphics_svg_export_smoke.svg
  grep -q '<rect x="8" y="8" width="24" height="24" fill="#00ff00"/>' /tmp/graphics_svg_export_smoke.svg
  grep -q '<rect x="32" y="8" width="24" height="24" fill="#00ffff"/>' /tmp/graphics_svg_export_smoke.svg
  grep -q '<polygon fill="#00ff00" stroke="none" points="8,8 56,8 56,56 8,56"/>' /tmp/graphics_svg_export_smoke.svg
  grep -q '<polyline fill="none" stroke="#ff0000" stroke-width="2" points="8,32 56,32"/>' /tmp/graphics_svg_export_smoke.svg
  grep -q '</svg>' /tmp/graphics_svg_export_smoke.svg
  grep -Eq '#[0-9a-f]{6}' /tmp/graphics_svg_export_smoke.svg

  echo "--> check graphics_epistemic_advanced_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_epistemic_advanced_smoke.sio"

  echo "--> compile + run graphics_epistemic_advanced_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_epistemic_advanced_smoke.sio" -o /tmp/graphics_epistemic_advanced_smoke
  /tmp/graphics_epistemic_advanced_smoke > /tmp/graphics_epistemic_advanced_smoke.svg
  grep -q 'Epistemic Confidence Decay' /tmp/graphics_epistemic_advanced_smoke.svg
  grep -q 'stroke-dasharray="3,2"' /tmp/graphics_epistemic_advanced_smoke.svg
  grep -q 'PASS graphics_epistemic_advanced_smoke' /tmp/graphics_epistemic_advanced_smoke.svg
fi

if [[ -f "${ROOT_DIR}/stdlib/graphics/view.sio" ]]; then
  echo "--> check graphics::view"
  "${SOUC}" check "${ROOT_DIR}/stdlib/graphics/view.sio"

  echo "--> check graphics_epistemic_view_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_epistemic_view_smoke.sio"

  echo "--> compile + run graphics_epistemic_view_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_epistemic_view_smoke.sio" -o /tmp/graphics_epistemic_view_smoke
  /tmp/graphics_epistemic_view_smoke > /tmp/graphics_epistemic_view_smoke.txt
  grep -q 'PASS graphics_epistemic_view_smoke' /tmp/graphics_epistemic_view_smoke.txt
fi

if [[ -f "${ROOT_DIR}/stdlib/graphics/tile.sio" ]]; then
  echo "--> check graphics_tile_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_tile_smoke.sio"

  echo "--> compile + run graphics_tile_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_tile_smoke.sio" -o /tmp/graphics_tile_smoke
  /tmp/graphics_tile_smoke

  echo "--> check graphics_tiled_ppm_export_smoke"
  "${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_tiled_ppm_export_smoke.sio"

  echo "--> compile + verify graphics_tiled_ppm_export_smoke"
  "${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_tiled_ppm_export_smoke.sio" -o /tmp/graphics_tiled_ppm_export_smoke
  /tmp/graphics_tiled_ppm_export_smoke > /tmp/graphics_tiled_ppm_export_smoke.ppm
  awk '
    $1 == "P3" { next }
    {
      for (i = 1; i <= NF; i++) {
        n[++count] = $i
      }
    }
    END {
      if (count != 27) {
        printf("expected 27 numeric tiled PPM tokens, got %d\n", count) > "/dev/stderr"
        exit 1
      }
      if (n[1] != 4 || n[2] != 2 || n[3] != 255) {
        printf("unexpected tiled PPM header: %s %s %s\n", n[1], n[2], n[3]) > "/dev/stderr"
        exit 1
      }
      if (n[4] != 255 || n[5] != 0 || n[6] != 0) {
        print "pixel 0 should be red" > "/dev/stderr"
        exit 1
      }
      if (n[7] != 0 || n[8] != 255 || n[9] != 0) {
        print "pixel 1 should be green" > "/dev/stderr"
        exit 1
      }
      if (n[10] != 0 || n[11] != 0 || n[12] != 255) {
        print "pixel 2 should be blue" > "/dev/stderr"
        exit 1
      }
      if (n[13] != 255 || n[14] != 255 || n[15] != 255) {
        print "pixel 3 should be white" > "/dev/stderr"
        exit 1
      }
      for (i = 16; i <= 27; i++) {
        if (n[i] != 0) {
          print "bottom row should remain black" > "/dev/stderr"
          exit 1
        }
      }
    }
  ' /tmp/graphics_tiled_ppm_export_smoke.ppm
fi

echo "--> check graphics_ppm_export_smoke"
"${SOUC}" check "${ROOT_DIR}/tests/run-pass/graphics_ppm_export_smoke.sio"

echo "--> compile + verify graphics_ppm_export_smoke"
"${SOUC}" compile "${ROOT_DIR}/tests/run-pass/graphics_ppm_export_smoke.sio" -o /tmp/graphics_ppm_export_smoke
/tmp/graphics_ppm_export_smoke > /tmp/graphics_ppm_export_smoke.ppm
grep -q '^P3$' /tmp/graphics_ppm_export_smoke.ppm
grep -q '^255$' /tmp/graphics_ppm_export_smoke.ppm
grep -q '^ 2$' /tmp/graphics_ppm_export_smoke.ppm
awk '
  $1 == "P3" { next }
  {
    for (i = 1; i <= NF; i++) {
      n[++count] = $i
    }
  }
  END {
    if (count != 15) {
      printf("expected 15 numeric PPM tokens, got %d\n", count) > "/dev/stderr"
      exit 1
    }
    if (n[1] != 2 || n[2] != 2 || n[3] != 255) {
      printf("unexpected PPM header: %s %s %s\n", n[1], n[2], n[3]) > "/dev/stderr"
      exit 1
    }
    if (n[4] != 0 || n[5] != 0 || n[6] != 0) {
      print "pixel 0 should be black" > "/dev/stderr"
      exit 1
    }
    if (n[7] != 255 || n[8] != 0 || n[9] != 0) {
      print "pixel 1 should be red" > "/dev/stderr"
      exit 1
    }
    if (n[10] != 0 || n[11] != 0 || n[12] != 0 || n[13] != 0 || n[14] != 0 || n[15] != 0) {
      print "pixels 2 and 3 should be black" > "/dev/stderr"
      exit 1
    }
  }
' /tmp/graphics_ppm_export_smoke.ppm

echo "=== Graphics Companion Gate: PASS ==="
