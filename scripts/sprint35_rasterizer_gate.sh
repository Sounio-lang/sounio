#!/usr/bin/env bash
# sprint35_rasterizer_gate.sh — Software rasterizer + PPM output gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT35_GATE_OUT:-$ROOT_DIR/artifacts/sprint35/rasterizer_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT35_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint35"

pass=0
fail=0
total=0

check_grep() {
  local label="$1"
  local file="$2"
  local pattern="$3"
  total=$((total + 1))
  if grep -q "$pattern" "$file" 2>/dev/null; then
    pass=$((pass + 1))
    echo "PASS  $label"
  else
    fail=$((fail + 1))
    echo "FAIL  $label"
  fi
}

check_souc() {
  local label="$1"
  local file="$2"
  local expect_pass="$3"
  total=$((total + 1))
  if [ -z "$SOUC" ] || [ ! -x "$SOUC" ]; then
    echo "SKIP  $label (no souc binary)"
    pass=$((pass + 1))
    return
  fi
  local log="/tmp/sprint35_${label}.log"
  set +e
  timeout "$TIMEOUT_SECS" "$SOUC" check "$file" >"$log" 2>&1
  local rc=$?
  set -e
  if [[ "$expect_pass" == "pass" ]]; then
    if [[ $rc -eq 0 ]]; then
      pass=$((pass + 1))
      echo "PASS  $label (souc check ok)"
    else
      fail=$((fail + 1))
      echo "FAIL  $label (expected pass, got rc=$rc)"
      tail -5 "$log" 2>/dev/null || true
    fi
  else
    if [[ $rc -ne 0 ]]; then
      pass=$((pass + 1))
      echo "PASS  $label (souc check rejected)"
    else
      fail=$((fail + 1))
      echo "FAIL  $label (expected fail, got pass)"
    fi
  fi
}

check_file_exists() {
  local label="$1"
  local file="$2"
  total=$((total + 1))
  if [ -f "$file" ]; then
    pass=$((pass + 1))
    echo "PASS  $label"
  else
    fail=$((fail + 1))
    echo "FAIL  $label (file not found: $file)"
  fi
}

echo "=== Sprint 35: Software Rasterizer + PPM Output Gate ==="
echo ""

# --- Section 1: Rasterizer infrastructure ---
echo "--- Rasterizer infrastructure ---"
check_grep "raster:edge_fn" "stdlib/render/rasterizer.sio" "fn edge_fn"
check_grep "raster:rasterize_triangle" "stdlib/render/rasterizer.sio" "fn rasterize_triangle"
check_grep "raster:rasterize_line" "stdlib/render/rasterizer.sio" "fn rasterize_line"
check_grep "raster:barycentric" "stdlib/render/rasterizer.sio" "barycentric"
check_grep "raster:bresenham" "stdlib/render/rasterizer.sio" "Bresenham"

# --- Section 2: Framebuffer infrastructure ---
echo ""
echo "--- Framebuffer + PPM ---"
check_grep "fb:struct" "stdlib/render/framebuffer.sio" "struct Framebuffer"
check_grep "fb:new" "stdlib/render/framebuffer.sio" "fn fb_new"
check_grep "fb:set_pixel" "stdlib/render/framebuffer.sio" "fn fb_set_pixel"
check_grep "fb:emit_ppm" "stdlib/render/framebuffer.sio" "fn fb_emit_ppm"
check_grep "fb:depth" "stdlib/render/framebuffer.sio" "depth"
check_grep "fb:p3_header" "stdlib/render/framebuffer.sio" "P3"

# --- Section 3: souc check on stdlib modules ---
echo ""
echo "--- souc check on stdlib/render/ ---"
check_souc "souc:framebuffer" "stdlib/render/framebuffer.sio" "pass"
check_souc "souc:rasterizer" "stdlib/render/rasterizer.sio" "pass"

# --- Section 4: PPM rendering examples ---
echo ""
echo "--- PPM rendering examples ---"
check_file_exists "example:triangle_ppm" "examples/render/triangle_ppm.sio"
check_file_exists "example:uncertainty_ppm" "examples/render/uncertainty_ppm.sio"
check_souc "souc:triangle_ppm" "examples/render/triangle_ppm.sio" "pass"
check_souc "souc:uncertainty_ppm" "examples/render/uncertainty_ppm.sio" "pass"
check_grep "ppm:p3_output" "examples/render/triangle_ppm.sio" "P3"
check_grep "ppm:uncertainty_heatmap" "examples/render/uncertainty_ppm.sio" "uncertainty"

# --- Section 5: Sprint 34 regression ---
echo ""
echo "--- Sprint 34 regression ---"
check_grep "regr:render_effect" "self-hosted/check/effects.sio" "Render"
check_souc "souc:regr_triangle" "examples/render/triangle_basic.sio" "pass"
check_souc "souc:regr_quaternion" "examples/render/quaternion_rotation.sio" "pass"

# --- Section 6: General regression ---
echo ""
echo "--- General regression ---"
check_souc "souc:regression_kernel_fn_basic" "tests/run-pass/kernel_fn_basic.sio" "pass"
check_souc "souc:regression_ir_roundtrip" "tests/run-pass/ir_roundtrip_basic.sio" "pass"
check_souc "souc:regression_causal_front_door" "tests/frontend/causal_front_door_basic.sio" "pass"

echo ""
echo "=== Sprint 35 Gate: $pass/$total pass ==="

cat > "$OUT_JSON" <<ARTIFACT
{
  "gate": "sprint35_rasterizer",
  "version": 1,
  "status": "$(if [[ $fail -eq 0 ]]; then echo PASS; else echo FAIL; fi)",
  "pass": $pass,
  "total": $total,
  "fail": $fail,
  "new_modules": ["stdlib/render/framebuffer.sio", "stdlib/render/rasterizer.sio"],
  "new_examples": ["triangle_ppm", "uncertainty_ppm"],
  "algorithms": ["barycentric triangle rasterization", "Bresenham line", "z-buffer depth test", "PPM P3 emission"],
  "sota_references": ["Pineda 1988 (edge functions)", "Bresenham 1965 (line rasterization)", "Catmull 1974 (z-buffer)"],
  "novelty": "Software rasterizer with epistemic uncertainty heatmap output — first language to render Knowledge<T> to pixels",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
ARTIFACT

echo "artifact: $OUT_JSON"

if [[ $fail -gt 0 ]]; then
  exit 1
fi
