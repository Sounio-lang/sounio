#!/usr/bin/env bash
# sprint36_gpu_render_gate.sh — GPU render pipeline (SPIR-V/Metal/WGSL/PTX) gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT36_GATE_OUT:-$ROOT_DIR/artifacts/sprint36/gpu_render_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT36_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint36"

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
  total=$((total + 1))
  if timeout "$TIMEOUT_SECS" "$SOUC" check "$file" 2>/dev/null | grep -q "All checks passed"; then
    pass=$((pass + 1))
    echo "PASS  $label"
  else
    fail=$((fail + 1))
    echo "FAIL  $label"
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
    echo "FAIL  $label"
  fi
}

echo "=== Sprint 36: GPU Render Pipeline Gate ==="
echo ""

# ── Section 1: Files exist ───────────────────────────────────────────────────
echo "--- Section 1: GPU render files exist ---"
check_file_exists "spirv_render.sio exists"    "self-hosted/gpu/spirv_render.sio"
check_file_exists "metal_render.sio exists"    "self-hosted/gpu/metal_render.sio"
check_file_exists "wgsl_render.sio exists"     "self-hosted/gpu/wgsl_render.sio"
check_file_exists "ptx_render.sio exists"      "self-hosted/gpu/ptx_render.sio"

# ── Section 2: SPIR-V render content checks ───────────────────────────────────
echo ""
echo "--- Section 2: SPIR-V render correctness ---"
check_grep "SPV_EXEC_VERTEX defined"              "self-hosted/gpu/spirv_render.sio" "SPV_EXEC_VERTEX"
check_grep "SPV_EXEC_FRAGMENT defined"            "self-hosted/gpu/spirv_render.sio" "SPV_EXEC_FRAGMENT"
check_grep "SPV_BUILTIN_POSITION defined"         "self-hosted/gpu/spirv_render.sio" "SPV_BUILTIN_POSITION"
check_grep "spirv_emit_vertex_shader defined"     "self-hosted/gpu/spirv_render.sio" "fn spirv_emit_vertex_shader"
check_grep "spirv_emit_fragment_shader defined"   "self-hosted/gpu/spirv_render.sio" "fn spirv_emit_fragment_shader"
check_grep "rspv_emit_execution_mode defined"     "self-hosted/gpu/spirv_render.sio" "fn rspv_emit_execution_mode"
check_grep "OriginUpperLeft mode (7)"             "self-hosted/gpu/spirv_render.sio" "7.*OriginUpperLeft\|OriginUpperLeft.*7"
check_grep "Location decoration (30)"             "self-hosted/gpu/spirv_render.sio" "30"

# ── Section 3: spirv.sio gains Location + ExecutionMode ───────────────────────
echo ""
echo "--- Section 3: spirv.sio additions ---"
check_grep "DECORATION_LOCATION in spirv.sio"         "self-hosted/gpu/spirv.sio" "DECORATION_LOCATION"
check_grep "spv_emit_decorate_location in spirv.sio"  "self-hosted/gpu/spirv.sio" "fn spv_emit_decorate_location"
check_grep "spv_emit_execution_mode in spirv.sio"     "self-hosted/gpu/spirv.sio" "fn spv_emit_execution_mode"

# ── Section 4: Metal render content checks ────────────────────────────────────
echo ""
echo "--- Section 4: Metal render correctness ---"
check_grep "metal_emit_vertex_shader defined"   "self-hosted/gpu/metal_render.sio" "fn metal_emit_vertex_shader"
check_grep "metal_emit_fragment_shader defined" "self-hosted/gpu/metal_render.sio" "fn metal_emit_fragment_shader"
check_grep "metal stage_in attribute"           "self-hosted/gpu/metal_render.sio" "stage_in"
check_grep "metal vertex entry point"           "self-hosted/gpu/metal_render.sio" "vertex VertexOut vertex_main"
check_grep "metal fragment entry point"         "self-hosted/gpu/metal_render.sio" "fragment float4 fragment_main"

# ── Section 5: WGSL render content checks ─────────────────────────────────────
echo ""
echo "--- Section 5: WGSL render correctness ---"
check_grep "wgsl_emit_vertex_shader defined"   "self-hosted/gpu/wgsl_render.sio"  "fn wgsl_emit_vertex_shader"
check_grep "wgsl_emit_fragment_shader defined" "self-hosted/gpu/wgsl_render.sio"  "fn wgsl_emit_fragment_shader"
check_grep "WGSL @vertex annotation"           "self-hosted/gpu/wgsl_render.sio"  "@vertex"
check_grep "WGSL @fragment annotation"         "self-hosted/gpu/wgsl_render.sio"  "@fragment"
check_grep "WGSL builtin position"             "self-hosted/gpu/wgsl_render.sio"  "builtin(position)"

# ── Section 6: PTX render content checks ──────────────────────────────────────
echo ""
echo "--- Section 6: PTX render correctness ---"
check_grep "ptx_emit_fill_kernel defined" "self-hosted/gpu/ptx_render.sio" "fn ptx_emit_fill_kernel"
check_grep "PTX SM_75 target"             "self-hosted/gpu/ptx_render.sio" "sm_75"
check_grep "PTX framebuffer params"       "self-hosted/gpu/ptx_render.sio" "param_fb_r"
check_grep "PTX bounds check"             "self-hosted/gpu/ptx_render.sio" "setp.lt.u32"

# ── Section 7: souc check all 4 render files ─────────────────────────────────
echo ""
echo "--- Section 7: souc check all GPU render backends ---"
check_souc "souc check spirv_render.sio"  "self-hosted/gpu/spirv_render.sio"
check_souc "souc check metal_render.sio"  "self-hosted/gpu/metal_render.sio"
check_souc "souc check wgsl_render.sio"   "self-hosted/gpu/wgsl_render.sio"
check_souc "souc check ptx_render.sio"    "self-hosted/gpu/ptx_render.sio"

# ── Section 8: Sprint 35 regression ───────────────────────────────────────────
echo ""
echo "--- Section 8: Sprint 35 regression ---"
check_souc "souc check framebuffer.sio"   "stdlib/render/framebuffer.sio"
check_souc "souc check rasterizer.sio"    "stdlib/render/rasterizer.sio"
check_souc "souc check triangle_ppm.sio"  "examples/render/triangle_ppm.sio"
check_souc "souc check uncertainty_ppm"   "examples/render/uncertainty_ppm.sio"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Results: $pass/$total passed ==="

START_TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "unknown")
cat > "$OUT_JSON" <<EOF
{
  "sprint": 36,
  "gate": "gpu_render_gate",
  "version": "v1",
  "timestamp": "$START_TS",
  "pass": $pass,
  "fail": $fail,
  "total": $total,
  "result": "$([ $fail -eq 0 ] && echo PASS || echo FAIL)"
}
EOF

echo "Artifact: $OUT_JSON"

if [ $fail -gt 0 ]; then
  exit 1
fi
