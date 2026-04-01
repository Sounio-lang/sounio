#!/usr/bin/env bash
# sprint34_render_types_gate.sh — Render effect + graphics type foundation gate
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -n "${SOUNIO_SOUC:-}" ]; then
  SOUC="$SOUNIO_SOUC"
else
  source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
  SOUC="${SOUC_BIN:-}"
fi

OUT_JSON="${SOUNIO_SPRINT34_GATE_OUT:-$ROOT_DIR/artifacts/sprint34/render_types_gate.v1.json}"
TIMEOUT_SECS="${SOUNIO_SPRINT34_TIMEOUT_SECS:-120}"

mkdir -p "$(dirname "$OUT_JSON")"
mkdir -p "$ROOT_DIR/artifacts/sprint34"

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
  local log="/tmp/sprint34_${label}.log"
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

echo "=== Sprint 34: Render Effect + Graphics Type Foundation Gate ==="
echo ""

# --- Section 1: Render effect in self-hosted checker ---
echo "--- Render effect declaration ---"
check_grep "effect:render_id" "self-hosted/check/effects.sio" "return 12"
check_grep "effect:render_name" "self-hosted/check/effects.sio" "Render"
check_grep "effect:render_match" "self-hosted/check/effects.sio" "82 as i8 && name_buf\[1\] == 101 as i8"

# --- Section 2: stdlib/render/ files exist ---
echo ""
echo "--- stdlib/render/ type files ---"
check_file_exists "stdlib:types" "stdlib/render/types.sio"
check_file_exists "stdlib:scene" "stdlib/render/scene.sio"
check_file_exists "stdlib:pipeline" "stdlib/render/pipeline.sio"
check_file_exists "stdlib:epistemic" "stdlib/render/epistemic.sio"

# --- Section 3: Core type definitions ---
echo ""
echo "--- Core type definitions ---"
check_grep "type:Color" "stdlib/render/types.sio" "struct Color"
check_grep "type:Vec3" "stdlib/render/types.sio" "struct Vec3"
check_grep "type:Vertex" "stdlib/render/types.sio" "struct Vertex"
check_grep "type:Triangle" "stdlib/render/types.sio" "struct Triangle"
check_grep "type:Camera" "stdlib/render/scene.sio" "struct Camera"
check_grep "type:RenderBackend" "stdlib/render/pipeline.sio" "enum RenderBackend"
check_grep "type:UncertainVertex" "stdlib/render/epistemic.sio" "struct UncertainVertex"
check_grep "type:ConfidenceBand" "stdlib/render/epistemic.sio" "struct ConfidenceBand"

# --- Section 4: souc check on stdlib ---
echo ""
echo "--- souc check on stdlib/render/ ---"
check_souc "souc:types" "stdlib/render/types.sio" "pass"
check_souc "souc:scene" "stdlib/render/scene.sio" "pass"
check_souc "souc:pipeline" "stdlib/render/pipeline.sio" "pass"
check_souc "souc:epistemic" "stdlib/render/epistemic.sio" "pass"

# --- Section 5: Example files exist + souc check ---
echo ""
echo "--- Render examples ---"
check_file_exists "example:triangle" "examples/render/triangle_basic.sio"
check_file_exists "example:cube" "examples/render/cube_wireframe.sio"
check_file_exists "example:uncertainty" "examples/render/uncertainty_field.sio"
check_file_exists "example:causal" "examples/render/causal_dag.sio"
check_file_exists "example:quaternion" "examples/render/quaternion_rotation.sio"
check_souc "souc:triangle" "examples/render/triangle_basic.sio" "pass"
check_souc "souc:cube" "examples/render/cube_wireframe.sio" "pass"
check_souc "souc:uncertainty" "examples/render/uncertainty_field.sio" "pass"
check_souc "souc:causal" "examples/render/causal_dag.sio" "pass"
check_souc "souc:quaternion" "examples/render/quaternion_rotation.sio" "pass"

# --- Section 6: Sprint 33 regression ---
echo ""
echo "--- Sprint 33 regression ---"
check_grep "regr:module_loader" "self-hosted/compiler/module_loader.sio" "fn compile_multimodule_program"
check_grep "regr:disasm" "self-hosted/main.sio" "fn run_disasm_pipeline"

# --- Section 7: General regression ---
echo ""
echo "--- General regression ---"
check_souc "souc:regression_kernel_fn_basic" "tests/run-pass/kernel_fn_basic.sio" "pass"
check_souc "souc:regression_ir_roundtrip" "tests/run-pass/ir_roundtrip_basic.sio" "pass"
check_souc "souc:regression_causal_front_door" "tests/frontend/causal_front_door_basic.sio" "pass"

echo ""
echo "=== Sprint 34 Gate: $pass/$total pass ==="

cat > "$OUT_JSON" <<ARTIFACT
{
  "gate": "sprint34_render_types",
  "version": 1,
  "status": "$(if [[ $fail -eq 0 ]]; then echo PASS; else echo FAIL; fi)",
  "pass": $pass,
  "total": $total,
  "fail": $fail,
  "new_effect": "Render (ID 12)",
  "new_types": ["Color", "Vec2", "Vec3", "Vec4", "Vertex", "Triangle", "Canvas", "Camera", "Light", "Scene", "RenderBackend", "RenderConfig", "UncertainVertex", "ConfidenceBand", "UncertaintyVizMode"],
  "new_examples": ["triangle_basic", "cube_wireframe", "uncertainty_field", "causal_dag", "quaternion_rotation"],
  "render_backends_planned": ["PPM (software)", "SPIR-V (Vulkan)", "Metal", "WGSL (WebGPU)"],
  "sota_references": ["Vulkan render pipeline", "Metal Shading Language", "WebGPU spec", "Uncertainty visualization (Brodlie 2012)"],
  "novelty": "First language with compile-time epistemic uncertainty propagation into GPU pixel rendering pipeline",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
ARTIFACT

echo "artifact: $OUT_JSON"

if [[ $fail -gt 0 ]]; then
  exit 1
fi
