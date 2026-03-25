#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0
FAIL=0
NOT_RUN=0
TOTAL=0
SOUC="./bin/souc"

check_grep() {
    local name="$1"
    local pattern="$2"
    local file="$3"
    TOTAL=$((TOTAL + 1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file '$file' not found)"
        NOT_RUN=$((NOT_RUN + 1))
        return
    fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name (pattern '$pattern' not found in $file)"
        FAIL=$((FAIL + 1))
    fi
}

check_run() {
    local name="$1"
    local command="$2"
    TOTAL=$((TOTAL + 1))
    if eval "$command" >/dev/null 2>&1; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name"
        FAIL=$((FAIL + 1))
    fi
}

check_run_contains() {
    local name="$1"
    local file="$2"
    local needle="$3"
    TOTAL=$((TOTAL + 1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc not executable)"
        NOT_RUN=$((NOT_RUN + 1))
        return
    fi

    local out
    out="$(mktemp)"
    if timeout 60 "$SOUC" run "$file" >"$out" 2>&1; then
        if grep -qF "$needle" "$out"; then
            echo "PASS  $name"
            PASS=$((PASS + 1))
        else
            echo "FAIL  $name (missing '$needle')"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL  $name (run failed)"
        FAIL=$((FAIL + 1))
    fi
    rm -f "$out"
}

echo "=== Sprint 54: GPU Scene Contract ==="
echo ""

echo "--- Group 1: souc check + source contract ---"
check_run "check:ptx" "$SOUC check self-hosted/gpu/ptx_render.sio"
check_run "check:metal" "$SOUC check self-hosted/gpu/metal_render.sio"
check_run "check:spirv" "$SOUC check self-hosted/gpu/spirv_render.sio"
check_run "check:wgsl" "$SOUC check self-hosted/gpu/wgsl_render.sio"
check_grep "ptx:RenderScene" "struct RenderScene" "self-hosted/gpu/ptx_render.sio"
check_grep "metal:RenderScene" "struct RenderScene" "self-hosted/gpu/metal_render.sio"
check_grep "spirv:RenderScene" "struct RenderScene" "self-hosted/gpu/spirv_render.sio"
check_grep "wgsl:RenderScene" "struct RenderScene" "self-hosted/gpu/wgsl_render.sio"
check_grep "ptx:render_scene_default" "fn render_scene_default" "self-hosted/gpu/ptx_render.sio"
check_grep "metal:render_scene_default" "fn render_scene_default" "self-hosted/gpu/metal_render.sio"
check_grep "spirv:render_scene_default" "fn render_scene_default" "self-hosted/gpu/spirv_render.sio"
check_grep "wgsl:render_scene_default" "fn render_scene_default" "self-hosted/gpu/wgsl_render.sio"

echo ""
echo "--- Group 2: runtime scene-backed output ---"
check_run_contains "ptx:version_marker" "self-hosted/gpu/ptx_render.sio" ".version 7.5"
check_run_contains "ptx:scene_constant" "self-hosted/gpu/ptx_render.sio" "SCENE_V1X_MILLI = -760"
check_run_contains "metal:stdlib_marker" "self-hosted/gpu/metal_render.sio" "#include <metal_stdlib>"
check_run_contains "metal:scene_constant" "self-hosted/gpu/metal_render.sio" "constant int SCENE_V1X_MILLI = -760;"
check_run_contains "metal:scene_banner" "self-hosted/gpu/metal_render.sio" "// Sounio scene: triangle_count=1"
check_run_contains "wgsl:vertex_marker" "self-hosted/gpu/wgsl_render.sio" "@vertex"
check_run_contains "wgsl:scene_constant" "self-hosted/gpu/wgsl_render.sio" "const SCENE_V1X_MILLI: i32 = -760;"
check_run_contains "wgsl:scene_banner" "self-hosted/gpu/wgsl_render.sio" "// Sounio scene: 1 triangle"
check_run_contains "spirv:scene_signature" "self-hosted/gpu/spirv_render.sio" "// scene milli signature: 720 -760 760"
check_run_contains "spirv:vertex_dump" "self-hosted/gpu/spirv_render.sio" "vertex shader scene words:"
check_run_contains "spirv:scene_constant_opcode" "self-hosted/gpu/spirv_render.sio" "262187"

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"
REASON="all_cases_passed"
if [ "$FAIL" -gt 0 ]; then
    STATUS="fail"
    REASON="${FAIL}_cases_failed"
fi
if [ "$NOT_RUN" -gt 0 ] && [ "$FAIL" -eq 0 ]; then
    REASON="all_run_passed_${NOT_RUN}_not_run"
fi

mkdir -p "$ROOT_DIR/artifacts/sprint54"
cat > "$ROOT_DIR/artifacts/sprint54/gpu_contract_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint54.gpu_contract_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR",
    "timeout_seconds": 60
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "novel_claims": [
    "All standalone GPU emitters expose a shared RenderScene contract and a default triangle scene",
    "PTX, Metal, WGSL, and SPIR-V outputs now embed milli-scaled scene constants derived from Sounio render data",
    "Scene metadata changes backend output text or words instead of being limited to comments or static boilerplate",
    "Platform regression markers remain present while the scene contract stays additive"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint54/gpu_contract_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
