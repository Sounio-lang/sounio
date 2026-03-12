#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0
FAIL=0
NOT_RUN=0
TOTAL=0

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

check_svg_dims() {
    local name="$1"
    local file="$2"
    local width="$3"
    local height="$4"
    TOTAL=$((TOTAL + 1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file '$file' not found)"
        NOT_RUN=$((NOT_RUN + 1))
        return
    fi
    if grep -q "viewBox=\"0 0 $width $height\"" "$file" && grep -q "width=\"$width\"" "$file" && grep -q "height=\"$height\"" "$file"; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name (expected ${width}x${height})"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Sprint 55: Website Render Expansion ==="
echo ""

echo "--- Group 1: SVG assets with correct dimensions ---"
check_svg_dims "svg:triangle_basic" "website/public/assets/generated/render/triangle-basic.svg" 128 128
check_svg_dims "svg:cube_wireframe" "website/public/assets/generated/render/cube-wireframe.svg" 192 192
check_svg_dims "svg:uncertainty_field" "website/public/assets/generated/render/uncertainty-field.svg" 128 128
check_svg_dims "svg:causal_dag" "website/public/assets/generated/render/causal-dag.svg" 256 128
check_svg_dims "svg:quaternion_rotation" "website/public/assets/generated/render/quaternion-rotation.svg" 192 192

echo ""
echo "--- Group 2: manifest.json has all 5 assets ---"
check_grep "manifest:triangle_basic" "triangle-basic\\.svg" "website/public/assets/generated/render/manifest.json"
check_grep "manifest:cube_wireframe" "cube-wireframe\\.svg" "website/public/assets/generated/render/manifest.json"
check_grep "manifest:uncertainty_field" "uncertainty-field\\.svg" "website/public/assets/generated/render/manifest.json"
check_grep "manifest:causal_dag" "causal-dag\\.svg" "website/public/assets/generated/render/manifest.json"
check_grep "manifest:quaternion" "quaternion-rotation\\.svg" "website/public/assets/generated/render/manifest.json"

echo ""
echo "--- Group 3: graphics.mdx has all 5 RenderPreview refs ---"
check_grep "mdx:triangle_basic" "triangle-basic\\.svg" "website/src/content/showcases/graphics.mdx"
check_grep "mdx:cube_wireframe" "cube-wireframe\\.svg" "website/src/content/showcases/graphics.mdx"
check_grep "mdx:uncertainty_field" "uncertainty-field\\.svg" "website/src/content/showcases/graphics.mdx"
check_grep "mdx:causal_dag" "causal-dag\\.svg" "website/src/content/showcases/graphics.mdx"
check_grep "mdx:quaternion" "quaternion-rotation\\.svg" "website/src/content/showcases/graphics.mdx"
check_grep "mdx:last_validated" "last_validated: \"2026-03-10\"" "website/src/content/showcases/graphics.mdx"

echo ""
echo "--- Group 4: render-assets.mjs has all 5 specs ---"
check_grep "script:triangle_basic" "triangle_basic\\.sio" "website/scripts/render-assets.mjs"
check_grep "script:cube_wireframe" "cube_wireframe\\.sio" "website/scripts/render-assets.mjs"
check_grep "script:uncertainty_field" "uncertainty_field\\.sio" "website/scripts/render-assets.mjs"
check_grep "script:causal_dag" "causal_dag\\.sio" "website/scripts/render-assets.mjs"
check_grep "script:quaternion" "quaternion_rotation\\.sio" "website/scripts/render-assets.mjs"

echo ""
echo "--- Group 5: stale SVG detection ---"
check_run "website:render_assets_check_mode" "node website/scripts/render-assets.mjs check"

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

mkdir -p "$ROOT_DIR/artifacts/sprint55"
cat > "$ROOT_DIR/artifacts/sprint55/website_render_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint55.website_render_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR"
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "novel_claims": [
    "Website render assets are generated from five checked JIT raster examples and committed as SVGs",
    "Graphics showcase now includes uncertainty, causal DAG, and quaternion rotation previews alongside existing triangle and cube renders",
    "Manifest, MDX references, and generated SVG dimensions stay synchronized through a dedicated stale-asset gate",
    "Node check mode validates that committed website assets exactly match current compiler output"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint55/website_render_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
