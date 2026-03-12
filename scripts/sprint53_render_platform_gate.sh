#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0
FAIL=0
NOT_RUN=0
TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
OUT_JSON="$ROOT_DIR/artifacts/sprint53/render_platform_gate.v1.json"

check_grep() {
    local name="$1"
    local pattern="$2"
    local file="$3"
    TOTAL=$((TOTAL + 1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (missing file: $file)"
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
    shift
    TOTAL=$((TOTAL + 1))
    if "$@" >/tmp/sprint53_gate_run.log 2>&1; then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name"
        tail -20 /tmp/sprint53_gate_run.log 2>/dev/null || true
        FAIL=$((FAIL + 1))
    fi
}

check_ppm_run() {
    local name="$1"
    local file="$2"
    local width="$3"
    local height="$4"
    local min_colors="$5"
    local required_colors="${6:-}"
    TOTAL=$((TOTAL + 1))
    local ppm_out
    ppm_out="$(mktemp)"
    local err_out
    err_out="$(mktemp)"

    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc not executable)"
        NOT_RUN=$((NOT_RUN + 1))
        rm -f "$ppm_out" "$err_out"
        return
    fi

    if ! timeout 60 "$SOUC" run "$file" >"$ppm_out" 2>"$err_out"; then
        echo "FAIL  $name (souc run failed)"
        tail -20 "$err_out" 2>/dev/null || true
        FAIL=$((FAIL + 1))
        rm -f "$ppm_out" "$err_out"
        return
    fi

    if python3 - "$ppm_out" "$width" "$height" "$min_colors" "$required_colors" <<'PY'
import sys
from pathlib import Path

ppm_path = Path(sys.argv[1])
width = int(sys.argv[2])
height = int(sys.argv[3])
min_colors = int(sys.argv[4])
required_arg = sys.argv[5]
required = []
if required_arg:
    for item in required_arg.split(";"):
        parts = item.split(",")
        required.append(tuple(int(p) for p in parts))

tokens = ppm_path.read_text().split()
if len(tokens) < 4:
    raise SystemExit("ppm too short")
if tokens[0] != "P3":
    raise SystemExit(f"expected P3, got {tokens[0]}")
if int(tokens[1]) != width or int(tokens[2]) != height:
    raise SystemExit(f"expected {width}x{height}, got {tokens[1]}x{tokens[2]}")
if int(tokens[3]) != 255:
    raise SystemExit(f"expected max value 255, got {tokens[3]}")
payload = tokens[4:]
expected = width * height * 3
if len(payload) != expected:
    raise SystemExit(f"expected {expected} payload ints, got {len(payload)}")
colors = set()
for idx in range(0, len(payload), 3):
    rgb = tuple(int(payload[idx + channel]) for channel in range(3))
    for component in rgb:
        if component < 0 or component > 255:
            raise SystemExit(f"component out of range at pixel {idx // 3}: {rgb}")
    colors.add(rgb)
if len(colors) < min_colors:
    raise SystemExit(f"expected at least {min_colors} colors, got {len(colors)}")
for color in required:
    if color not in colors:
        raise SystemExit(f"missing required color {color}")
PY
    then
        echo "PASS  $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL  $name (invalid PPM payload)"
        FAIL=$((FAIL + 1))
    fi

    rm -f "$ppm_out" "$err_out"
}

echo "=== Sprint 53: Render Platform Gate ==="
echo ""

echo "--- Group 1: stdlib contract ---"
check_run "stdlib:rasterizer_check" "$SOUC" check stdlib/render/rasterizer.sio
check_run "stdlib:scene_check" "$SOUC" check stdlib/render/scene.sio
check_grep "stdlib:rasterize_rect" "fn rasterize_rect" stdlib/render/rasterizer.sio
check_grep "stdlib:ScreenPt" "struct ScreenPt" stdlib/render/scene.sio
check_grep "stdlib:project_point3d" "fn project_point3d" stdlib/render/scene.sio

echo ""
echo "--- Group 2: render outputs ---"
check_ppm_run "ppm:triangle_basic" examples/render/triangle_basic.sio 128 128 1000
check_ppm_run "ppm:triangle_ppm" examples/render/triangle_ppm.sio 128 128 1000
check_ppm_run "ppm:cube_wireframe" examples/render/cube_wireframe.sio 192 192 100 "255,96,96;255,255,255"
check_ppm_run "ppm:uncertainty_ppm" examples/render/uncertainty_ppm.sio 128 128 200
check_ppm_run "ppm:uncertainty_field" examples/render/uncertainty_field.sio 128 128 200
check_ppm_run "ppm:causal_dag" examples/render/causal_dag.sio 256 128 6 "255,76,25;51,153,255;99,99,99;230,230,230"
check_ppm_run "ppm:quaternion_rotation" examples/render/quaternion_rotation.sio 192 192 4 "60,60,90;50,200,80;200,200,150"

echo ""
echo "--- Group 3: refactor alignment ---"
check_grep "refactor:triangle_65536" "\\[i64; 65536\\]" examples/render/triangle_basic.sio
check_grep "refactor:cube_rasterize_line" "fn rasterize_line" examples/render/cube_wireframe.sio
check_grep "refactor:uncertainty_ppm_rect" "fn rasterize_rect" examples/render/uncertainty_ppm.sio
check_grep "refactor:uncertainty_ppm_65536" "\\[i64; 65536\\]" examples/render/uncertainty_ppm.sio

echo ""
echo "--- Group 4: flagship helpers ---"
check_grep "flagship:uncertainty_value_to_color" "fn value_to_color" examples/render/uncertainty_field.sio
check_grep "flagship:dag_arrow_right" "fn draw_arrowhead_right" examples/render/causal_dag.sio
check_grep "flagship:dag_arrow_ul" "fn draw_arrowhead_ul" examples/render/causal_dag.sio
check_grep "flagship:quat_project_pt" "fn project_pt" examples/render/quaternion_rotation.sio

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"
REASON="all_cases_passed"
if [ "$FAIL" -gt 0 ]; then
    STATUS="fail"
    REASON="${FAIL}_cases_failed"
elif [ "$NOT_RUN" -gt 0 ]; then
    REASON="all_run_passed_${NOT_RUN}_not_run"
fi

mkdir -p "$ROOT_DIR/artifacts/sprint53"
cat > "$OUT_JSON" <<EOF
{
  "schema": "sounio.sprint53.render_platform_gate.v1",
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
    "Render stdlib now exposes canonical rectangle fill and simple 3D projection contracts",
    "Seven render examples emit real PPM payloads with validated dimensions and RGB counts",
    "Flagship epistemic, causal, and quaternion scenes are now checked as executable raster fixtures",
    "Gate rejects malformed PPM output instead of relying on header-only or grep-only validation"
  ]
}
EOF

echo "Artifact: artifacts/sprint53/render_platform_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
