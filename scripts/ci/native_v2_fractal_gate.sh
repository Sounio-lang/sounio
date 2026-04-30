#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_FRACTAL_DIR:-$(mktemp -d /tmp/sounio-native-v2-fractal.XXXXXX)}"

echo "[native-v2-fractal] out=$OUT_DIR"
echo "[native-v2-fractal] IrFractalEscape: first compiler primitive for hypercomplex fractal iteration"
echo "[native-v2-fractal] Test: quaternion Julia c=(-0.2,0.8,0,0), z0=(1.5,0,0,0) → escaped in 1 iter"
echo "[native-v2-fractal] Epistemic return: basin_confidence via ISO GUM from floating-point error"

MANIFEST_FILE="$(mktemp /tmp/fractal-manifest.XXXXXX.tsv)"
cat >"$MANIFEST_FILE" <<'EOF'
# case_id	program	expected_exit	expected_stdout	claim_class
fractal_quaternion_julia	tests/native-v2/science_spine/fractal_quaternion_julia.sio	0	tests/native-v2/science_spine/fractal_quaternion_julia.stdout	fractal_primitive
EOF

SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="$MANIFEST_FILE" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

echo "[native-v2-fractal] PASS: quaternion Julia escape gate"
