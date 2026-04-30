#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_QDEFORM_DIR:-$(mktemp -d /tmp/sounio-native-v2-qdeform.XXXXXX)}"

echo "[native-v2-qdeform] out=$OUT_DIR"
echo "[native-v2-qdeform] Door Ξ: q-deformed Arithmetic / Quantum Groups at IR Level"
echo "[native-v2-qdeform] U_q(sl₂): [E,F]_q = (q^H - q^{-H})/(q - q^{-1}); at q=1 → classical"
echo "[native-v2-qdeform] At q=e^{2πi/30}: E₈ WZW conformal field theory at level 1"
echo "[native-v2-qdeform] q=1 fast path: 1 VMULPD; q≠1: VMULPD + VFMADD231PD correction"
echo "[native-v2-qdeform] First compiler where q-deformation is an IR primitive"

MANIFEST_FILE="$(mktemp /tmp/qdeform-manifest.XXXXXX.tsv)"
cat >"$MANIFEST_FILE" <<'EOF'
# case_id	program	expected_exit	expected_stdout	claim_class
q_deform_arithmetic	tests/native-v2/science_spine/q_deform_arithmetic.sio	0	tests/native-v2/science_spine/q_deform_arithmetic.stdout	door_xi_q_deform
EOF

SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="$MANIFEST_FILE" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

echo "[native-v2-qdeform] PASS: q-deformed arithmetic gate"
