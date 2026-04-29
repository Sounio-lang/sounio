#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_QDEFORM_OCT_DIR:-$(mktemp -d /tmp/sounio-native-v2-qdeform-oct.XXXXXX)}"

echo "[native-v2-qdeform-oct] out=$OUT_DIR"
echo "[native-v2-qdeform-oct] Door Ξ extended: q-deformed arithmetic with non-zero commutator"
echo "[native-v2-qdeform-oct] Octonion anti-commutativity: [e₁,e₂] = 2·e₃ (known commutator)"
echo "[native-v2-qdeform-oct] q=2: a·_2 b = a·b + (q−1)·φ = 1.0 + 1.0·2.0 = 3.0 (VFMADD path)"
echo "[native-v2-qdeform-oct] Contrast with scalar test: φ_q=0 for commuting scalars"
echo "[native-v2-qdeform-oct] Connection: q=e^{2πi/30} + IrE8RootReflectO = E₈ WZW level-1"

MANIFEST_FILE="$(mktemp /tmp/qdeform-oct-manifest.XXXXXX.tsv)"
cat >"$MANIFEST_FILE" <<'EOF'
# case_id	program	expected_exit	expected_stdout	claim_class
q_deform_octonion	tests/native-v2/science_spine/q_deform_octonion.sio	0	tests/native-v2/science_spine/q_deform_octonion.stdout	door_xi_q_deform_noncomm
EOF

SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="$MANIFEST_FILE" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

echo "[native-v2-qdeform-oct] PASS: q-deformed arithmetic non-commutative gate"
