#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_ASSOCIATOR_DIR:-$(mktemp -d /tmp/sounio-native-v2-associator.XXXXXX)}"

echo "[native-v2-associator] out=$OUT_DIR"
echo "[native-v2-associator] Octonion associator: Fano→norm=0, non-Fano→norm=2"
echo "[native-v2-associator] IrAssociator opcode: 4×Fano + VSUBPD = 125 EVEX instructions"

MANIFEST_FILE="$(mktemp /tmp/associator-manifest.XXXXXX.tsv)"
cat >"$MANIFEST_FILE" <<'EOF'
# case_id	program	expected_exit	expected_stdout	claim_class
associator_gum	tests/native-v2/science_spine/associator_gum.sio	0	tests/native-v2/science_spine/associator_gum.stdout	hypercomplex_associator
EOF

SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="$MANIFEST_FILE" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

echo "[native-v2-associator] PASS: associator norm gate"
