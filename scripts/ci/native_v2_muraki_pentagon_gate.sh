#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_MURAKI_DIR:-$(mktemp -d /tmp/sounio-native-v2-muraki.XXXXXX)}"

echo "[native-v2-muraki] out=$OUT_DIR"
echo "[native-v2-muraki] Door Δ: Muraki's Independence Pentagon (Boolean + Monotone)"
echo "[native-v2-muraki] Muraki 2003: exactly 5 universal non-commutative independence types"
echo "[native-v2-muraki] Boolean κ=1 (minimum) < Monotone κ=3/2 < Free κ=2 < Classical κ=3"
echo "[native-v2-muraki] First compiler where independence type is an IR-level distinction"

MANIFEST_FILE="$(mktemp /tmp/muraki-manifest.XXXXXX.tsv)"
cat >"$MANIFEST_FILE" <<'EOF'
# case_id	program	expected_exit	expected_stdout	claim_class
boolean_independence	tests/native-v2/science_spine/boolean_independence.sio	0	tests/native-v2/science_spine/boolean_independence.stdout	door_delta_muraki_pentagon
monotone_independence	tests/native-v2/science_spine/monotone_independence.sio	0	tests/native-v2/science_spine/monotone_independence.stdout	door_delta_muraki_pentagon
EOF

SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="$MANIFEST_FILE" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

echo "[native-v2-muraki] PASS: Muraki independence pentagon gate"
