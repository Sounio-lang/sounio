#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_SED_ZD_DIR:-$(mktemp -d /tmp/sounio-native-v2-sed-zd.XXXXXX)}"

echo "[native-v2-sed-zd] out=$OUT_DIR"
echo "[native-v2-sed-zd] Sedenion ZD gate: canonical pair (e₃+e₁₀, e₆-e₁₅) product = 0"
echo "[native-v2-sed-zd] IrSedZDCheck: first compile-time ZD detection node (84 pairs, Cawagas 2004)"
echo "[native-v2-sed-zd] Manifold: ZD pairs ≅ G₂ (arxiv:2411.18881)"

MANIFEST_FILE="$(mktemp /tmp/sed-zd-manifest.XXXXXX.tsv)"
cat >"$MANIFEST_FILE" <<'EOF'
# case_id	program	expected_exit	expected_stdout	claim_class
sedenion_zd_static	tests/native-v2/science_spine/sedenion_zd_static.sio	0	tests/native-v2/science_spine/sedenion_zd_static.stdout	hypercomplex_zd_gate
EOF

SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="$MANIFEST_FILE" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

echo "[native-v2-sed-zd] PASS: sedenion zero-divisor static gate"
