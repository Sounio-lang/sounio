#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_WASSERSTEIN_DIR:-$(mktemp -d /tmp/sounio-native-v2-wasserstein.XXXXXX)}"

echo "[native-v2-wasserstein] out=$OUT_DIR"
echo "[native-v2-wasserstein] Door Φ: Wasserstein Distance + Ollivier-Ricci Curvature"
echo "[native-v2-wasserstein] W₁ = Earth Mover's Distance -- canonical metric on probability distributions"
echo "[native-v2-wasserstein] ORC κ(x,y) = 1 - W₁(mₓ,mᵧ)/d(x,y) -- curvature of graph geometry"
echo "[native-v2-wasserstein] Connection: Paper 4 -- ORC on ABIDE-I brain connectomes"
echo "[native-v2-wasserstein] First compiler with W₁ + ORC as IR primitives"

MANIFEST_FILE="$(mktemp /tmp/wasserstein-manifest.XXXXXX.tsv)"
cat >"$MANIFEST_FILE" <<'EOF'
# case_id	program	expected_exit	expected_stdout	claim_class
wasserstein_orc	tests/native-v2/science_spine/wasserstein_orc.sio	0	tests/native-v2/science_spine/wasserstein_orc.stdout	door_phi_wasserstein_orc
EOF

SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="$MANIFEST_FILE" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

echo "[native-v2-wasserstein] PASS: Wasserstein + ORC gate"
