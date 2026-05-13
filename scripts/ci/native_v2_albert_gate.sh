#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_NATIVE_V2_ALBERT_DIR:-$(mktemp -d /tmp/sounio-native-v2-albert.XXXXXX)}"

echo "[native-v2-albert] out=$OUT_DIR"
echo "[native-v2-albert] IrJ3OJordanMul: first compiler with native Albert algebra J₃(O) primitive"
echo "[native-v2-albert] J₃(O): 27-dim, automorphism group F₄, unique exceptional simple Jordan algebra"
echo "[native-v2-albert] Test: Jordan identity (D²)∘D = D∘(D²) for diagonal elements"
echo "[native-v2-albert] Connection: arxiv:2006.16265 -- 3 fermion generations ↔ 3×3 block structure"

MANIFEST_FILE="$(mktemp /tmp/albert-manifest.XXXXXX.tsv)"
cat >"$MANIFEST_FILE" <<'EOF'
# case_id	program	expected_exit	expected_stdout	claim_class
albert_jordan_product	tests/native-v2/science_spine/albert_jordan_product.sio	0	tests/native-v2/science_spine/albert_jordan_product.stdout	albert_j3o
EOF

SOUNIO_NATIVE_V2_SCIENCE_SPINE_DIR="$OUT_DIR" \
SOUNIO_NATIVE_V2_SCIENCE_SPINE_MANIFEST="$MANIFEST_FILE" \
  bash scripts/ci/native_v2_epistemic_science_spine_gate.sh

echo "[native-v2-albert] PASS: Albert algebra Jordan product gate"
