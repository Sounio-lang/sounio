#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${HYPERCOMPLEX_WAVE4_OUT_JSON:-$ROOT_DIR/artifacts/research/hypercomplex_wave4_validation.v1.json}"

python3 "$ROOT_DIR/scripts/research/validate_hypercomplex_wave4.py" --out-json "$OUT_JSON"
echo "[hypercomplex-wave4] validation passed: $OUT_JSON"
