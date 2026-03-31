#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_JSON="${HYPERCOMPLEX_WAVE7_OUT_JSON:-$ROOT_DIR/artifacts/research/hypercomplex_wave7_validation.v1.json}"

python3 "$ROOT_DIR/scripts/research/validate_hypercomplex_wave7.py" --out-json "$OUT_JSON"
echo "[hypercomplex-wave7] validation passed: $OUT_JSON"
