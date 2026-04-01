#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_WRAPPER="${1:-/tmp/sounio-ontology-validation-souc}"

bash scripts/ci/build_ontology_validation_souc.sh "$OUT_WRAPPER"
"$OUT_WRAPPER" info
