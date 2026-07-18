#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -n "${SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_MADAROS_BIN:-}" ]]; then
  export SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_MADAROS_BIN="$SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_MADAROS_BIN"
fi

echo 'SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_GATE_START'
bash "$ROOT_DIR/scripts/ci/physical_extraction_canonical_production_gap_gate.sh"
python3 "$ROOT_DIR/scripts/ci/physical_extraction_canonical_production_mapping_decision_gate.py"
