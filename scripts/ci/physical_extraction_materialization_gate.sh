#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -n "${SOUNIO_PHYSICAL_EXTRACTION_MATERIALIZATION_MADAROS_BIN:-}" ]]; then
  export SOUNIO_PHYSICAL_EXTRACTION_MADAROS_BIN="$SOUNIO_PHYSICAL_EXTRACTION_MATERIALIZATION_MADAROS_BIN"
fi

# Materialization can only consume an inventory accepted by the complete
# R0-R3 stack against the same current-source Madaros.
bash "$ROOT_DIR/scripts/ci/physical_extraction_inventory_gate.sh"
exec python3 "$ROOT_DIR/scripts/ci/physical_extraction_materialization_gate.py"
