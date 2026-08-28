#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -n "${SOUNIO_PHYSICAL_EXTRACTION_SOURCE_REMOVAL_MADAROS_BIN:-}" ]]; then
  export SOUNIO_PHYSICAL_EXTRACTION_MATERIALIZATION_MADAROS_BIN="$SOUNIO_PHYSICAL_EXTRACTION_SOURCE_REMOVAL_MADAROS_BIN"
fi

# Authorization consumes the complete R0-R3 materialization witness against
# the same source snapshot before it simulates removal in a temporary copy.
bash "$ROOT_DIR/scripts/ci/physical_extraction_materialization_gate.sh"
exec python3 "$ROOT_DIR/scripts/ci/physical_extraction_source_removal_authorization_gate.py"
