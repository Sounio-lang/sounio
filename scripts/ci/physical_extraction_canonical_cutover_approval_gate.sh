#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -n "${SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_MADAROS_BIN:-}" ]]; then
  export SOUNIO_PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_MADAROS_BIN="$SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_MADAROS_BIN"
fi

# Canonical cutover approval is accepted only after the complete local
# source-removal execution stack passes against the same current-source witness.
bash "$ROOT_DIR/scripts/ci/physical_extraction_source_removal_execution_gate.sh"
exec python3 "$ROOT_DIR/scripts/ci/physical_extraction_canonical_cutover_approval_gate.py"
