#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -n "${SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_MADAROS_BIN:-}" ]]; then
  export SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_MADAROS_BIN="$SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_MADAROS_BIN"
fi

# Canonical execution is accepted only after the complete approval stack passes
# against the same current-source witness. The focused gate uses disposable Git
# repositories and never creates a production execution policy for Sounio.
bash "$ROOT_DIR/scripts/ci/physical_extraction_canonical_cutover_approval_gate.sh"
exec python3 "$ROOT_DIR/scripts/ci/physical_extraction_canonical_cutover_execution_gate.py"
