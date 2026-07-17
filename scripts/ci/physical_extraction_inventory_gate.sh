#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -n "${SOUNIO_PHYSICAL_EXTRACTION_MADAROS_BIN:-}" ]]; then
  export SOUNIO_REGISTRY_ATTESTATION_MADAROS_BIN="$SOUNIO_PHYSICAL_EXTRACTION_MADAROS_BIN"
fi

# R3 inventories the physical source boundary prepared by R0-R2.6. The
# inventory result is authoritative only after the complete upstream stack
# passes against the same current-source Madaros.
bash "$ROOT_DIR/scripts/ci/registry_attestation_spec_gate.sh"
exec python3 "$ROOT_DIR/scripts/ci/physical_extraction_inventory_gate.py"
