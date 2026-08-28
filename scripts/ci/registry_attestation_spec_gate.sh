#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -n "${SOUNIO_REGISTRY_ATTESTATION_MADAROS_BIN:-}" ]]; then
  export SOUNIO_PACKAGE_BOUNDARY_MADAROS_BIN="$SOUNIO_REGISTRY_ATTESTATION_MADAROS_BIN"
fi

# R2.6 is authoritative only when the complete R0-R2 and R2.5 gates pass with
# the same current-source Madaros used by the registry attestation gate.
bash "$ROOT_DIR/scripts/ci/package_boundary_release_gate.sh"
exec python3 "$ROOT_DIR/scripts/ci/registry_attestation_spec_gate.py"
