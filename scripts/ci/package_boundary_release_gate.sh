#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

if [[ -n "${SOUNIO_PACKAGE_BOUNDARY_MADAROS_BIN:-}" ]]; then
  export SOUNIO_SCIENCE_BOUNDARY_MADAROS_BIN="$SOUNIO_PACKAGE_BOUNDARY_MADAROS_BIN"
fi

# R2.5 extends R0-R2. A package release result is not authoritative unless the
# underlying boundary gate also passes against the same current-source Madaros.
bash "$ROOT_DIR/scripts/ci/science_boundary_gate.sh"
exec python3 "$ROOT_DIR/scripts/ci/package_boundary_release_gate.py"
