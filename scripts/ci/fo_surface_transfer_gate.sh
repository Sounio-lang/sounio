#!/usr/bin/env bash
# Semantic bridge residual §5.4 compiler half — FoExpr desugar surface transfer.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 scripts/research/fo_surface_transfer_cert.py | tee /tmp/fo_surface_transfer_cert.out
grep -q 'FO_SURFACE_TRANSFER_CERT_OK' /tmp/fo_surface_transfer_cert.out

if [[ "${FO_CSS_LEAN_BUILD:-0}" == "1" ]]; then
  export ELAN_HOME="${ELAN_HOME:-/workspace/.home/openvscode-server/.elan}"
  export PATH="${ELAN_HOME}/bin:${PATH}"
  (cd formal/lean4 && lake build SounioFoSurfaceTransfer)
fi

echo "FO_SURFACE_TRANSFER_GATE_OK"
