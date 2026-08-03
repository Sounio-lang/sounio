#!/usr/bin/env bash
# Algebraic FO Css surface-parity residual (§5.4 mathematical half).
# Executable mirror of formal/lean4/SounioFoCssSurfaceParity.lean.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 scripts/research/fo_css_surface_parity_cert.py | tee /tmp/fo_css_surface_parity_cert.out
grep -q 'FO_CSS_SURFACE_PARITY_CERT_OK' /tmp/fo_css_surface_parity_cert.out

# Optional Lean build when lake is available and not resource-starved.
if [[ "${FO_CSS_LEAN_BUILD:-0}" == "1" ]]; then
  export ELAN_HOME="${ELAN_HOME:-/workspace/.home/openvscode-server/.elan}"
  export PATH="${ELAN_HOME}/bin:${PATH}"
  (cd formal/lean4 && lake build SounioFoCssSurfaceParity)
fi

echo "FO_CSS_SURFACE_PARITY_GATE_OK"
