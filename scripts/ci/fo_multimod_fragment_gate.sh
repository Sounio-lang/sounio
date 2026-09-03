#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
python3 scripts/research/fo_multimod_fragment_cert.py | tee /tmp/fo_multimod_fragment_cert.out
grep -q 'FO_MULTIMOD_FRAGMENT_CERT_OK' /tmp/fo_multimod_fragment_cert.out
if [[ "${FO_CSS_LEAN_BUILD:-0}" == "1" ]]; then
  export ELAN_HOME="${ELAN_HOME:-/workspace/.home/openvscode-server/.elan}"
  export PATH="${ELAN_HOME}/bin:${PATH}"
  (cd formal/lean4 && lake build SounioFoMultimodFragment)
fi
echo "FO_MULTIMOD_FRAGMENT_GATE_OK"
