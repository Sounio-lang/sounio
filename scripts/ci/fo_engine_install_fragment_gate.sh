#!/usr/bin/env bash
# L2 engine-install fragment — multipass register pure helpers for oral Css.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 scripts/research/fo_engine_install_fragment_cert.py | tee /tmp/fo_engine_install_fragment_cert.out
grep -q 'FO_ENGINE_INSTALL_FRAGMENT_CERT_OK' /tmp/fo_engine_install_fragment_cert.out

if [[ "${FO_CSS_LEAN_BUILD:-0}" == "1" ]]; then
  export ELAN_HOME="${ELAN_HOME:-/workspace/.home/openvscode-server/.elan}"
  export PATH="${ELAN_HOME}/bin:${PATH}"
  (cd formal/lean4 && lake build SounioFoEngineInstallFragment)
fi

echo "FO_ENGINE_INSTALL_FRAGMENT_GATE_OK"
