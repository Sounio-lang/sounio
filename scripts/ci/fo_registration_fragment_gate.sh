#!/usr/bin/env bash
# L2 registration fragment — multipass FO_XFER expand for oral Css helpers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 scripts/research/fo_registration_fragment_cert.py | tee /tmp/fo_registration_fragment_cert.out
grep -q 'FO_REGISTRATION_FRAGMENT_CERT_OK' /tmp/fo_registration_fragment_cert.out

if [[ "${FO_CSS_LEAN_BUILD:-0}" == "1" ]]; then
  export ELAN_HOME="${ELAN_HOME:-/workspace/.home/openvscode-server/.elan}"
  export PATH="${ELAN_HOME}/bin:${PATH}"
  (cd formal/lean4 && lake build SounioFoRegistrationFragment)
fi

echo "FO_REGISTRATION_FRAGMENT_GATE_OK"
