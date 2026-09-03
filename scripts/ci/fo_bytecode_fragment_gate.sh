#!/usr/bin/env bash
# L2 fragment — FO bytecode stack machine for oral Css (ops 1–6).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 scripts/research/fo_bytecode_fragment_cert.py | tee /tmp/fo_bytecode_fragment_cert.out
grep -q 'FO_BYTECODE_FRAGMENT_CERT_OK' /tmp/fo_bytecode_fragment_cert.out

if [[ "${FO_CSS_LEAN_BUILD:-0}" == "1" ]]; then
  export ELAN_HOME="${ELAN_HOME:-/workspace/.home/openvscode-server/.elan}"
  export PATH="${ELAN_HOME}/bin:${PATH}"
  (cd formal/lean4 && lake build SounioFoBytecodeFragment)
fi

echo "FO_BYTECODE_FRAGMENT_GATE_OK"
