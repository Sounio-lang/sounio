#!/usr/bin/env bash
# L2 pure-emit — fo_bc_compile_expr pure fragment for oral Css.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

python3 scripts/research/fo_emit_pure_cert.py | tee /tmp/fo_emit_pure_cert.out
grep -q 'FO_EMIT_PURE_CERT_OK' /tmp/fo_emit_pure_cert.out

if [[ "${FO_CSS_LEAN_BUILD:-0}" == "1" ]]; then
  export ELAN_HOME="${ELAN_HOME:-/workspace/.home/openvscode-server/.elan}"
  export PATH="${ELAN_HOME}/bin:${PATH}"
  (cd formal/lean4 && lake build SounioFoEmitPure)
fi

echo "FO_EMIT_PURE_GATE_OK"
