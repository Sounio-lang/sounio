#!/usr/bin/env bash
# Build gate for formal/lean4/SounioNonUnitaryNWA.lean
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/formal/lean4"
if [[ ! -x ./lake && ! -x "$(command -v lake || true)" ]]; then
  # Prefer elan/lake from formal toolchain
  if [[ -x "$HOME/.elan/bin/lake" ]]; then
    export PATH="$HOME/.elan/bin:$PATH"
  elif [[ -x "$ROOT/formal/lean4/.lake/bin/lake" ]]; then
    export PATH="$ROOT/formal/lean4/.lake/bin:$PATH"
  fi
fi
echo "== lake build SounioNonUnitaryNWA =="
lake build SounioNonUnitaryNWA
echo "SOUNIO_NUNWA_LEAN_GATE_OK"
