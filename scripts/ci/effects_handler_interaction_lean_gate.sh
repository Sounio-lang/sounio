#!/usr/bin/env bash
# Build gate for SounioEffects handler×payload interaction (§19) + N×NWA §4.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT/formal/lean4"
if [[ ! -x ./lake && ! -x "$(command -v lake || true)" ]]; then
  if [[ -x "$HOME/.elan/bin/lake" ]]; then
    export PATH="$HOME/.elan/bin:$PATH"
  elif [[ -x "$ROOT/formal/lean4/.lake/bin/lake" ]]; then
    export PATH="$ROOT/formal/lean4/.lake/bin:$PATH"
  fi
fi
echo "== lake build SounioEffects =="
lake build SounioEffects
echo "== lake build SounioNonUnitaryNWA =="
lake build SounioNonUnitaryNWA
echo "SOUNIO_EFFECTS_HANDLER_INTERACTION_LEAN_GATE_OK"
