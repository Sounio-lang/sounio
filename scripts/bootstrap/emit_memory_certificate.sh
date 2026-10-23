#!/usr/bin/env bash
# emit_memory_certificate.sh — Spine A: emit a Memory Certificate for one generation
#
# Usage:
#   bash scripts/bootstrap/emit_memory_certificate.sh <generation> [elf] [out.json]
#
# Defaults:
#   elf = artifacts/self-hosted/madaros (else bin/madaros-linux-x86_64)
#   out = artifacts/memory_certs/madaros_gen<N>.json
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GEN="${1:?usage: emit_memory_certificate.sh <generation> [elf] [out.json]}"
ELF="${2:-}"
OUT="${3:-$ROOT/artifacts/memory_certs/madaros_gen${GEN}.json}"
BOUNDS="$ROOT/artifacts/memory_certs/madaros_phase_bounds.v1.json"

if [[ -z "$ELF" ]]; then
  if [[ "$GEN" -eq 0 ]]; then
    # Gen 0 = committed / seed prebuilt when present
    for candidate in \
      "$ROOT/bin/madaros-linux-x86_64" \
      "$ROOT/bin/madaros" \
      "$ROOT/artifacts/self-hosted/madaros"
    do
      if [[ -f "$candidate" ]]; then
        ELF="$candidate"
        break
      fi
    done
  else
    # Gen ≥1 prefers from-source Madaros
    for candidate in \
      "$ROOT/artifacts/self-hosted/madaros" \
      "$ROOT/bin/madaros-linux-x86_64" \
      "$ROOT/bin/madaros"
    do
      if [[ -f "$candidate" ]]; then
        ELF="$candidate"
        break
      fi
    done
  fi
fi
[[ -n "$ELF" && -f "$ELF" ]] || { echo "error: no madaros ELF found" >&2; exit 2; }
[[ -f "$BOUNDS" ]] || { echo "error: missing $BOUNDS" >&2; exit 2; }

KIND=Estimated
NOTES="Bootstrap Gen ${GEN}: Estimated phase budgets bound to ELF sha256."
if [[ "$GEN" -ge 1 ]]; then
  NOTES="Bootstrap Gen ${GEN}: certificate for from-source Madaros; bounds still Estimated until Lean."
fi

exec python3 "$ROOT/scripts/memory_cert/emit_certificate.py" \
  --generation "$GEN" \
  --out "$OUT" \
  --repo "$ROOT" \
  --bounds "$BOUNDS" \
  --elf "$ELF" \
  --kind "$KIND" \
  --notes "$NOTES"
