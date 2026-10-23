#!/usr/bin/env bash
# verify_memory_certificate.sh — Spine A: verify Gen N cert (+ optional Gen N-1 peer)
#
# Usage:
#   bash scripts/bootstrap/verify_memory_certificate.sh <cert.json> [elf]
#   bash scripts/bootstrap/verify_memory_certificate.sh <cert.json> [elf] --prev <prev.json>
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CERT="${1:?usage: verify_memory_certificate.sh <cert.json> [elf] [--prev prev.json]}"
shift || true
ELF=""
PREV=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prev)
      PREV="${2:?}"
      shift 2
      ;;
    *)
      if [[ -z "$ELF" && -f "$1" ]]; then
        ELF="$1"
        shift
      else
        echo "error: unexpected arg: $1" >&2
        exit 2
      fi
      ;;
  esac
done

if [[ -n "$PREV" ]]; then
  PREV_ARGS=(python3 "$ROOT/scripts/memory_cert/verify_certificate.py" "$PREV")
  if [[ -n "$ELF" ]]; then
    PREV_ARGS+=(--elf "$ELF")
  fi
  "${PREV_ARGS[@]}"
fi

ARGS=(python3 "$ROOT/scripts/memory_cert/verify_certificate.py" "$CERT")
if [[ -n "$ELF" ]]; then
  ARGS+=(--elf "$ELF")
fi
if [[ -n "$PREV" ]]; then
  ARGS+=(--fixed-point-with "$PREV")
fi
exec "${ARGS[@]}"
