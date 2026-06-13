#!/usr/bin/env bash
# g6_madaros_identity.sh - Madaros Stage1 identity gate.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# shellcheck source=/dev/null
source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"

if ! sounio_require_madaros_raw >/dev/null 2>&1; then
  echo "[g6] Madaros raw ELF not found; building..."
  bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$ROOT_DIR/artifacts/self-hosted/madaros"
  unset _SOUNIO_RESOLVE_MADAROS_LOADED
  # shellcheck source=/dev/null
  source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
fi

sounio_require_madaros

VERSION_OUTPUT="$("$MADAROS_BIN" --version 2>&1 || true)"
IDENTITY="$(printf '%s\n' "$VERSION_OUTPUT" | sed -n '1p')"

if [[ -z "$IDENTITY" ]]; then
  echo "[g6] FAIL: Madaros produced no --version output" >&2
  exit 1
fi

if [[ "$VERSION_OUTPUT" != *"Madaros"* && "$VERSION_OUTPUT" != *"Madares"* ]]; then
  echo "[g6] FAIL: expected Madaros/Madares identity, got:" >&2
  printf '%s\n' "$VERSION_OUTPUT" >&2
  exit 1
fi

if [[ "$VERSION_OUTPUT" != *"Sounio self-hosted compiler"* ]]; then
  echo "[g6] FAIL: expected 'Sounio self-hosted compiler' in identity, got:" >&2
  printf '%s\n' "$VERSION_OUTPUT" >&2
  exit 1
fi

echo "[g6] PASS: Madaros identity OK: $IDENTITY"
