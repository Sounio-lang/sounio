#!/usr/bin/env bash
# g6_madaros_identity.sh
#
# Identity gate: the Stage1 modular compiler must identify itself as Madaros.
#
# Pass: bin/madaros --version (or the raw Madaros ELF) prints a line containing
#       "Madaros" and "Sounio self-hosted compiler".
#
# If bin/madaros / artifacts/self-hosted/madaros are absent, the gate builds
# Madaros via make build-madaros (using the global build lock).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"

if ! sounio_require_madaros >/dev/null 2>&1; then
    echo "[g6] Madaros binary not found; building..."
    make build-madaros
    # Re-source to pick up the newly built binary.
    unset _SOUNIO_RESOLVE_MADAROS_LOADED
    source "$ROOT_DIR/scripts/lib/resolve_madaros.sh"
fi

IDENTITY="$($MADAROS_BIN --version 2>/dev/null | head -n 1)"

if [[ -z "$IDENTITY" ]]; then
    echo "[g6] FAIL: Madaros produced no --version output"
    exit 1
fi

if [[ "$IDENTITY" != *"Madaros"* ]]; then
    echo "[g6] FAIL: expected 'Madaros' in identity, got: $IDENTITY"
    exit 1
fi

if [[ "$IDENTITY" != *"Sounio self-hosted compiler"* ]]; then
    echo "[g6] FAIL: expected 'Sounio self-hosted compiler' in identity, got: $IDENTITY"
    exit 1
fi

echo "[g6] PASS: Madaros identity OK: $IDENTITY"
exit 0
