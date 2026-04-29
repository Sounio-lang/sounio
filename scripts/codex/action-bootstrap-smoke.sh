#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

TMP_DIR="$(mktemp -d)"
OUT_BIN="$TMP_DIR/bootstrap-smoke.out"
trap 'rm -rf "$TMP_DIR"' EXIT

"$SOUC_BIN" self-hosted/compiler/lean_single.sio "$OUT_BIN"
test -s "$OUT_BIN"
