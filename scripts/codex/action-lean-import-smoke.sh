#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

TMP_DIR="$(mktemp -d)"
OUT_BIN="$TMP_DIR/lean-import-smoke.out"
trap 'rm -rf "$TMP_DIR"' EXIT

# Canonical bootstrap lane must still self-host from lean_single.
"$SOUC_BIN" self-hosted/compiler/lean_single.sio "$OUT_BIN"
test -s "$OUT_BIN"

# Import-based user programs should continue to work even while the modular
# compiler entrypoint remains a shadow/repair lane.
"$SOUC_BIN" check tests/run-pass/import_basic_main.sio
"$SOUC_BIN" run tests/run-pass/import_basic_main.sio

"$SOUC_BIN" check tests/multimodule/thin_single_main.sio
set +e
"$SOUC_BIN" run tests/multimodule/thin_single_main.sio
RC=$?
set -e

# thin_single_main returns add_public(3, 4) = 7.
if [[ "$RC" -ne 7 ]]; then
  echo "error: thin_single_main.sio returned $RC, expected 7" >&2
  exit 1
fi
