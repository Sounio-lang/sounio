#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export SOUNIO_REPO_HARD_NO_RUST="${SOUNIO_REPO_HARD_NO_RUST:-1}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

echo "SOUNIO_REPO_HARD_NO_RUST=$SOUNIO_REPO_HARD_NO_RUST"
echo "SOUNIO_STDLIB_PATH=$SOUNIO_STDLIB_PATH"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

TMP_DIR="$(mktemp -d)"
OUT_BIN="$TMP_DIR/bootstrap-smoke.out"
trap 'rm -rf "$TMP_DIR"' EXIT

"$SOUC_BIN" self-hosted/compiler/lean_single.sio "$OUT_BIN"
test -s "$OUT_BIN"
