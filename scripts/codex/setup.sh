#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export SOUNIO_REPO_HARD_NO_RUST="${SOUNIO_REPO_HARD_NO_RUST:-1}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

# shellcheck source=../lib/resolve_souc.sh
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

echo "SOUNIO_REPO_HARD_NO_RUST=$SOUNIO_REPO_HARD_NO_RUST"
echo "SOUNIO_STDLIB_PATH=$SOUNIO_STDLIB_PATH"
echo "SOUC_BIN=$SOUC_BIN"

"$SOUC_BIN" check self-hosted/compiler/lean_single.sio
