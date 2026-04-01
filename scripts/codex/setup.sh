#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export SOUNIO_REPO_HARD_NO_RUST="${SOUNIO_REPO_HARD_NO_RUST:-1}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

echo "SOUNIO_REPO_HARD_NO_RUST=$SOUNIO_REPO_HARD_NO_RUST"
echo "SOUNIO_STDLIB_PATH=$SOUNIO_STDLIB_PATH"

./bin/souc check self-hosted/compiler/lean_single.sio
