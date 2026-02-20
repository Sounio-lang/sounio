#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[full-gate] 1/4 fast gate"
bash "$ROOT_DIR/scripts/fast_gate.sh"

echo "[full-gate] 2/4 integration tests"
(cd "$ROOT_DIR" && cargo test -p souc --tests)

echo "[full-gate] 3/4 e2e backend gate"
"$ROOT_DIR/scripts/e2e_gate.sh"

echo "[full-gate] 4/4 website quality"
npm --prefix "$ROOT_DIR/website" run check:quality

echo "[full-gate] ok"
