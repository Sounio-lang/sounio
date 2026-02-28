#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

echo "[full-gate] 1/5 claude operational contract gate"
bash "$ROOT_DIR/scripts/claude_operational_contract_gate.sh"

echo "[full-gate] 2/5 fast gate"
bash "$ROOT_DIR/scripts/fast_gate.sh"

echo "[full-gate] 3/5 integration tests"
if [[ "${SOUNIO_REPO_HARD_NO_RUST:-1}" = "1" ]]; then
  echo "[full-gate] integration tests skipped (repo-hard no-rust mode)"
else
  (cd "$ROOT_DIR" && sounio_cargo test -p souc --tests)
fi

echo "[full-gate] 4/5 e2e backend gate"
"$ROOT_DIR/scripts/e2e_gate.sh"

echo "[full-gate] 5/5 website quality"
npm --prefix "$ROOT_DIR/website" run check:quality

echo "[full-gate] ok"
