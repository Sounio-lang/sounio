#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

echo "[full-gate] 1/7 claude operational contract gate"
bash "$ROOT_DIR/scripts/ci/claude_operational_contract_gate.sh"

echo "[full-gate] 2/7 Madaros operational contract gate"
bash "$ROOT_DIR/scripts/ci/madaros_operational_contract_gate.sh"

echo "[full-gate] 3/7 plan big unified gate"
bash "$ROOT_DIR/scripts/ci/plan_big_gate.sh"

echo "[full-gate] 4/7 fast gate"
bash "$ROOT_DIR/scripts/dev/fast_gate.sh"

echo "[full-gate] 5/7 integration tests"
if [[ "${SOUNIO_REPO_HARD_NO_RUST:-1}" = "1" ]]; then
  echo "[full-gate] integration tests skipped (repo-hard no-rust mode)"
else
  (cd "$ROOT_DIR" && sounio_cargo test -p souc --tests)
fi

echo "[full-gate] 6/7 e2e backend gate"
"$ROOT_DIR/scripts/dev/e2e_gate.sh"

echo "[full-gate] 7/7 website quality"
npm --prefix "$ROOT_DIR/website" run check:quality

echo "[full-gate] ok"
