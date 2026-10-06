#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

echo "[full-gate] 1/6 Madaros operational contract gate"
bash "$ROOT_DIR/scripts/ci/madaros_operational_contract_gate.sh"

echo "[full-gate] 2/6 plan big unified gate"
bash "$ROOT_DIR/scripts/ci/plan_big_gate.sh"

echo "[full-gate] 3/6 fast gate"
bash "$ROOT_DIR/scripts/dev/fast_gate.sh"

echo "[full-gate] 4/6 integration tests"
echo "[full-gate] skipped (these were the Rust compiler crate's tests; that crate no longer exists)"

echo "[full-gate] 5/6 e2e backend gate"
"$ROOT_DIR/scripts/dev/e2e_gate.sh"

echo "[full-gate] 6/6 website quality"
npm --prefix "$ROOT_DIR/website" run check:quality

echo "[full-gate] ok"
