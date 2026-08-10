#!/usr/bin/env bash
# CI-facing alias for EXP17 ZWH ledger (lean_single + Madaros).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
bash scripts/research/particle_exp17_zwh_ledger_gate.sh
echo "PARTICLE_EXP17_CI_GATE_OK"
