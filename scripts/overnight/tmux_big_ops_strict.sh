#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Strict profile: force gate to real plan_big gate script.
export PLAN_BIG_OVERNIGHT_GATE_SCRIPT="${PLAN_BIG_STRICT_GATE_SCRIPT:-$ROOT_DIR/scripts/ci/plan_big_gate.sh}"
exec bash "$ROOT_DIR/scripts/tmux_big_ops_default.sh" "$@"
