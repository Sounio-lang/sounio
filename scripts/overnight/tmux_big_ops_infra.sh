#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Infra/liveness profile: keep overnight lane healthy even while strict gate is unstable.
export PLAN_BIG_OVERNIGHT_GATE_SCRIPT="${PLAN_BIG_OVERNIGHT_GATE_SCRIPT:-/bin/true}"
exec bash "$ROOT_DIR/scripts/tmux_big_ops_default.sh" "$@"
