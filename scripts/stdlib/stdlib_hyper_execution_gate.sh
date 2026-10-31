#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc
export SOUC_BIN
exec node "$ROOT_DIR/scripts/stdlib/hyper_gate.mjs" "$@"
