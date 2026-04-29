#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Exercise the repaired raw source-bootstrap lane end-to-end, including the
# fixed-point self-host check and secondary macOS cross-target smoke.
export SOUNIO_FORCE_SOURCE_BOOTSTRAP=1

bash "$ROOT_DIR/scripts/ci/selfhost_host_gate.sh"
bash "$ROOT_DIR/scripts/codex/action-source-bootstrap-bitwise-smoke.sh"
