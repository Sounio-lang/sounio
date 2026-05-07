#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

FILTER="${1:-r2::parity::}"
echo "[golden-update] filter=$FILTER"

sounio_require_souc

echo "[golden-update] running parity spec lint"
python3 "$ROOT_DIR/scripts/r2/parity_spec_lint.py"

echo "[golden-update] validating parity contracts in self-hosted mode"
python3 "$ROOT_DIR/scripts/r2/parity_spec_exec.py" --souc-bin "$SOUC_BIN"

echo "[golden-update] no-rust mode: parity contracts validated (manual fixture edits only)"
