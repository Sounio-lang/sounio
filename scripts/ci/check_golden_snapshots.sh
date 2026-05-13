#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

if [[ -n "${UPDATE_GOLDEN:-}" ]]; then
  echo "Refusing to run golden drift check with UPDATE_GOLDEN set."
  exit 1
fi

sounio_require_souc

echo "[golden-snapshots] lint parity spec"
python3 "$ROOT_DIR/scripts/r2/parity_spec_lint.py"

echo "[golden-snapshots] execute parity contracts with self-hosted souc"
python3 "$ROOT_DIR/scripts/r2/parity_spec_exec.py" --souc-bin "$SOUC_BIN"

echo "[golden-snapshots] ok"
