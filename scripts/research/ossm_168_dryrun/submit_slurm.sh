#!/usr/bin/env bash
# Legacy LEMON submit entrypoint.
# Keep this path working by delegating to the maintained confirmatory launcher.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TARGET="$ROOT_DIR/slurm-jobs/lemon_confirmatory/submit_full_lemon.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "error: missing maintained LEMON submit launcher: $TARGET" >&2
  exit 2
fi

echo "[lemon-submit] delegating to slurm-jobs/lemon_confirmatory/submit_full_lemon.sh" >&2
exec bash "$TARGET" "$@"
