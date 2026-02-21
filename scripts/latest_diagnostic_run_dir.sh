#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIAG_ROOT="${1:-$ROOT_DIR/artifacts/diagnostic}"

if [[ ! -d "$DIAG_ROOT" ]]; then
  exit 1
fi

latest_run="$(
  find "$DIAG_ROOT" -mindepth 1 -maxdepth 3 -type d -name logs -printf '%T@ %h\n' \
    | sort -nr \
    | awk 'NR==1 {print $2}'
)"

if [[ -z "${latest_run:-}" || ! -d "$latest_run" ]]; then
  exit 1
fi

echo "$latest_run"
