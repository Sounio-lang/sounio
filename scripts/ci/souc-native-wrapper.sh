#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -n "${SOUC_NATIVE_BIN:-}" && -x "${SOUC_NATIVE_BIN:-}" ]]; then
  export SOUNIO_SOUC_BIN="$SOUC_NATIVE_BIN"
fi
exec "$ROOT_DIR/bin/souc" "$@"
