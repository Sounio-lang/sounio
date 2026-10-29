#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${NODE_BIN:-node}" "$ROOT/tests/fixtures/wave4_variance_shadow/harness.mjs" "$@"
