#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

node "$ROOT_DIR/scripts/docs/check_docs_registry.mjs"
node "$ROOT_DIR/scripts/docs/check_docs_registry_selftest.mjs"
