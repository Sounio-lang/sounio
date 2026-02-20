#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

FILTER="${1:-e2e::golden::}"
echo "[golden-update] filter=$FILTER"

UPDATE_GOLDEN=1 cargo test -p souc --test e2e_tests "$FILTER" -- --nocapture
