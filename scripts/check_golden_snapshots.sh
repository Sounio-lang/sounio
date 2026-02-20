#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${UPDATE_GOLDEN:-}" ]]; then
  echo "Refusing to run golden drift check with UPDATE_GOLDEN set."
  exit 1
fi

# Golden fixtures are normalized for Linux paths and are guarded by Linux-only tests.
cargo test -p souc --test e2e_tests e2e::golden:: -- --nocapture
