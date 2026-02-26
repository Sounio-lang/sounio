#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

if [[ "$SKIP_BUILD" = "1" ]]; then
  echo "[golden-snapshots] skipped (SKIP_BUILD=1, requires cargo test)"
  exit 0
fi

if [[ -n "${UPDATE_GOLDEN:-}" ]]; then
  echo "Refusing to run golden drift check with UPDATE_GOLDEN set."
  exit 1
fi

# Golden fixtures are normalized for Linux paths and are guarded by Linux-only tests.
cargo test -p souc --test e2e_tests e2e::golden:: -- --nocapture
