#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

FILES=(
  ".github/workflows/ci.yml"
  ".github/workflows/gpu-research.yml"
  "scripts/check_all_examples.sh"
  "scripts/run_glm_benchmarks.sh"
  "README.md"
  "INSTALL.md"
  "docs/FEATURE_FLAGS.md"
)

PATTERN='cd compiler|working-directory:\s*compiler|ROOT_DIR/compiler|compiler/target|compiler/src/|compiler/tests/|compiler/docs/|compiler/benches/'
TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

if grep -En "$PATTERN" "${FILES[@]}" >"$TMP"; then
  echo "Legacy compiler path references detected in critical files:"
  cat "$TMP"
  exit 1
fi

echo "Legacy path check passed for critical files."
