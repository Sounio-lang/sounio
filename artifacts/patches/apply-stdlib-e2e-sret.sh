#!/usr/bin/env bash
# Apply stdlib-only E2E SRET workarounds (no compiler/toolchain files).
# Usage: from repo root on target branch/worktree:
#   bash artifacts/patches/apply-stdlib-e2e-sret.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
PATCH="$ROOT/artifacts/patches/stdlib-e2e-sret-workarounds.patch"
LIST="$ROOT/artifacts/patches/stdlib-e2e-sret-untracked.list"
if [[ ! -f "$PATCH" ]]; then
  echo "missing patch: $PATCH" >&2
  exit 1
fi
git apply --check "$PATCH"
git apply "$PATCH"
echo "Applied tracked patch ($(wc -l < "$PATCH") lines)."
if [[ -f "$LIST" ]]; then
  echo "Untracked files to copy manually from source worktree (see $LIST):"
  grep -v '^#' "$LIST" | grep -v '^$' || true
fi
echo "Validate (no compiler rebuild):"
echo "  export SOUNIO_TEST_SOUC_BIN=scripts/ci/souc-native-wrapper.sh"
echo "  export SOUNIO_SOUC_BIN=artifacts/self-hosted/souc-self-hosted-x86_64"
echo "  bash scripts/run_sio_test_suite.sh epistemic_backward"
