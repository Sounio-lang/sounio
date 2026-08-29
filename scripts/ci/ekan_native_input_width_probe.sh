#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

./scripts/ci/ekan_native_frontier_matrix.sh | grep -E 'THREE_INPUT_TWO_HIDDEN|FOUR_INPUT|EKAN_NATIVE_FRONTIER_MATRIX_PASS'
echo "EKAN_NATIVE_INPUT_WIDTH_PROBE_PASS"
