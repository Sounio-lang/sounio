#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

./scripts/ci/ekan_native_frontier_matrix.sh | grep -E 'LINEAR_READOUT|HAT3_READOUT|HAT5_READOUT|EKAN_NATIVE_FRONTIER_MATRIX_PASS'
echo "EKAN_NATIVE_READOUT_PROBE_PASS"
