#!/usr/bin/env bash
# Thin wrapper around lean_lvalue_shape_matrix.py. Run from repo root.
#
# Usage: scripts/ci/lean_lvalue_shape_matrix.sh [OUT_DIR]
#
# Env:
#   SOUNIO_LVALUE_SEED           path to lean_single compiler binary
#   SOUNIO_SHAPE_MATRIX_BASELINE if "1", always exit 0 (baseline mode)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUT_DIR="${1:-$(mktemp -d /tmp/lean_lvalue_shapes.XXXXXX)}"

cd "${REPO_ROOT}"
python3 scripts/ci/lean_lvalue_shape_matrix.py "${OUT_DIR}"
