#!/usr/bin/env bash
# CI gate for the level-9 (512-dim) ZD census fast exact contract.
#
# Compiles scripts/research/l9_zd_census_fast.c into a scratch dir, runs it,
# and cross-checks the Cayley-Dickson sign table against the NumPy reference
# (scripts/research/routon_zd_contract.py) via the FNV-1a-64 hash printed by
# both implementations.
#
# Default build runs the full contract: census + exact GF(65521) rank
# verification of all 260610 pair-signs (~2-4 min).  Set L9_GATE_FAST=1 for
# the census-only build (seconds), which skips Method 2 and is marked as
# such in the verdict line.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC="${REPO_ROOT}/scripts/research/l9_zd_census_fast.c"

if [[ ! -f "${SRC}" ]]; then
    echo "L9_ZD_CENSUS_GATE_FAIL: missing ${SRC}"
    exit 1
fi

CC_BIN="${CC:-cc}"
if ! command -v "${CC_BIN}" >/dev/null 2>&1; then
    echo "L9_ZD_CENSUS_GATE_FAIL: no C compiler (${CC_BIN})"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

SCRATCH="$(mktemp -d)"
trap 'rm -rf "${SCRATCH}"' EXIT
BIN="${SCRATCH}/l9_zd_census_fast"

EXTRA_FLAGS=()
MODE="full (census + GF(65521) verification)"
if [[ "${L9_GATE_FAST:-0}" == "1" ]]; then
    EXTRA_FLAGS+=("-DL9_SKIP_VERIFY")
    MODE="fast (census-only)"
fi

echo "Building level-9 ZD census fast contract [${MODE}]..."
"${CC_BIN}" -O2 -Wall -Wextra "${EXTRA_FLAGS[@]}" -o "${BIN}" "${SRC}"

echo "Running level-9 ZD census fast contract (dim 512)..."
OUT="$("${BIN}")" || {
    echo "${OUT}"
    echo "L9_ZD_CENSUS_GATE_FAIL: contract exited non-zero"
    exit 1
}
echo "${OUT}"

if ! grep -q '^L9_ZD_FAST_VERDICT PASS' <<<"${OUT}"; then
    echo "L9_ZD_CENSUS_GATE_FAIL: verdict not PASS"
    exit 1
fi

C_HASH="$(sed -n 's/^L9_FAST_SIGN_TABLE .*fnv1a=\([0-9a-f]\{16\}\)$/\1/p' <<<"${OUT}")"
PY_HASH="$(REPO_ROOT="${REPO_ROOT}" "${PYTHON}" - <<'EOF'
import os
import sys
sys.path.insert(0, os.path.join(os.environ["REPO_ROOT"], "scripts", "research"))
from routon_zd_contract import get_sign_matrix
S = get_sign_matrix(9)
h = 1469598103934665603
for b in S.astype("<i1").tobytes():
    h ^= b
    h = (h * 1099511628211) % (1 << 64)
print(f"{h:016x}")
EOF
)"

echo "Sign-table cross-hash: C=${C_HASH} numpy=${PY_HASH}"
if [[ -z "${C_HASH}" || "${C_HASH}" != "${PY_HASH}" ]]; then
    echo "L9_ZD_CENSUS_GATE_FAIL: sign-table hash mismatch"
    exit 1
fi

echo "L9_ZD_CENSUS_GATE_OK"
