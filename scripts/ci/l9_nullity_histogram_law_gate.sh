#!/usr/bin/env bash
# CI gate for the level-9 (512-dim) nullity-histogram counting-law contract.
#
# Compiles scripts/research/l9_nullity_histogram_law_contract.c into a scratch
# dir, runs it (exact 2-cycle census at level 9, lemma checks L1-L4 at level 9,
# counting-law comparison, GF(65521) subsample audit), and cross-checks the
# Cayley-Dickson sign table against the NumPy reference
# (scripts/research/routon_zd_contract.py) via the FNV-1a-64 hash printed by
# both implementations.
#
# Set L9_FULL_VERIFY=1 to additionally run the complete GF(65521) audit of
# all 260610 candidate pair-signs (~2.5 min instead of ~7 s).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC="${REPO_ROOT}/scripts/research/l9_nullity_histogram_law_contract.c"

if [[ ! -f "${SRC}" ]]; then
    echo "L9_NULLITY_LAW_GATE_FAIL: missing ${SRC}"
    exit 1
fi

CC_BIN="${CC:-cc}"
if ! command -v "${CC_BIN}" >/dev/null 2>&1; then
    echo "L9_NULLITY_LAW_GATE_FAIL: no C compiler (${CC_BIN})"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

SCRATCH="$(mktemp -d)"
trap 'rm -rf "${SCRATCH}"' EXIT
BIN="${SCRATCH}/l9_nullity_histogram_law_contract"

echo "Building level-9 nullity-histogram counting-law contract..."
"${CC_BIN}" -O2 -Wall -Wextra -o "${BIN}" "${SRC}"

echo "Running level-9 nullity-histogram counting-law contract (dim 512)..."
OUT="$("${BIN}")" || {
    echo "${OUT}"
    echo "L9_NULLITY_LAW_GATE_FAIL: contract exited non-zero"
    exit 1
}
echo "${OUT}"

if ! grep -q '^L9_NULLITY_LAW_VERDICT PASS$' <<<"${OUT}"; then
    echo "L9_NULLITY_LAW_GATE_FAIL: verdict not PASS"
    exit 1
fi

C_HASH="$(sed -n 's/^L9_SIGN_TABLE .*fnv1a=\([0-9a-f]\{16\}\)$/\1/p' <<<"${OUT}")"
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
    echo "L9_NULLITY_LAW_GATE_FAIL: sign-table hash mismatch"
    exit 1
fi

echo "L9_NULLITY_LAW_GATE_OK"
