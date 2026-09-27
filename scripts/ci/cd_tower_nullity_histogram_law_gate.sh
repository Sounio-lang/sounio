#!/usr/bin/env bash
# CI gate for the Cayley-Dickson tower nullity histogram law
# (levels 4..8, up to 256 dimensions).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/cd_tower_nullity_histogram_law_contract.py"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "CD_HISTOGRAM_LAW_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running CD-tower nullity histogram law contract (levels 4..8)..."
EXIT_CODE=0
"${PYTHON}" "${CONTRACT}" || EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "CD_HISTOGRAM_LAW_GATE_OK"
    exit 0
else
    echo "CD_HISTOGRAM_LAW_GATE_FAIL: contract did not reach C_GREEN"
    exit ${EXIT_CODE}
fi
