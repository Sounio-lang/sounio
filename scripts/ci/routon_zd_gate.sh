#!/usr/bin/env bash
# CI gate for routon (level-7, 128-dim) ZD structure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/routon_zd_contract.py"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "ROUTON_ZD_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running routon ZD structure contract (level 7, dim 128)..."
EXIT_CODE=0
"${PYTHON}" "${CONTRACT}" || EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "ROUTON_ZD_GATE_OK"
    exit 0
else
    echo "ROUTON_ZD_GATE_FAIL: contract did not reach C_GREEN"
    exit ${EXIT_CODE}
fi
