#!/usr/bin/env bash
# CI gate for G2 action on ZD fibers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/g2_zd_fibers_contract.py"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "G2_ZD_FIBERS_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running G2 action on ZD fibers contract..."
"${PYTHON}" "${CONTRACT}"
EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "G2_ZD_FIBERS_GATE_OK"
    exit 0
else
    echo "G2_ZD_FIBERS_GATE_FAIL: contract did not reach G_GREEN"
    exit ${EXIT_CODE}
fi
