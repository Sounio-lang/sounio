#!/usr/bin/env bash
# CI gate for Mercyful Learning Runtime prototype.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/mercyful_runtime_contract.py"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "MERCYFUL_RUNTIME_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running Mercyful Learning Runtime contract..."
"${PYTHON}" "${CONTRACT}"
EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "MERCYFUL_RUNTIME_GATE_OK"
    exit 0
else
    echo "MERCYFUL_RUNTIME_GATE_FAIL: contract did not reach M_GREEN"
    exit ${EXIT_CODE}
fi
