#!/usr/bin/env bash
# CI gate for R2 continuous tube law theorem verification.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/r2_continuous_law_theorem_contract.py"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "R2_CONTINUOUS_LAW_THEOREM_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running R2 continuous law theorem contract..."
"${PYTHON}" "${CONTRACT}"
EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "R2_CONTINUOUS_LAW_THEOREM_GATE_OK"
    exit 0
else
    echo "R2_CONTINUOUS_LAW_THEOREM_GATE_FAIL: contract did not reach T_GREEN"
    exit ${EXIT_CODE}
fi
