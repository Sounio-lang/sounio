#!/usr/bin/env bash
# CI gate for Functor F G2-covariance across the seven Fano lines.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/functor_f_g2_covariance_contract.py"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "FUNCTOR_F_G2_COVARIANCE_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running Functor F G2-covariance contract..."
"${PYTHON}" "${CONTRACT}"
EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "FUNCTOR_F_G2_COVARIANCE_GATE_OK"
    exit 0
else
    echo "FUNCTOR_F_G2_COVARIANCE_GATE_FAIL: contract did not reach G_GREEN"
    exit ${EXIT_CODE}
fi
