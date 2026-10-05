#!/usr/bin/env bash
# CI gate for the sedenion ZD crown-graph code physical prediction.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/zd_qec_prediction_contract.py"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "ZD_QEC_PREDICTION_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running sedenion ZD crown-graph code prediction contract (levels 4-6)..."
EXIT_CODE=0
"${PYTHON}" "${CONTRACT}" || EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "ZD_QEC_PREDICTION_GATE_OK"
    exit 0
else
    echo "ZD_QEC_PREDICTION_GATE_FAIL: contract did not reach Q_GREEN"
    exit ${EXIT_CODE}
fi
