#!/usr/bin/env bash
# CI gate for the Falsification Ledger.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/falsification_ledger_contract.py"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "FALSIFICATION_LEDGER_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running Falsification Ledger contract..."
"${PYTHON}" "${CONTRACT}"
EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "FALSIFICATION_LEDGER_GATE_OK"
    exit 0
else
    echo "FALSIFICATION_LEDGER_GATE_FAIL: contract did not reach L_GREEN"
    exit ${EXIT_CODE}
fi
