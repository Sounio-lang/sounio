#!/usr/bin/env bash
# CI gate for zero-provenance claims in the Falsification Ledger.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/zero_provenance_claims_contract.py"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "ZERO_PROVENANCE_CLAIMS_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running zero-provenance claims contract..."
"${PYTHON}" "${CONTRACT}"
EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "ZERO_PROVENANCE_CLAIMS_GATE_OK"
    exit 0
else
    echo "ZERO_PROVENANCE_CLAIMS_GATE_FAIL: contract did not reach Z_GREEN"
    exit ${EXIT_CODE}
fi
