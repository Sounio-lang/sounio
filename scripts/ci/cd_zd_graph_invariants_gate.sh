#!/usr/bin/env bash
# CI gate for the Cayley-Dickson tower ZD-graph invariants theorem package
# (pair criterion, crown-join recursion, degree law, generator isolation,
# independence law, clique/chromatic law; levels 4..9, up to 512 dimensions).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/cd_zd_graph_invariants_contract.py"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "CD_ZD_GRAPH_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running CD-tower ZD-graph invariants contract (levels 4..9)..."
EXIT_CODE=0
(cd "${REPO_ROOT}" && "${PYTHON}" "${CONTRACT}") || EXIT_CODE=$?

if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "CD_ZD_GRAPH_GATE_OK"
    exit 0
else
    echo "CD_ZD_GRAPH_GATE_FAIL: contract did not reach C_GREEN"
    exit ${EXIT_CODE}
fi
