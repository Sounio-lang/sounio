#!/usr/bin/env bash
# CI gate for the Garden-to-Claim pipeline (Zero of Encounter instantiation).
#
# Composes the two executable witness gates with the pipeline contract.
# The witness gates execute explicitly through lean_single; the native-v2
# frontier stays classified in scripts/ci/zero_event_native_v2_matrix.sh and
# is not hidden here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/garden_to_claim_pipeline_contract.py"

fail() {
    echo "GARDEN_TO_CLAIM_GATE_FAIL: $*" >&2
    exit 1
}

[[ -f "${CONTRACT}" ]] || fail "missing ${CONTRACT}"

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running zero-event stdlib witness gate..."
bash "${SCRIPT_DIR}/zero_event_gate.sh"

echo "Running zero-provenance witness gate..."
bash "${SCRIPT_DIR}/zero_provenance_witness_gate.sh"

echo "Running Garden-to-Claim pipeline contract..."
"${PYTHON}" "${CONTRACT}" || fail "contract did not reach P_GREEN"

echo "GARDEN_TO_CLAIM_GATE_OK"
