#!/usr/bin/env bash
# CI gate for the E-series semantic germ contract.
#
# Companion to:
#   docs/research/e_series_semantic_germ_spec_2026-07-26.md
#   scripts/research/e_series_semantic_germ_contract.py
#
# Verifies:
#   - the E-series Petitot germs and their Milnor/unfolding data (G1..G2 PASS)
#   - the Morsification census mu = 6/7/8 over C (G3 PASS)
#   - the adjacency witnesses: E6 closure FULL, E7/E8 spines (G4..G6 PASS)
#   - the octonion/associator clauses O1..O4 PASS
#   - the verdict string E_SERIES_GERMS_CONSTRUCTED__WEAK_FORM_TESTABLE__WEAK_FORM_OPEN
#   - the spec carries the verdict and the testability statement
#
# Usage:
#   bash scripts/ci/e_series_semantic_germ_gate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/e_series_semantic_germ_contract.py"
SPEC="${REPO_ROOT}/docs/research/e_series_semantic_germ_spec_2026-07-26.md"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "E_SERIES_GERM_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi
if [[ ! -f "${SPEC}" ]]; then
    echo "E_SERIES_GERM_GATE_FAIL: missing spec ${SPEC}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running E-series semantic germ contract..."
"${PYTHON}" "${CONTRACT}" | tee /tmp/e_series_semantic_germ_out.txt
EXIT_CODE=${PIPESTATUS[0]}

if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "E_SERIES_GERM_GATE_FAIL: contract exited ${EXIT_CODE}"
    exit "${EXIT_CODE}"
fi

grep -q 'E_SERIES_SEMANTIC_GERM_OK' /tmp/e_series_semantic_germ_out.txt
grep -q 'E_SERIES_GERM_VERDICT E_SERIES_GERMS_CONSTRUCTED__WEAK_FORM_TESTABLE__WEAK_FORM_OPEN' /tmp/e_series_semantic_germ_out.txt
grep -q 'G1_GERMS_MILNOR' /tmp/e_series_semantic_germ_out.txt
grep -q 'G2_E_TYPE_JET' /tmp/e_series_semantic_germ_out.txt
grep -q 'G3_MORSIFICATION_E6' /tmp/e_series_semantic_germ_out.txt
grep -q 'G3_MORSIFICATION_E7' /tmp/e_series_semantic_germ_out.txt
grep -q 'G3_MORSIFICATION_E8' /tmp/e_series_semantic_germ_out.txt
grep -q 'G4_E6_ADJACENCY_FULL' /tmp/e_series_semantic_germ_out.txt
grep -q 'G5_E7_ADJACENCY_SPINE' /tmp/e_series_semantic_germ_out.txt
grep -q 'G6_E8_ADJACENCY_SPINE' /tmp/e_series_semantic_germ_out.txt
grep -q 'O1_PHI_IS_CUBIC_CROSSTERM' /tmp/e_series_semantic_germ_out.txt
grep -q 'O2_ASSOCIATOR_SEPARATE' /tmp/e_series_semantic_germ_out.txt
grep -q 'O3_MAGIC_SQUARE_CHAIN' /tmp/e_series_semantic_germ_out.txt
grep -q 'O4_NO_FORM_IDENTITY' /tmp/e_series_semantic_germ_out.txt
if grep -q 'CONTRACT_INCOMPLETE' /tmp/e_series_semantic_germ_out.txt; then
    echo "E_SERIES_GERM_GATE_FAIL: contract incomplete"
    exit 1
fi

# Spec must carry the verdict and the testability statement.
grep -q 'E_SERIES_GERMS_CONSTRUCTED__WEAK_FORM_TESTABLE__WEAK_FORM_OPEN' "${SPEC}"
grep -q 'testable' "${SPEC}"

echo "E_SERIES_SEMANTIC_GERM_GATE_OK"
