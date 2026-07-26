#!/usr/bin/env bash
# CI gate for the ADE-Wildgen McKay contract.
#
# Companion to:
#   docs/research/ade_wildgen_mckay_spec_2026-07-26.md
#   scripts/research/ade_wildgen_mckay_contract.py
#
# Verifies:
#   - the McKay correspondence for E6/E7/E8 (M1..M4 clause groups PASS)
#   - the G2/octonion comparison clauses (C1..C4 PASS)
#   - the verdict string STRONG_FORM_OBSTRUCTED__WEAK_FORM_UNDECIDABLE
#   - the semantic-side premise: the programme's operative Petitot germs are
#     A-series (cusp x^4, butterfly x^6) in docs/research/petitot_potential.py
#
# Usage:
#   bash scripts/ci/ade_wildgen_mckay_gate.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/ade_wildgen_mckay_contract.py"
SPEC="${REPO_ROOT}/docs/research/ade_wildgen_mckay_spec_2026-07-26.md"
PETITOT="${REPO_ROOT}/docs/research/petitot_potential.py"

if [[ ! -f "${CONTRACT}" ]]; then
    echo "ADE_WILDGEN_GATE_FAIL: missing ${CONTRACT}"
    exit 1
fi
if [[ ! -f "${SPEC}" ]]; then
    echo "ADE_WILDGEN_GATE_FAIL: missing spec ${SPEC}"
    exit 1
fi

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "Running ADE-Wildgen McKay contract..."
"${PYTHON}" "${CONTRACT}" | tee /tmp/ade_wildgen_mckay_out.txt
EXIT_CODE=${PIPESTATUS[0]}

if [[ ${EXIT_CODE} -ne 0 ]]; then
    echo "ADE_WILDGEN_GATE_FAIL: contract exited ${EXIT_CODE}"
    exit "${EXIT_CODE}"
fi

grep -q 'ADE_WILDGEN_MCKAY_OK' /tmp/ade_wildgen_mckay_out.txt
grep -q 'ADE_WILDGEN_VERDICT STRONG_FORM_OBSTRUCTED__WEAK_FORM_UNDECIDABLE' /tmp/ade_wildgen_mckay_out.txt
grep -q 'M3_MCKAY_FUSION_E6' /tmp/ade_wildgen_mckay_out.txt
grep -q 'M3_MCKAY_FUSION_E7' /tmp/ade_wildgen_mckay_out.txt
grep -q 'M3_MCKAY_FUSION_E8' /tmp/ade_wildgen_mckay_out.txt
grep -q 'C1_G2_EXCLUDED_FROM_SU2_MCKAY' /tmp/ade_wildgen_mckay_out.txt
grep -q 'C2_FANO_FINITE_CONTENT' /tmp/ade_wildgen_mckay_out.txt
grep -q 'C3_GERMS_MILNOR' /tmp/ade_wildgen_mckay_out.txt
grep -q 'C4_MAGIC_SQUARE_LINK' /tmp/ade_wildgen_mckay_out.txt
if grep -q 'CONTRACT_INCOMPLETE' /tmp/ade_wildgen_mckay_out.txt; then
    echo "ADE_WILDGEN_GATE_FAIL: contract incomplete"
    exit 1
fi

# Semantic-side premise: the programme's operative Petitot germs are A-series.
grep -q 'x^4/4' "${PETITOT}"
grep -q 'x^6/6' "${PETITOT}"

# Spec must carry the verdict and the decidability statement.
grep -q 'STRONG_FORM_OBSTRUCTED__WEAK_FORM_UNDECIDABLE' "${SPEC}"
grep -q 'currently undecidable' "${SPEC}"

echo "ADE_WILDGEN_MCKAY_GATE_OK"
