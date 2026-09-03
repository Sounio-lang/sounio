#!/usr/bin/env bash
# CI gate for Mercyful Learning x MIMIC-IV vancomycin TDM structural
# correspondence contract.
#
# Contract: scripts/research/mercyful_mimic_iv_vancomycin_contract.py (V1..V7)
# Report:   docs/research/mimic_iv_mercyful_validation_2026-07-26.md
# Paper:    docs/papers/mercyful_learning_medical_paper_2026-07-26.md (sec 8.3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/mercyful_mimic_iv_vancomycin_contract.py"
REPORT="${REPO_ROOT}/docs/research/mimic_iv_mercyful_validation_2026-07-26.md"
PAPER="${REPO_ROOT}/docs/papers/mercyful_learning_medical_paper_2026-07-26.md"

fail() {
    echo "MERCYFUL_MIMIC_IV_GATE_FAIL: $*" >&2
    exit 1
}

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

# V1..V7: Python contract.
[[ -f "${CONTRACT}" ]] || fail "missing ${CONTRACT}"
PY_OUTPUT=$("${PYTHON}" "${CONTRACT}" 2>&1) || fail "python contract failed"
for clause in \
    V1_NAIVE_TOXICITY_MINIMIZER_UNDERDOSES \
    V2_RAW_MINIMIZER_NEVER_TREATS \
    V3_TDM_NARROWS_FIELD \
    V4_VERIFY_GATE_IS_CAUSAL \
    V5_MERCYFUL_SELECTS_TDM \
    V6_MIMIC_IV_DIRECTION_MATCH \
    V7_NO_OVERREACH; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "${clause}.*PASS" || fail "python clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'MERCYFUL_MIMIC_IV_VERDICT V_GREEN' || fail "python verdict not V_GREEN"
echo "V1_V7_PYTHON_CONTRACT PASS"

# V8: canonical numbers agree with the clinical twin (clause C1/C3 values).
for num in 0.675679 0.059420 0.735099 1.410778; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "${num}" || fail "canonical number ${num} missing"
done
echo "V8_CANONICAL_NUMBERS PASS"

# V9: real MIMIC-IV statistics present in contract output.
for stat in 28451 10758 0.691 0.672 0.58; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "${stat}" || fail "MIMIC-IV statistic ${stat} missing from contract output"
done
echo "V9_REAL_STATS_PRESENT PASS"

# V10: scope guards in report and paper.
[[ -f "${REPORT}" ]] || fail "missing ${REPORT}"
grep -Fq 'not medical guidance' "${REPORT}" || fail "missing clinical warning in report"
grep -Fq 'synthetic' "${REPORT}" || fail "missing synthetic-data statement in report"
grep -Fq 'not medical guidance' "${PAPER}" || fail "missing clinical warning in paper"
grep -Fq '10.1038/s41598-026-42395-1' "${REPORT}" || fail "missing study DOI in report"
grep -Fq '10.1038/s41598-026-42395-1' "${PAPER}" || fail "missing study DOI in paper"
echo "V10_SCOPE_GUARDS PASS"

echo "MERCYFUL_MIMIC_IV_GATE_OK"
