#!/usr/bin/env bash
# CI gate for Mercyful Learning x INDEPENDENT vancomycin TDM datasets
# structural correspondence contract (cross-dataset validation).
#
# Contract: scripts/research/mercyful_independent_tdm_contract.py (I1..I7)
# Report:   docs/research/independent_dataset_vancomycin_tdm_validation_2026-07-26.md
# Anchor:   scripts/research/mercyful_mimic_iv_vancomycin_contract.py (must stay V_GREEN)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/mercyful_independent_tdm_contract.py"
ANCHOR="${REPO_ROOT}/scripts/research/mercyful_mimic_iv_vancomycin_contract.py"
REPORT="${REPO_ROOT}/docs/research/independent_dataset_vancomycin_tdm_validation_2026-07-26.md"

fail() {
    echo "MERCYFUL_INDEPENDENT_TDM_GATE_FAIL: $*" >&2
    exit 1
}

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

# I1..I7: Python contract.
[[ -f "${CONTRACT}" ]] || fail "missing ${CONTRACT}"
PY_OUTPUT=$("${PYTHON}" "${CONTRACT}" 2>&1) || fail "python contract failed"
for clause in \
    I1_INDEPENDENCE_FROM_MIMIC \
    I2_EFFICACY_DIRECTION_MATCH \
    I3_TOXICITY_DIRECTION_MATCH \
    I4_SCHEDULER_UNCHANGED_STILL_SELECTS_TDM \
    I5_VERIFY_GATE_STILL_CAUSAL \
    I6_EICU_BOUNDARY_CONDITION \
    I7_NO_OVERREACH; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "${clause}.*PASS" || fail "python clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'MERCYFUL_INDEPENDENT_TDM_VERDICT I_GREEN' || fail "python verdict not I_GREEN"
echo "I1_I7_PYTHON_CONTRACT PASS"

# I8: no-refit canonical numbers present (graph identity with anchor).
for num in 0.735099 0.675679 1.410778; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "${num}" || fail "canonical number ${num} missing"
done
# The contract must IMPORT the frozen anchor graph, not redeclare it.
grep -Fq 'from mercyful_mimic_iv_vancomycin_contract import' "${CONTRACT}" \
    || fail "contract does not import the frozen anchor graph (no-refit guarantee broken)"
echo "I8_NO_REFIT PASS"

# I9: independent statistics present in contract output.
for stat in 2.62 1.34 5.11 0.25 0.13 0.48 521 2.428 1.385 4.258; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "${stat}" || fail "independent statistic ${stat} missing from contract output"
done
echo "I9_INDEPENDENT_STATS_PRESENT PASS"

# I10: boundary-condition honesty markers survive in contract output.
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'trough_targeting_benefit_replicated=False' \
    || fail "eICU non-replication marker missing (boundary condition suppressed)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'mortality_not_independently_replicated' \
    || fail "mortality-scope note missing from verdict line"
echo "I10_BOUNDARY_HONESTY PASS"

# I11: scope guards and citations in the report.
[[ -f "${REPORT}" ]] || fail "missing ${REPORT}"
grep -Fq 'not medical guidance' "${REPORT}" || fail "missing clinical warning in report"
grep -Fq 'synthetic' "${REPORT}" || fail "missing synthetic-data statement in report"
grep -Fq '10.1371/journal.pone.0077169' "${REPORT}" || fail "missing Ye 2013 DOI in report"
grep -Fq '10.1002/jcph.2363' "${REPORT}" || fail "missing Yang 2024 DOI in report"
grep -Fq '10.3389/fphar.2021.690157' "${REPORT}" || fail "missing Hou 2021 DOI in report"
grep -Fq 'boundary condition' "${REPORT}" || fail "missing eICU boundary-condition section in report"
grep -Fq 'not independently replicated' "${REPORT}" || fail "missing mortality non-replication statement in report"
echo "I11_REPORT_GUARDS PASS"

# I12: anchor contract still green (the frozen graph this contract imports).
[[ -f "${ANCHOR}" ]] || fail "missing ${ANCHOR}"
ANCHOR_OUTPUT=$("${PYTHON}" "${ANCHOR}" 2>&1) || fail "anchor contract failed"
printf '%s\n' "${ANCHOR_OUTPUT}" | grep -Fq 'MERCYFUL_MIMIC_IV_VERDICT V_GREEN' \
    || fail "anchor MIMIC-IV contract no longer V_GREEN"
echo "I12_ANCHOR_STILL_GREEN PASS"

echo "MERCYFUL_INDEPENDENT_TDM_GATE_OK"
