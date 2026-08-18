#!/usr/bin/env bash
# CI gate for the Mercyful Learning x MIMIC-IV vancomycin TDM subgroup
# cross-validation contract.
#
# Contract: scripts/research/mercyful_mimic_iv_subgroup_contract.py (X1..X9)
# Spec:     docs/research/mimic_iv_subgroup_cross_validation_2026-07-26.md
# Parent:   scripts/ci/mercyful_mimic_iv_gate.sh (V1..V7 contract gate)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/mercyful_mimic_iv_subgroup_contract.py"
SPEC="${REPO_ROOT}/docs/research/mimic_iv_subgroup_cross_validation_2026-07-26.md"
COHORT="${REPO_ROOT}/scripts/clinical/data_synthetic/tdm_cohort_synthetic_v2.csv"

fail() {
    echo "MERCYFUL_MIMIC_IV_SUBGROUP_GATE_FAIL: $*" >&2
    exit 1
}

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

# X1..X9: Python contract.
[[ -f "${CONTRACT}" ]] || fail "missing ${CONTRACT}"
[[ -f "${COHORT}" ]] || fail "missing ${COHORT}"
PY_OUTPUT=$("${PYTHON}" "${CONTRACT}" 2>&1) || fail "python contract failed"
for clause in \
    X1_COHORT_SCHEMA_AND_STRATIFICATION \
    X2_POOLED_WINDOW_CURE_ASSOCIATION \
    X3_DIRECTION_HOLDS_ALL_STRATA \
    X4_SCHEDULER_SELECTS_TDM_ALL_STRATA \
    X5_NAIVE_MINIMIZER_UNDERDOSES_ALL_STRATA \
    X6_VERIFY_GATE_CAUSAL_ALL_STRATA \
    X7_SUFFERING_GRADIENT \
    X8_LITERATURE_STRATUM_ROBUSTNESS \
    X9_NO_OVERREACH; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "${clause}.*PASS" || fail "python clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'MERCYFUL_MIMIC_IV_SUBGROUP_VERDICT X_GREEN' || fail "python verdict not X_GREEN"
echo "X1_X9_PYTHON_CONTRACT PASS"

# X10: canonical numbers present (C1 scheduler values, pooled OR, stratum ORs).
for num in 0.735099 0.675679 1.410778 2.733 2.269 3.370 2.404 2.893 2.112 4.306; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "${num}" || fail "canonical number ${num} missing"
done
echo "X10_CANONICAL_NUMBERS PASS"

# X11: real MIMIC-IV stratified statistics present in contract output
# (adjustment strata for both mortality endpoints).
for stat in 0.49 0.58 0.63 0.672 0.51 0.64 0.72 0.691; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "${stat}" || fail "MIMIC-IV stratified statistic ${stat} missing from contract output"
done
echo "X11_REAL_STRATIFIED_STATS_PRESENT PASS"

# X12: scope guards and source anchoring in the spec.
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'credential-gated' "${SPEC}" || fail "missing credential-gated scoping statement in spec"
grep -Fq '10.1038/s41598-026-42395-1' "${SPEC}" || fail "missing study DOI in spec"
grep -Fq 'Scoping decision' "${SPEC}" || fail "missing scoping decision section in spec"
echo "X12_SCOPE_GUARDS PASS"

# X13: parent contract still green (cross-validation must not regress the
# pooled validation it extends).
PARENT_GATE="${REPO_ROOT}/scripts/ci/mercyful_mimic_iv_gate.sh"
[[ -f "${PARENT_GATE}" ]] || fail "missing ${PARENT_GATE}"
bash "${PARENT_GATE}" > /dev/null 2>&1 || fail "parent mercyful_mimic_iv_gate.sh not green"
echo "X13_PARENT_GATE_STILL_GREEN PASS"

echo "MERCYFUL_MIMIC_IV_SUBGROUP_GATE_OK"
