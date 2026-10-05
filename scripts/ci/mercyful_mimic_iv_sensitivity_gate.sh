#!/usr/bin/env bash
# CI gate for Mercyful Learning x MIMIC-IV vancomycin TDM sensitivity analysis.
#
# Contract: scripts/research/mercyful_mimic_iv_sensitivity_contract.py (S1..S7)
# Spec:     docs/research/mimic_iv_sensitivity_analysis_2026-07-26.md
# Parent:   scripts/ci/mercyful_mimic_iv_gate.sh (V1..V10 gate, unchanged)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/mercyful_mimic_iv_sensitivity_contract.py"
SPEC="${REPO_ROOT}/docs/research/mimic_iv_sensitivity_analysis_2026-07-26.md"
PARENT_CONTRACT="${REPO_ROOT}/scripts/research/mercyful_mimic_iv_vancomycin_contract.py"

fail() {
    echo "MERCYFUL_MIMIC_IV_SENSITIVITY_GATE_FAIL: $*" >&2
    exit 1
}

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

# S1..S7: Python contract.
[[ -f "${CONTRACT}" ]] || fail "missing ${CONTRACT}"
PY_OUTPUT=$("${PYTHON}" "${CONTRACT}" 2>&1) || fail "python contract failed"
for clause in \
    S1_DECLARATION_CONSISTENCY \
    S2_GATED_SELECTS_TDM_ALL_CELLS \
    S3_TWIN_ANCHORED_REFERENCE \
    S4_OPEN_GATE_LANDSCAPE \
    S5_GATE_IS_CAUSAL_ROBUST \
    S6_MIMIC_IV_DIRECTION_UNCHANGED \
    S7_NO_OVERREACH; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "${clause}.*PASS" || fail "python clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'MERCYFUL_MIMIC_IV_SENSITIVITY_VERDICT S_GREEN' || fail "python verdict not S_GREEN"
echo "S1_S7_PYTHON_CONTRACT PASS"

# S8: canonical numbers — parent V5 anchor, frozen landscape counts, crossover.
for num in 0.735099 0.675679 1.410778 1.443156 1792/1792 const_std=110 flips=16; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "${num}" || fail "canonical number ${num} missing"
done
echo "S8_CANONICAL_NUMBERS PASS"

# S9: frozen per-mu strict TDM-win counts and tie counts present verbatim.
for stat in "0.0: 110" "1.0: 121" "20.0: 126" "const_tie=20" "scheduler_xcheck=1648/1648"; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "${stat}" || fail "frozen count ${stat} missing from contract output"
done
echo "S9_FROZEN_COUNTS PASS"

# S10: parent contract still green (sensitivity analysis must not regress V1..V7).
[[ -f "${PARENT_CONTRACT}" ]] || fail "missing ${PARENT_CONTRACT}"
PARENT_OUTPUT=$("${PYTHON}" "${PARENT_CONTRACT}" 2>&1) || fail "parent contract failed"
printf '%s\n' "${PARENT_OUTPUT}" | grep -Fq 'MERCYFUL_MIMIC_IV_VERDICT V_GREEN' || fail "parent verdict not V_GREEN"
echo "S10_PARENT_CONTRACT_STILL_GREEN PASS"

# S11: scope guards and honest-negative reporting in the spec.
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'conclusion does not' "${SPEC}" || fail "spec no longer reports the open-gate conclusion change"
grep -Fq '110/256' "${SPEC}" || fail "spec missing frozen const-STD landscape count"
grep -Fq '1.443156' "${SPEC}" || fail "spec missing twin-anchored crossover mu*"
echo "S11_SCOPE_GUARDS PASS"

echo "MERCYFUL_MIMIC_IV_SENSITIVITY_GATE_OK"
