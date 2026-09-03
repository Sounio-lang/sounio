#!/usr/bin/env bash
# CI gate for the Mercyful Learning LEARNED suffering field contract.
#
# Contract: scripts/research/mercyful_learned_field_contract.py (L1..L8)
# Model:    scripts/research/mercyful_suffering_field_learned.py
# Frozen:   scripts/research/mercyful_learned_field_coefficients_v1.json
# Spec:     docs/research/mercyful_learned_suffering_field_spec_2026-07-26.md
#
# Convention: like the other mercyful gates, this gate is invoked
# manually/locally and referenced from its spec; the mercyful family is
# deliberately not wired into .github/workflows/ci.yml (a shared control
# file listed in AGENTS.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/mercyful_learned_field_contract.py"
FROZEN="${REPO_ROOT}/scripts/research/mercyful_learned_field_coefficients_v1.json"
SPEC="${REPO_ROOT}/docs/research/mercyful_learned_suffering_field_spec_2026-07-26.md"
FAERS_AUDIT="${REPO_ROOT}/docs/research/faers_mercyful_analysis_2026-07-26.md"

fail() {
    echo "MERCYFUL_LEARNED_FIELD_GATE_FAIL: $*" >&2
    exit 1
}

PYTHON="${REPO_ROOT}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

# L1..L8: Python contract.
[[ -f "${CONTRACT}" ]] || fail "missing ${CONTRACT}"
[[ -f "${FROZEN}" ]] || fail "missing ${FROZEN}"
PY_OUTPUT=$("${PYTHON}" "${CONTRACT}" 2>&1) || fail "python contract failed"
for clause in \
    L1_DATA_PROVENANCE \
    L2_OUTCOME_MODELS_LEARN \
    L3_FIELD_DECOMPOSITION \
    L4_TDM_NARROWS_LEARNED_FIELD \
    L5_ANCHOR_ORDERING \
    L6_SCHEDULER_EQUIVALENCE \
    L7_TEACHER_RANK_AGREEMENT \
    L8_NO_OVERREACH; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "${clause}.*PASS" || fail "python clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'MERCYFUL_LEARNED_FIELD_VERDICT L_GREEN' || fail "python verdict not L_GREEN"
echo "L1_L8_PYTHON_CONTRACT PASS"

# L9: canonical learned-field numbers (deterministic; spec section 7.2).
for num in 0.637356 0.285521 0.954318 0.724605; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "${num}" || fail "canonical learned number ${num} missing"
done
echo "L9_CANONICAL_NUMBERS PASS"

# L10: learned-vs-synthetic comparison anchors present (spec section 7.2).
for num in 0.675679 0.059420; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "${num}" || fail "synthetic analog number ${num} missing"
done
echo "L10_SYNTHETIC_ANALOGS PASS"

# L11: scope guards and negative provenance in the spec.
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'faers_mercyful_analysis_2026-07-26' "${SPEC}" || fail "missing FAERS negative provenance in spec"
grep -Fq '10.1038/s41598-026-42395-1' "${SPEC}" || fail "missing MIMIC-IV study DOI in spec"
grep -Fiq 'machine suffering' "${SPEC}" || fail "missing machine-suffering section in spec"
[[ -f "${FAERS_AUDIT}" ]] || fail "missing ${FAERS_AUDIT}"
grep -Fq 'Verdict: NEGATIVE' "${FAERS_AUDIT}" || fail "FAERS audit no longer records NEGATIVE verdict"
echo "L11_SCOPE_GUARDS PASS"

echo "MERCYFUL_LEARNED_FIELD_GATE_OK"
