#!/usr/bin/env bash
# CI gate for the Mercyful Learning expanded-ethics mathematics (Task 3):
# suffering minimization as the antithesis of RL, patient + machine channels.
#
# Spec:     docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md
# Contract: scripts/research/mercyful_expanded_ethics_contract.py (E1..E8)
# Parent:   scripts/research/mercyful_runtime_contract.py (M1..M6, imported)
#
# Self-contained: intentionally NOT wired into .github/workflows/ci.yml yet
# (shared control file under active edit by other lanes on this branch);
# wiring is left to the integrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONTRACT="${REPO_ROOT}/scripts/research/mercyful_expanded_ethics_contract.py"
PARENT="${REPO_ROOT}/scripts/research/mercyful_runtime_contract.py"
SPEC="${REPO_ROOT}/docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md"

fail() {
    echo "MERCYFUL_EXPANDED_ETHICS_GATE_FAIL: $*" >&2
    exit 1
}

# X1: contract exists and runs green.
[[ -f "${CONTRACT}" ]] || fail "missing ${CONTRACT}"
X1_OUTPUT=$(python3 "${CONTRACT}" 2>&1) || fail "contract exited non-zero: ${X1_OUTPUT}"
printf '%s\n' "${X1_OUTPUT}" | grep -Fq 'MERCYFUL_EXPANDED_ETHICS_VERDICT E_GREEN (8/8 clauses PASS)' \
    || fail "contract not E_GREEN 8/8"
echo "X1_CONTRACT_GREEN PASS"

# X2: canonical numbers cross-check (spec section 7 anchors).
printf '%s\n' "${X1_OUTPUT}" | grep -Fq 'lambda*=0.666667' \
    || fail "lambda* crossover missing or wrong (expected 2/3)"
printf '%s\n' "${X1_OUTPUT}" | grep -Fq 'gap=2.500' \
    || fail "E5 optimality gap missing or wrong (expected 2.5)"
printf '%s\n' "${X1_OUTPUT}" | grep -Fq 'course=(Jp 12.0, Jm 3.0)' \
    || fail "E7 two-channel course costs missing or wrong"
echo "X2_CANONICAL_NUMBERS PASS"

# X3: parent runtime contract still green (cross-implementation agreement —
# the expanded contract imports MercyGraph/enumerate_paths from it).
[[ -f "${PARENT}" ]] || fail "missing ${PARENT}"
PARENT_OUTPUT=$(python3 "${PARENT}" 2>&1) || fail "parent contract failed"
printf '%s\n' "${PARENT_OUTPUT}" | grep -Fq 'MERCYFUL_RUNTIME_VERDICT M_GREEN' \
    || fail "parent runtime contract not M_GREEN"
echo "X3_PARENT_CONTRACT_GREEN PASS"

# X4: spec exists with the four required components (a)-(d).
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
grep -Fq 'Suffering as a first-class cost' "${SPEC}" || fail "spec missing component (a)"
grep -Fq 'multi-objective optimization' "${SPEC}" || fail "spec missing component (b)"
grep -Fq 'Utilitarianism' "${SPEC}" || fail "spec missing utilitarianism mapping"
grep -Fq 'Deontology' "${SPEC}" || fail "spec missing deontology mapping"
grep -Fq 'Care ethics' "${SPEC}" || fail "spec missing care-ethics mapping"
grep -Fq 'Lipschitz stability' "${SPEC}" || fail "spec missing stability theorem (d)"
grep -Fq 'Knightian robustness' "${SPEC}" || fail "spec missing robustness theorem (d)"
grep -Fq 'convexity in the field' "${SPEC}" || fail "spec missing convexity theorem (d)"
echo "X4_SPEC_COMPONENTS PASS"

# X5: scope guards — no clinical overreach, no machine-consciousness claim,
# no novelty overclaim.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'no claim of machine consciousness' "${SPEC}" \
    || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'no new algorithmic primitive' "${SPEC}" || fail "missing novelty disclaimer in spec"
printf '%s\n' "${X1_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "contract output missing no_consciousness_claim note"
echo "X5_SCOPE_GUARDS PASS"

echo "MERCYFUL_EXPANDED_ETHICS_GATE_OK"
