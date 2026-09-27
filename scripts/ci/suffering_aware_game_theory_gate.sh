#!/usr/bin/env bash
# CI gate for SAMA-GT: game theory of the Mercyful Learning Suffering-Aware
# Multi-Agent system (Nash equilibrium, mechanism design, fair division of
# suffering).
#
# Spec:    docs/research/suffering_aware_game_theory_spec_2026-07-31.md
# Harness: scripts/research/suffering_aware_game_theory.py (GT1..GT8)
# Parent:  scripts/ci/suffering_aware_multi_agent_gate.sh (SAMA contract G1..G8)
#
# Execution path: repo .venv Python (numpy only), importing the pinned SAMA
# harness as a library. Pure synthetic data; no Sounio-native leg (Python
# reference implementation; scope note in spec section 9). Self-contained:
# intentionally NOT wired into .github/workflows/ci.yml yet (shared control
# file under active edit by other lanes on this branch); wiring is left to
# the integrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_game_theory.py"
BASE_HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_multi_agent.py"
SPEC="${REPO_ROOT}/docs/research/suffering_aware_game_theory_spec_2026-07-31.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"

fail() {
    echo "SAMA_GAME_THEORY_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${BASE_HARNESS}" ]] || fail "missing ${BASE_HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# GT1..GT8: game-theory contract.
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "SAMA-GT harness failed to run"
for clause in GT1 GT2 GT3 GT4 GT5 GT6 GT7 GT8; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}: PASS" || fail "clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAMA_GAME_THEORY_VERDICT GT_GREEN (8/8 clauses PASS)' \
    || fail "verdict not GT_GREEN 8/8"
echo "GT1_GT8_CONTRACT PASS"

# C9: canonical numbers cross-check (spec sections 6 / 7 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'anchor vs pinned SAMA harness: exact=True' \
    || fail "harness does not reproduce pinned SAMA numbers"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'phi3(e=0)=+0.0464' \
    || fail "abstainer harm share missing or wrong (expected +0.0464)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'PoA=(10.67, 14.04)' \
    || fail "mechanism-M price of anarchy missing or wrong (expected 10.67/14.04)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'worst-NE PoA=(0.8533, 1.3911)' \
    || fail "mechanism-M+ worst-NE price of anarchy missing or wrong"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'eff_err=0.0e+00' \
    || fail "Shapley efficiency error missing or wrong (expected 0.0e+00)"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (model, equilibrium, mechanism design, fair division,
# repair, theorems).
grep -Fq 'formal model' "${SPEC}" || fail "spec missing formal model"
grep -Fq 'Nash equilibrium' "${SPEC}" || fail "spec missing Nash equilibrium analysis"
grep -Fq 'dominant-strategy truthful reporting' "${SPEC}" || fail "spec missing T-GT2"
grep -Fq 'minority immunity' "${SPEC}" || fail "spec missing T-GT3"
grep -Fq 'incentive scope' "${SPEC}" || fail "spec missing T-GT4"
grep -Fq 'liveness failure at equilibrium' "${SPEC}" || fail "spec missing T-GT5"
grep -Fq 'unique fair division' "${SPEC}" || fail "spec missing T-GT7"
grep -Fq 'Rawlsian patient protection' "${SPEC}" || fail "spec missing T-GT8"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim,
# non-scalarization of the suffering pair preserved.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
echo "C11_SCOPE_GUARDS PASS"

echo "SAMA_GAME_THEORY_GATE_OK"
