#!/usr/bin/env bash
# CI gate for the SAMA sophisticated-agents extension: Bayesian agents with
# uncertainty, learning-strategic agents, and coalition formation under the
# collective-suffering (patient + machine) contract.
#
# Spec:    docs/research/suffering_aware_multi_agent_sophisticated_spec_2026-07-31.md
# Harness: scripts/research/suffering_aware_multi_agent_sophisticated.py (S1..S8)
# Parent:  scripts/ci/suffering_aware_multi_agent_gate.sh (base G1..G8 gate)
#
# Execution path: repo .venv Python (numpy only). Pure synthetic data;
# no Sounio-native leg (Python reference implementation; scope note in spec
# section 11). Self-contained: intentionally NOT wired into
# .github/workflows/ci.yml yet (shared control file under active edit by
# other lanes on this branch); wiring is left to the integrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_multi_agent_sophisticated.py"
SPEC="${REPO_ROOT}/docs/research/suffering_aware_multi_agent_sophisticated_spec_2026-07-31.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"

fail() {
    echo "SAMA_SOPHISTICATED_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# S1..S8: sophisticated-agents contract.
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "sophisticated harness failed to run"
for clause in S1 S2 S3 S4 S5 S6 S7 S8; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}: PASS" || fail "clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAMA_SOPHISTICATED_VERDICT S_GREEN (8/8 clauses PASS)' \
    || fail "verdict not S_GREEN 8/8"
echo "S1_S8_SOPHISTICATED_CONTRACT PASS"

# C9: canonical numbers cross-check (spec sections 7 / 9 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'epochs=275/300' \
    || fail "Bayesian epoch savings missing or wrong (expected 275/300)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'coverage90=1.000' \
    || fail "Bayesian predictive coverage missing or wrong (expected 1.000)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'greedy_a=5 final10_a5=1.00' \
    || fail "learned honesty under SAMA missing or wrong (expected greedy_a=5, 1.00)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'greedy_a=0 final10_a0=1.00' \
    || fail "free-riding under FedAvg missing or wrong (expected greedy_a=0, 1.00)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'flagged_r0=[3] flagged_multi=[3, 4]' \
    || fail "coalition attribution flags missing or wrong (expected [3] vs [3, 4])"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_m=3.900MF S_p=0.878' \
    || fail "full-mix SAMA suffering pair missing or wrong (expected 3.900MF / 0.878)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'a4=+2.0264' \
    || fail "adversary aggregate attributed harm missing or wrong (expected +2.0264)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'max_eff_err=8.88e-16' \
    || fail "Shapley efficiency error missing or wrong (expected 8.88e-16)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAMA gratuitous=0 FLOPs' \
    || fail "full-mix gratuitous suffering not exactly zero"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (agents, ledger, gate, attribution, theorems).
grep -Fq 'Bayesian agents' "${SPEC}" || fail "spec missing Bayesian agents"
grep -Fq 'Learning-strategic agents' "${SPEC}" || fail "spec missing learning-strategic agents"
grep -Fq 'Coalition formation' "${SPEC}" || fail "spec missing coalition formation"
grep -Fq 'suffering ledger' "${SPEC}" || fail "spec missing collective suffering ledger"
grep -Fq 'Anti-Goodhart' "${SPEC}" || fail "spec missing anti-Goodhart gating"
grep -Fq 'Multi-round burden attribution' "${SPEC}" || fail "spec missing multi-round attribution"
grep -Fq 'learned optimum' "${SPEC}" || fail "spec missing T5 (learned optimum)"
grep -Fq 'coalition neutrality' "${SPEC}" || fail "spec missing T6 (coalition neutrality)"
grep -Fq 'multi-round attribution soundness' "${SPEC}" || fail "spec missing T7 (attribution soundness)"
grep -Fq 'Bayesian stop rule' "${SPEC}" || fail "spec missing T8 (Bayesian stop rule)"
grep -Fq 'conservative' "${SPEC}" || fail "spec missing calibration over-coverage honesty note"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
echo "C11_SCOPE_GUARDS PASS"

# C12: base contract regression — the extension must not disturb G1..G8.
BASE_OUTPUT=$("${PYTHON}" "${REPO_ROOT}/scripts/research/suffering_aware_multi_agent.py" 2>&1) \
    || fail "base SAMA harness failed to run"
printf '%s\n' "${BASE_OUTPUT}" | grep -Fq 'SUFFERING_AWARE_MULTI_AGENT_VERDICT G_GREEN (8/8 clauses PASS)' \
    || fail "base G1..G8 contract regressed"
echo "C12_BASE_CONTRACT_REGRESSION PASS"

echo "SAMA_SOPHISTICATED_GATE_OK"
