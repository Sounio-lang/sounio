#!/usr/bin/env bash
# CI gate for Mercyful Learning x continuous optimal control (Pontryagin rung).
#
# Spec:           docs/research/mercyful_pontryagin_control_spec_2026-07-26.md
# Python harness: scripts/research/mercyful_pontryagin_control_contract.py (K1..K9)
# Sounio harness: tests/run-pass/mercyful_pontryagin_control.sio (K1, K4, K5, K6 native)
#
# Execution path: the Sounio test is self-contained (no module imports) but is
# run through scripts/dev/run_clinical_twin.sh (lean_single bootstrap engine)
# for consistency with the clinical-twin path; the default Madaros native path
# has an open multi-module lowering bug
# (docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md) and is not
# the validated engine for this tree of benchmarks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TEST="${REPO_ROOT}/tests/run-pass/mercyful_pontryagin_control.sio"
PYCONTRACT="${REPO_ROOT}/scripts/research/mercyful_pontryagin_control_contract.py"
SPEC="${REPO_ROOT}/docs/research/mercyful_pontryagin_control_spec_2026-07-26.md"

fail() {
    echo "MERCYFUL_PONTRYAGIN_GATE_FAIL: $*" >&2
    exit 1
}

# K1..K9: Python contract.
[[ -f "${PYCONTRACT}" ]] || fail "missing ${PYCONTRACT}"
PY_OUTPUT=$(python3 "${PYCONTRACT}" 2>&1) || fail "python contract failed"
for clause in \
    K1_BASELINE \
    K2_ANTI_GOODHART \
    K3_BANG_BANG \
    K4_EQUIOSCILLATION \
    K5_SMOOTH_CROSSOVER \
    K6_INFEASIBLE \
    K7_TWO_MERCIES \
    K8_PMP_BOUNDARY_ARC \
    K9_CANONICAL; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "${clause}.*PASS" || fail "python clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'MERCYFUL_PONTRYAGIN_VERDICT K_GREEN' || fail "python verdict not K_GREEN"
echo "K1_K9_PYTHON_CONTRACT PASS"

# Sounio-native clauses via the lean_single clinical path.
[[ -f "${TEST}" ]] || fail "missing ${TEST}"
OUTPUT=$("${REPO_ROOT}/scripts/dev/run_clinical_twin.sh" "${TEST}" 2>&1) || fail "sounio test execution failed"
for clause in \
    K1_BASELINE \
    K4_FRONTIER \
    K5_SMOOTH_CROSSOVER \
    K6_BUDGETARY_NECESSITY; do
    printf '%s\n' "${OUTPUT}" | grep -Fq "${clause} PASS" || fail "sounio clause ${clause} did not pass"
done
printf '%s\n' "${OUTPUT}" | grep -Fq 'MERCYFUL_PONTRYAGIN_PASS' || fail "missing pass marker"
echo "K_SOUNIO_NATIVE PASS"

# K9: Sounio and Python agree on every canonical number.
# Both print 6dp; lean_single println concatenates without newlines, so these
# are substring checks (same convention as the chemo rung gate).
for num in 2.995732 1.497866 3.000000 3.470732 3.663562 1.831781 4.060443 2.706962 1.867628 1.402552; do
    printf '%s\n' "${OUTPUT}" | grep -Fq "${num}" || fail "sounio canonical number ${num} missing (K9 disagreement)"
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "${num}" || fail "python canonical number ${num} missing (K9 disagreement)"
done
echo "K9_CROSS_IMPLEMENTATION_AGREEMENT PASS"

# K10: no-clinical-claim warnings present in test and spec.
grep -Fq 'not medical guidance' "${TEST}" || fail "missing clinical warning in test"
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'not medical guidance' "${PYCONTRACT}" || fail "missing clinical warning in python contract"
echo "K10_NO_CLINICAL_CLAIM PASS"

echo "MERCYFUL_PONTRYAGIN_GATE_OK"
