#!/usr/bin/env bash
# CI gate for Mercyful Learning x cancer chemotherapy sequencing benchmark.
#
# Spec:           docs/research/mercyful_chemo_sequencing_spec_2026-07-26.md
# Python harness: scripts/research/mercyful_chemo_contract.py (H1..H8)
# Sounio harness: tests/run-pass/mercyful_chemo_sequencing.sio
#
# Execution path: the Sounio test imports stdlib/clinical/mercyful.sio, so it
# MUST run through scripts/dev/run_clinical_twin.sh (lean_single bootstrap
# engine). The default Madaros native path segfaults at runtime for this
# program (multi-module native lowering bug — see
# docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODULE="${REPO_ROOT}/stdlib/clinical/mercyful.sio"
TEST="${REPO_ROOT}/tests/run-pass/mercyful_chemo_sequencing.sio"
PYCONTRACT="${REPO_ROOT}/scripts/research/mercyful_chemo_contract.py"
SPEC="${REPO_ROOT}/docs/research/mercyful_chemo_sequencing_spec_2026-07-26.md"

fail() {
    echo "MERCYFUL_CHEMO_GATE_FAIL: $*" >&2
    exit 1
}

# H0: scheduler module still type-checks.
[[ -f "${MODULE}" ]] || fail "missing ${MODULE}"
"${REPO_ROOT}/bin/souc" check "${MODULE}" > /dev/null 2>&1 || fail "mercyful.sio does not type-check"
echo "H0_MODULE_TYPECHECKS PASS"

# H1..H8: Python contract.
[[ -f "${PYCONTRACT}" ]] || fail "missing ${PYCONTRACT}"
PY_OUTPUT=$(python3 "${PYCONTRACT}" 2>&1) || fail "python contract failed"
for clause in \
    H1_BASELINE \
    H2_ANTI_GOODHART \
    H3_MU_CROSSOVER \
    H4_FRONTIER \
    H5_GCSF_GATE \
    H6_BUDGET_HARDNESS \
    H7_BUDGETARY_NECESSITY \
    H8_CANONICAL; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "${clause}.*PASS" || fail "python clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'MERCYFUL_CHEMO_VERDICT H_GREEN' || fail "python verdict not H_GREEN"
echo "H1_H8_PYTHON_CONTRACT PASS"

# Sounio-native clauses via the lean_single clinical path.
[[ -f "${TEST}" ]] || fail "missing ${TEST}"
OUTPUT=$("${REPO_ROOT}/scripts/dev/run_clinical_twin.sh" "${TEST}" 2>&1) || fail "sounio test execution failed"
for clause in \
    H1_BASELINE \
    H2_ANTI_GOODHART \
    H3_MU_CROSSOVER \
    H5_GCSF_GATE \
    H6_BUDGET_HARDNESS \
    H7_BUDGETARY_NECESSITY; do
    printf '%s\n' "${OUTPUT}" | grep -Fq "${clause} PASS" || fail "sounio clause ${clause} did not pass"
done
printf '%s\n' "${OUTPUT}" | grep -Fq 'MERCYFUL_CHEMO_PASS' || fail "missing pass marker"
echo "H_SOUNIO_NATIVE PASS"

# H8: Sounio and Python agree on every canonical number.
# Python prints 8.0/48.0/56.0 and 24.0/81.0/5.0/86.0; Sounio prints 6dp.
for num in 8.000000 48.000000 56.000000 24.000000 81.000000 5.000000 86.000000; do
    printf '%s\n' "${OUTPUT}" | grep -Fq "${num}" || fail "sounio canonical number ${num} missing (H8 disagreement)"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'gcsf_on=(8.0, 48.0, 8.0, 56.0) gcsf_off=(24.0, 81.0, 5.0, 86.0)' \
    || fail "python canonical numbers missing (H8 disagreement)"
echo "H8_CROSS_IMPLEMENTATION_AGREEMENT PASS"

# H9: no-clinical-claim warnings present in test and spec.
grep -Fq 'not medical guidance' "${TEST}" || fail "missing clinical warning in test"
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
echo "H9_NO_CLINICAL_CLAIM PASS"

echo "MERCYFUL_CHEMO_GATE_OK"
