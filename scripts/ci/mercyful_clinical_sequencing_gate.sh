#!/usr/bin/env bash
# CI gate for Mercyful Learning x clinical PK twins treatment-sequencing
# scheduler.
#
# Spec:    docs/research/mercyful_clinical_integration_spec_2026-07-25.md
# Harness: tests/run-pass/mercyful_clinical_sequencing.sio
#
# Execution path: the test imports three stdlib/clinical modules plus
# epistemic::knightian, so it MUST run through scripts/dev/run_clinical_twin.sh
# (lean_single bootstrap engine), not the default Madaros `bin/souc run`
# (multi-module native-v2 segfault — see stdlib/clinical/README.md and
# docs/audit/MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODULE="${REPO_ROOT}/stdlib/clinical/mercyful.sio"
TEST="${REPO_ROOT}/tests/run-pass/mercyful_clinical_sequencing.sio"

fail() {
    echo "MERCYFUL_CLINICAL_SEQ_GATE_FAIL: $*" >&2
    exit 1
}

# C0: scheduler module still type-checks after the flat-queue patch.
[[ -f "${MODULE}" ]] || fail "missing ${MODULE}"
"${REPO_ROOT}/bin/souc" check "${MODULE}" > /dev/null 2>&1 || fail "mercyful.sio does not type-check"
echo "C0_MODULE_TYPECHECKS PASS"

# C1..C6: run the sequencing contract via the lean_single clinical path.
[[ -f "${TEST}" ]] || fail "missing ${TEST}"
OUTPUT=$("${REPO_ROOT}/scripts/dev/run_clinical_twin.sh" "${TEST}" 2>&1) || fail "test execution failed"

for clause in \
    C1_PATH_FOUND \
    C2_GOODHART_SHORTCUT_BLOCKED \
    C3_TDM_REDUCES_SUFFERING \
    C4_CONTRACT_VIOLATION_PENALTY \
    C5_DDI_NEPHROTOXIN_INFEASIBLE \
    C6_DDI_CYP_GATE_BLOCKS; do
    printf '%s\n' "${OUTPUT}" | grep -Fq "${clause} PASS" || fail "clause ${clause} did not pass"
    echo "${clause} PASS"
done

printf '%s\n' "${OUTPUT}" | grep -Fq 'MERCYFUL_CLINICAL_SEQ_PASS' || fail "missing pass marker"

# C7: no-clinical-claim warnings present in test and spec.
grep -Fq 'not medical guidance' "${TEST}" || fail "missing clinical warning in test"
grep -Fq 'Not medical guidance' "${REPO_ROOT}/docs/research/mercyful_clinical_integration_spec_2026-07-25.md" \
    || fail "missing clinical warning in spec"
echo "C7_NO_CLINICAL_CLAIM PASS"

echo "MERCYFUL_CLINICAL_SEQ_GATE_OK"
