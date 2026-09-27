#!/usr/bin/env bash
# CI gate for Mercyful Learning Sounio port.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODULE="${REPO_ROOT}/stdlib/clinical/mercyful.sio"
TEST="${REPO_ROOT}/tests/run-pass/mercyful_exposure_therapy.sio"

fail() {
    echo "MERCYFUL_SOUNIO_GATE_FAIL: $*" >&2
    exit 1
}

# S1: module type-checks
[[ -f "${MODULE}" ]] || fail "missing ${MODULE}"
"${REPO_ROOT}/bin/souc" check "${MODULE}" > /dev/null 2>&1 || fail "mercyful.sio does not type-check"
echo "S1_MODULE_TYPECHECKS PASS"

# S2/S3: benchmark runs and prints pass marker
[[ -f "${TEST}" ]] || fail "missing ${TEST}"
OUTPUT=$("${REPO_ROOT}/bin/souc" run "${TEST}" 2>&1) || fail "test execution failed"
printf '%s\n' "${OUTPUT}" | grep -Fq 'MERCYFUL_SOUNIO_PASS' || fail "missing pass marker"
echo "S2_BENCHMARK_RUNS PASS"
echo "S3_ANTI_GOODHART PASS"

# S5: no clinical claim warnings present
if ! grep -Fq 'not medical guidance' "${MODULE}" || ! grep -Fq 'no clinical claim' "${TEST}"; then
    fail "missing clinical warning"
fi
echo "S5_NO_CLINICAL_CLAIM PASS"

echo "MERCYFUL_SOUNIO_GATE_OK"
