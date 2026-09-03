#!/usr/bin/env bash
# CI gate for the Mercyful Learning machine-channel structural benchmark
# (OPUS 5, critique 5): mu decides STRUCTURE (width/depth/params/FLOPs/
# energy), not early stopping.
#
# Spec:     docs/research/mercyful_machine_channel_benchmark_spec_2026-07-26.md
# Harness:  scripts/research/mercyful_machine_channel_benchmark.py (M1..M8)
#
# Execution path: repo .venv Python (torch CPU + numpy). Pure synthetic data;
# no Sounio-native leg (Python reference implementation; scope note in
# spec section 6).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/mercyful_machine_channel_benchmark.py"
SPEC="${REPO_ROOT}/docs/research/mercyful_machine_channel_benchmark_spec_2026-07-26.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"

fail() {
    echo "MERCYFUL_MACHINE_CHANNEL_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# M1..M8: machine-channel structural contract.
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "machine-channel benchmark failed to run"
for clause in M1 M2 M3 M4 M5 M6 M7 M8; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}: PASS" || fail "clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'MERCYFUL_MACHINE_CHANNEL_VERDICT C_GREEN' || fail "verdict not C_GREEN"
echo "M1_M8_MACHINE_CHANNEL_CONTRACT PASS"

# C9: no-clinical-claim / synthetic-data warnings present in harness and spec.
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
echo "C9_NO_CLINICAL_CLAIM PASS"

echo "MERCYFUL_MACHINE_CHANNEL_GATE_OK"
