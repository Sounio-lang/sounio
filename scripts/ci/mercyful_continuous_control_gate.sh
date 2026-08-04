#!/usr/bin/env bash
# CI gate for the Mercyful Learning upgraded-algorithm validation:
# ML suffering field + continuous optimal control vs the discrete scheduler.
#
# Spec:     docs/research/mercyful_continuous_control_spec_2026-07-26.md
# Harness:  scripts/research/mercyful_continuous_control_contract.py (V1..V10)
#
# Execution path: pure Python stdlib (no Sounio-native leg — the continuous
# algorithm is a Python reference implementation; scope note in spec §6).
# The gate re-runs the discrete baseline contracts so the comparison anchor
# cannot silently rot.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYCONTRACT="${REPO_ROOT}/scripts/research/mercyful_continuous_control_contract.py"
BASE_RUNTIME="${REPO_ROOT}/scripts/research/mercyful_runtime_contract.py"
BASE_CHEMO="${REPO_ROOT}/scripts/research/mercyful_chemo_contract.py"
SPEC="${REPO_ROOT}/docs/research/mercyful_continuous_control_spec_2026-07-26.md"

fail() {
    echo "MERCYFUL_CONTINUOUS_GATE_FAIL: $*" >&2
    exit 1
}

# V0: files present.
[[ -f "${PYCONTRACT}" ]] || fail "missing ${PYCONTRACT}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
echo "V0_FILES_PRESENT PASS"

# V1..V10: upgraded-algorithm contract.
PY_OUTPUT=$(python3 "${PYCONTRACT}" 2>&1) || fail "continuous-control contract failed"
for clause in \
    V1_CONSISTENCY \
    V2_EXPOSURE_QUADRATURE \
    V3_EXPOSURE_PACING \
    V4_CHEMO_JENSEN \
    V5_CHEMO_FRONTIER_CONTINUUM \
    V6_VANCO_TDM_TIMING \
    V7_VANCO_INFUSION \
    V8_GENERALITY_OFF_NODE_TARGET \
    V9_EFFICIENCY \
    V10_SUBSTRATE_MACHINE_SUFFERING; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "${clause}.*PASS" || fail "clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'MERCYFUL_CONTINUOUS_VERDICT V_GREEN' || fail "verdict not V_GREEN"
echo "V1_V10_CONTINUOUS_CONTRACT PASS"

# Anchor 1: discrete runtime baseline still green (the comparison rests on it).
[[ -f "${BASE_RUNTIME}" ]] || fail "missing ${BASE_RUNTIME}"
BASE_OUTPUT=$(python3 "${BASE_RUNTIME}" 2>&1) || fail "discrete runtime baseline failed"
printf '%s\n' "${BASE_OUTPUT}" | grep -Fq 'MERCYFUL_RUNTIME_VERDICT M_GREEN' || fail "runtime baseline not M_GREEN"
echo "V_ANCHOR_RUNTIME_BASELINE PASS"

# Anchor 2: discrete chemo baseline still green (canonical lifted numbers).
[[ -f "${BASE_CHEMO}" ]] || fail "missing ${BASE_CHEMO}"
CHEMO_OUTPUT=$(python3 "${BASE_CHEMO}" 2>&1) || fail "discrete chemo baseline failed"
printf '%s\n' "${CHEMO_OUTPUT}" | grep -Fq 'MERCYFUL_CHEMO_VERDICT H_GREEN' || fail "chemo baseline not H_GREEN"
echo "V_ANCHOR_CHEMO_BASELINE PASS"

# V11: no-clinical-claim warnings present in harness and spec.
grep -Fq 'no clinical claim' "${PYCONTRACT}" || fail "missing clinical warning in harness"
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
echo "V11_NO_CLINICAL_CLAIM PASS"

echo "MERCYFUL_CONTINUOUS_GATE_OK"
