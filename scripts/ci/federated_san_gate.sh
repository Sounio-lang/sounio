#!/usr/bin/env bash
# CI gate for the Mercyful Learning FEDERATED Suffering-Aware neural Network
# (FED-SAN): the suffering-aware architecture distributed across federated
# nodes, with a two-level gated aggregator and a compute+communication
# suffering ledger.
#
# Spec:    docs/research/federated_san_spec_2026-07-30.md
# Harness: scripts/research/federated_san.py (F1..F9)
#
# Execution path: repo .venv Python (torch CPU + numpy) on real data:
#   - clinical leg: WDBC (569 real de-identified patients, UCI #17, vendored
#     at datasets/san_real_patient/)
#   - vision leg: CIFAR-10 (real images, vendored at
#     datasets/cifar-10-batches-py)
# No Sounio-native leg (Python reference implementation; scope note in spec
# section 10). Self-contained: intentionally NOT wired into
# .github/workflows/ci.yml yet (shared control file under active edit by
# other lanes on this branch); wiring is left to the integrator.
#
# Runtime: the full contract runs the federated protocol on both legs
# (clinical ~1 min, vision ~5-10 min on CPU; FED_SAN_THREADS controls
# parallelism). A fast mechanics-only check is available as FED_SAN_SMOKE=1
# against synthetic stand-ins, but this gate runs the canonical full
# contract.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/federated_san.py"
SPEC="${REPO_ROOT}/docs/research/federated_san_spec_2026-07-30.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"
CLIN_DIR="${REPO_ROOT}/datasets/san_real_patient"
CIFAR_DIR="${REPO_ROOT}/datasets/cifar-10-batches-py"

fail() {
    echo "FEDERATED_SAN_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# C0b: datasets present (real-data requirement).
[[ -f "${CLIN_DIR}/wdbc.data" ]] || fail "WDBC missing at ${CLIN_DIR}/wdbc.data (real patient cohort required)"
[[ -d "${CIFAR_DIR}" ]] || fail "CIFAR-10 missing at ${CIFAR_DIR}; fetch with: curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xz -C ${REPO_ROOT}/datasets"
echo "C0B_REAL_DATA_PRESENT PASS"

# F1..F9: federated SAN contract (full canonical run, both legs).
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "federated-SAN harness failed to run"
for clause in "F1[clinical]" "F1[vision]" "F2[clinical]" "F2[vision]" F3 \
              "F4[clinical]" "F4[vision]" "F5[clinical]" "F5[vision]" \
              "F6[clinical]" "F6[vision]" "F7[clinical]" "F7[vision]" F8 F9; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq "  ${clause}: " \
        || fail "clause ${clause} output missing"
done
if printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  F[0-9]+\[[a-z]+\]: FAIL|^  F[0-9]: FAIL"; then
    fail "an F1..F9 clause failed"
fi
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'FEDERATED_SAN_VERDICT F_GREEN (15/15 clauses PASS)' \
    || fail "verdict not F_GREEN 15/15"
echo "F1_F9_FEDERATED_CONTRACT PASS"

# C9: canonical anchors (spec sections 8.3 / 8.4).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'FED-SAN gratuitous=0 FLOPs, 0 bytes [clinical]' \
    || fail "clinical gratuitous suffering not exactly zero"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'FED-SAN gratuitous=0 FLOPs, 0 bytes [vision]' \
    || fail "vision gratuitous suffering not exactly zero"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'asymmetry 5.0x' \
    || fail "harm-matrix asymmetry anchor missing or wrong"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'poison in no accepted average=True' \
    || fail "adversarial containment anchor missing (F8)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'MATCH' \
    || fail "WDBC provenance anchor missing (F9)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'federation_overhead[clinical]' \
    || fail "federation-overhead finding line missing"
echo "C9_CANONICAL_ANCHORS PASS"

# C10: spec components (federation setting, gated aggregator, metering,
# theorems, honest limitations).
grep -Fq 'federated' "${SPEC}" || fail "spec missing federation setting"
grep -Fq 'metering conservation under federation' "${SPEC}" || fail "spec missing T1"
grep -Fq 'gated-aggregation soundness' "${SPEC}" || fail "spec missing T2"
grep -Fq 'suffering bounds' "${SPEC}" || fail "spec missing T3"
grep -Fq 'necessary/gratuitous separation, federated' "${SPEC}" || fail "spec missing T4"
grep -Fq 'Dirichlet' "${SPEC}" || fail "spec missing non-IID model"
grep -Fq 'reject-and-freeze deadlock' "${SPEC}" || fail "spec missing the found-failure account (section 5.1)"
grep -Fq 'WDBC' "${SPEC}" || fail "spec missing real patient cohort"
grep -Fq 'CIFAR-10' "${SPEC}" || fail "spec missing real image dataset"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim,
# honest synthetic-harm statement for the vision leg, privacy statement.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'de-identified and public' "${SPEC}" || fail "missing privacy/provenance statement in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
echo "C11_SCOPE_GUARDS PASS"

echo "FEDERATED_SAN_GATE_OK"
