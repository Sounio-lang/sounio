#!/usr/bin/env bash
# CI gate for the Mercyful Learning Suffering-Aware neural Network (SAN) on
# REAL PATIENT DATA: the suffering field grounded in real clinical outcomes.
#
# Spec:    docs/research/san_real_patient_data_spec_2026-07-28.md
# Harness: scripts/research/san_real_patient_data.py (R1..R10)
# Data:    datasets/san_real_patient/ (vendored real de-identified public
#          cohorts: WDBC 569, Haberman 306, Cleveland 297; UCI ML Repository,
#          CC-BY 4.0, no credentialing — fetch commands in spec section 9)
#
# Execution path: repo .venv Python (torch CPU + numpy). No Sounio-native leg
# (Python reference implementation; scope note in spec section 11).
# Self-contained: intentionally NOT wired into .github/workflows/ci.yml yet
# (shared control file under active edit by other lanes on this branch);
# wiring is left to the integrator.
#
# Runtime: ~1-2 minutes (three small real cohorts + a live re-run of the
# synthetic A-line instance for the R10 consistency check). A fast
# mechanics-only check is available as SAN_REAL_SMOKE=1 against a synthetic
# stand-in, but this gate runs the canonical real-data contract.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/san_real_patient_data.py"
SPEC="${REPO_ROOT}/docs/research/san_real_patient_data_spec_2026-07-28.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"
DATA_DIR="${REPO_ROOT}/datasets/san_real_patient"

fail() {
    echo "SAN_REAL_PATIENT_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# C0b: real patient cohorts present (real-data requirement; fetch commands
# in spec section 9).
[[ -f "${DATA_DIR}/wdbc.data" ]] || fail "WDBC missing; fetch per spec section 9 (UCI #17)"
[[ -f "${DATA_DIR}/haberman.data" ]] || fail "Haberman missing; fetch per spec section 9 (UCI #43)"
[[ -f "${DATA_DIR}/processed.cleveland.data" ]] || fail "Cleveland missing; fetch per spec section 9 (UCI #45)"
echo "C0B_REAL_COHORTS_PRESENT PASS"

# R1..R10: real-patient-data SAN contract (canonical full run).
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "real-patient SAN harness failed to run"
for clause in R1 R2 R3 R4 R5 R6 R7 R8; do
    for ds in wdbc haberman cleveland; do
        printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}\[${ds}\]" \
            || fail "clause ${clause}[${ds}] output missing"
    done
done
for clause in R9 R10; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}:" \
        || fail "clause ${clause} output missing"
done
if printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  R[0-9]+(\[.*\])?: FAIL"; then
    fail "an R1..R10 clause failed"
fi
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAN_REAL_PATIENT_VERDICT R_GREEN (26/26 clauses PASS)' \
    || fail "verdict not R_GREEN 26/26"
echo "R1_R10_REAL_PATIENT_CONTRACT PASS"

# C9: canonical numbers cross-check (spec sections 6/7 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAN gratuitous=0 FLOPs' \
    || fail "SAN gratuitous suffering not exactly zero"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'harm offdiag max/min=5.0x' \
    || fail "harm-matrix asymmetry anchor missing or wrong"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'agree_with_synthetic=True' \
    || fail "synthetic-real consistency anchor missing"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq '569 real patients' \
    || fail "WDBC real cohort anchor missing"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (real cohorts, suffering field, theorems, consistency).
grep -Fq 'real patient' "${SPEC}" || fail "spec missing real-patient statement"
grep -Fq 'WDBC' "${SPEC}" || fail "spec missing WDBC cohort"
grep -Fq 'Haberman' "${SPEC}" || fail "spec missing Haberman cohort"
grep -Fq 'Cleveland' "${SPEC}" || fail "spec missing Cleveland cohort"
grep -Fq 'metering conservation' "${SPEC}" || fail "spec missing T1 (metering conservation)"
grep -Fq 'anti-Goodhart' "${SPEC}" || fail "spec missing T2 (anti-Goodhart)"
grep -Fq 'synthetic-real consistency' "${SPEC}" || fail "spec missing consistency section"
grep -Fq 'MIMIC-IV' "${SPEC}" || fail "spec missing MIMIC-IV availability analysis"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim,
# privacy statement (de-identified public data only).
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'de-identified' "${SPEC}" || fail "missing privacy statement in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
echo "C11_SCOPE_GUARDS PASS"

echo "SAN_REAL_PATIENT_GATE_OK"
