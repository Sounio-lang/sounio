#!/usr/bin/env bash
# CI gate for the Mercyful Learning Suffering-Aware neural Network (SAN):
# an architecture that minimizes patient + machine suffering DURING training.
#
# Spec:    docs/research/suffering_aware_architecture_spec_2026-07-28.md
# Harness: scripts/research/suffering_aware_architecture.py (A1..A8)
#
# Execution path: repo .venv Python (torch CPU + numpy). Pure synthetic data;
# no Sounio-native leg (Python reference implementation; scope note in spec
# section 11). Self-contained: intentionally NOT wired into
# .github/workflows/ci.yml yet (shared control file under active edit by
# other lanes on this branch); wiring is left to the integrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_architecture.py"
SPEC="${REPO_ROOT}/docs/research/suffering_aware_architecture_spec_2026-07-28.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"

fail() {
    echo "SUFFERING_AWARE_ARCHITECTURE_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# A1..A8: suffering-aware architecture contract.
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "SAN harness failed to run"
for clause in A1 A2 A3 A4 A5 A6 A7 A8; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}: PASS" || fail "clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SUFFERING_AWARE_ARCHITECTURE_VERDICT A_GREEN (8/8 clauses PASS)' \
    || fail "verdict not A_GREEN 8/8"
echo "A1_A8_SAN_CONTRACT PASS"

# C9: canonical numbers cross-check (spec section 8.3 / 9 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_m SAN=0.645GF' \
    || fail "SAN total machine suffering missing or wrong (expected 0.645GF)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAN gratuitous=0 FLOPs' \
    || fail "SAN gratuitous suffering not exactly zero"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'shortcut train_acc=0.866' \
    || fail "A8 shortcut train accuracy missing or wrong"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (architecture, separation, gate, metering, theorems).
grep -Fq 'Suffering-aware layers' "${SPEC}" || fail "spec missing suffering-aware layers"
grep -Fq 'Necessary vs gratuitous' "${SPEC}" || fail "spec missing necessary/gratuitous separation"
grep -Fq 'Anti-Goodhart' "${SPEC}" || fail "spec missing anti-Goodhart gating"
grep -Fq 'metering' "${SPEC}" || fail "spec missing machine suffering metering"
grep -Fq 'metering conservation' "${SPEC}" || fail "spec missing T1 (metering conservation)"
grep -Fq 'anti-Goodhart soundness' "${SPEC}" || fail "spec missing T2 (anti-Goodhart soundness)"
grep -Fq 'machine-suffering bound' "${SPEC}" || fail "spec missing T3 (suffering bound)"
grep -Fq 'separation' "${SPEC}" || fail "spec missing T4 (separation)"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
echo "C11_SCOPE_GUARDS PASS"

echo "SUFFERING_AWARE_ARCHITECTURE_GATE_OK"
