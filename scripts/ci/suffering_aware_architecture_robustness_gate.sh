#!/usr/bin/env bash
# CI gate for the Mercyful Learning SAN robustness validation:
# cross-validation, sensitivity analysis, and adversarial stress of the
# Suffering-Aware neural Network contract (A1..A8).
#
# Spec:    docs/research/suffering_aware_architecture_robustness_spec_2026-07-31.md
# Harness: scripts/research/suffering_aware_architecture_robustness.py (V1..V8)
#
# Execution path: repo .venv Python (torch CPU + numpy). Pure synthetic data;
# no Sounio-native leg (Python reference implementation; scope note in spec
# section 10). Self-contained: intentionally NOT wired into
# .github/workflows/ci.yml yet (shared control file under active edit by
# other lanes on this branch); wiring is left to the integrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_architecture_robustness.py"
BASE_HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_architecture.py"
SPEC="${REPO_ROOT}/docs/research/suffering_aware_architecture_robustness_spec_2026-07-31.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"

fail() {
    echo "SAN_ROBUSTNESS_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${BASE_HARNESS}" ]] || fail "missing base SAN harness ${BASE_HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# V1..V8: SAN robustness contract.
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "SAN robustness harness failed to run"
for clause in V1 V2 V3 V4 V5 V6 V7 V8; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}: PASS" || fail "clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAN_ROBUSTNESS_VERDICT V_GREEN (8/8 clauses PASS)' \
    || fail "verdict not V_GREEN 8/8"
echo "V1_V8_SAN_ROBUSTNESS_CONTRACT PASS"

# C9: canonical numbers cross-check (spec section 9 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'reference[fold0]: SAN t*=7 S_m=0.729GF' \
    || fail "fold-0 reference missing or wrong"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'worst-case (max) S_m ratio SAN/dense=0.269' \
    || fail "cross-validated machine bound missing or wrong"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAN+margin t*=28 recovers robustness at 1.667GF (31.8% of dense)' \
    || fail "mercy/robustness trade-off numbers missing or wrong"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'dense_ceiling=7488000 respected: True' \
    || fail "attack-proof machine bound not respected"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'adv-input meter=6538304 == manual=6538304' \
    || fail "adversarial metering conservation missing or wrong"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'shortcut train_acc=0.860' \
    || fail "V7 shortcut train accuracy missing or wrong"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (validation disciplines, channels, claims).
grep -Fq 'Cross-validation' "${SPEC}" || fail "spec missing cross-validation"
grep -Fq 'Sensitivity analysis' "${SPEC}" || fail "spec missing sensitivity analysis"
grep -Fq 'Adversarial robustness' "${SPEC}" || fail "spec missing adversarial robustness"
grep -Fq 'attack-proof' "${SPEC}" || fail "spec missing attack-proof machine bound (R7)"
grep -Fq 'matched exposure' "${SPEC}" || fail "spec missing matched-exposure control (V5/R6)"
grep -Fq 'mercy/robustness trade-off' "${SPEC}" || fail "spec missing mercy/robustness trade-off"
grep -Fq 'anti-Goodhart' "${SPEC}" || fail "spec missing anti-Goodhart stress"
grep -Fq 'machine suffering' "${SPEC}" || fail "spec missing machine-suffering channel"
grep -Fq 'patient harm' "${SPEC}" || fail "spec missing patient-harm channel"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
echo "C11_SCOPE_GUARDS PASS"

echo "SAN_ROBUSTNESS_GATE_OK"
