#!/usr/bin/env bash
# CI gate for the Mercyful Learning Suffering-Aware neural Network (SAN) at
# scale: deep residual networks and transformers on real data (CIFAR-10).
#
# Spec:    docs/research/suffering_aware_deep_architecture_spec_2026-07-28.md
# Harness: scripts/research/suffering_aware_deep_architecture.py (D1..D9)
#
# Execution path: repo .venv Python (torch CPU + numpy) on CIFAR-10 (real
# dataset, fetched out-of-band — see below). No Sounio-native leg (Python
# reference implementation; scope note in spec section 11). Self-contained:
# intentionally NOT wired into .github/workflows/ci.yml yet (shared control
# file under active edit by other lanes on this branch); wiring is left to
# the integrator.
#
# Runtime: the full contract trains ResNet-18 and a ViT-small on a 4000-
# sample CIFAR-10 subset on CPU; expect ~30-60 minutes wall-clock
# (SAN_DEEP_THREADS controls parallelism). A fast mechanics-only check is
# available as SAN_DEEP_SMOKE=1 against a synthetic stand-in dataset, but
# this gate runs the canonical full contract.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_deep_architecture.py"
SPEC="${REPO_ROOT}/docs/research/suffering_aware_deep_architecture_spec_2026-07-28.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"
DATA_DIR="${REPO_ROOT}/datasets/cifar-10-batches-py"

fail() {
    echo "SUFFERING_AWARE_DEEP_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# C0b: dataset present (real-data requirement; fetch command documented in
# the harness docstring).
[[ -d "${DATA_DIR}" ]] || fail "CIFAR-10 missing at ${DATA_DIR}; fetch with: curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xz -C ${REPO_ROOT}/datasets"
echo "C0B_CIFAR10_PRESENT PASS"

# D1..D9: deep suffering-aware architecture contract (full canonical run).
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "deep-SAN harness failed to run"
for clause in D1 D2 D3 D4 D5 D6 D7 D8 D9; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}\[" || fail "clause ${clause} output missing"
done
# every per-family clause line must be PASS
if printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  D[1-8]\[.*\]: FAIL"; then
    fail "a D1..D8 family clause failed"
fi
if printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  D9\[.*\]: FAIL"; then
    fail "a D9 scalability-sweep clause failed"
fi
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SUFFERING_AWARE_DEEP_VERDICT D_GREEN (9/9 clauses PASS)' \
    || fail "verdict not D_GREEN 9/9"
echo "D1_D9_DEEP_SAN_CONTRACT PASS"

# C9: canonical numbers cross-check (spec section 8.3 / 9 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_m SAN=' \
    || fail "SAN total machine suffering line missing"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAN gratuitous=0 FLOPs' \
    || fail "SAN gratuitous suffering not exactly zero"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'asymmetry 5.0x' \
    || fail "harm-matrix asymmetry anchor missing or wrong"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (scaling argument, real data, metering, theorems).
grep -Fq 'real dataset' "${SPEC}" || fail "spec missing real-dataset statement"
grep -Fq 'ResNet-18' "${SPEC}" || fail "spec missing ResNet-18"
grep -Fq 'Transformer' "${SPEC}" || fail "spec missing transformer leg"
grep -Fq 'CIFAR-10' "${SPEC}" || fail "spec missing CIFAR-10"
grep -Fq 'metering conservation' "${SPEC}" || fail "spec missing T1 (metering conservation)"
grep -Fq 'anti-Goodhart soundness' "${SPEC}" || fail "spec missing T2 (anti-Goodhart soundness)"
grep -Fq 'machine-suffering bound' "${SPEC}" || fail "spec missing T3 (suffering bound)"
grep -Fq 'scalability' "${SPEC}" || fail "spec missing scalability argument"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim,
# honest statement that the harm matrix is synthetic over real labels.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-harm statement in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
echo "C11_SCOPE_GUARDS PASS"

echo "SUFFERING_AWARE_DEEP_GATE_OK"
