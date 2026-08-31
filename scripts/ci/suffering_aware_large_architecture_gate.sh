#!/usr/bin/env bash
# CI gate for the Mercyful Learning Suffering-Aware neural Network (SAN) at
# larger scale: SAN-ResNet-50 (bottleneck), SAN-ViT-large (contract scale),
# and SAN-GPT (decoder-only transformer LM) on real data.
#
# Spec:    docs/research/suffering_aware_large_architecture_spec_2026-07-31.md
# Harness: scripts/research/suffering_aware_large_architecture.py (L1..L9)
#
# Execution path: repo .venv Python (torch CPU + numpy) on CIFAR-10 (real
# dataset, fetched out-of-band — see below) and the repository's own
# docs/research/*.md as the GPT text corpus (in-repo, no fetch). No
# Sounio-native leg (Python reference implementation; scope note in the
# spec). Self-contained: intentionally NOT wired into
# .github/workflows/ci.yml yet (shared control file under active edit by
# other lanes on this branch); wiring is left to the integrator.
#
# Runtime: the full contract trains ResNet-50, ViT-large (d=384, 12 blocks)
# and a GPT (d=384, 10 blocks) on CPU; expect ~2-4 hours wall-clock
# (SAN_LARGE_THREADS controls parallelism; families can also be run in
# parallel with SAN_LARGE_ONLY=resnet50|vitlarge|gpt plus a final
# SAN_LARGE_ONLY=sweep pass). A fast mechanics-only check is available as
# SAN_LARGE_SMOKE=1, but this gate runs the canonical full contract.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_large_architecture.py"
SPEC="${REPO_ROOT}/docs/research/suffering_aware_large_architecture_spec_2026-07-31.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"
DATA_DIR="${REPO_ROOT}/datasets/cifar-10-batches-py"

fail() {
    echo "SUFFERING_AWARE_LARGE_GATE_FAIL: $*" >&2
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

# L1..L9: large suffering-aware architecture contract (full canonical run).
# Thread count is part of the canonical numeric environment: CPU conv/GEMM
# reduction order depends on it, and several calibrated margins (declared
# per-family tau/delta) are tight at the level of that numeric noise. The
# canonical legs ran at SAN_LARGE_THREADS=16; the gate reproduces them
# there. Harness output is saved for diagnosability.
export SAN_LARGE_THREADS="${SAN_LARGE_THREADS:-16}"
GATE_LOG="${REPO_ROOT}/artifacts/san_large/gate_harness_output.log"
mkdir -p "${REPO_ROOT}/artifacts/san_large"
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1 | tee "${GATE_LOG}") || fail "large-SAN harness failed to run (see ${GATE_LOG})"
for clause in L1 L2 L3 L4 L5 L6 L7 L8 L9; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}\[" || fail "clause ${clause} output missing"
done
# every per-family clause line must be PASS
if printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  L[1-8]\[.*\]: FAIL"; then
    fail "an L1..L8 family clause failed"
fi
if printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  L9\[.*\]: FAIL"; then
    fail "an L9 scalability-sweep clause failed"
fi
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SUFFERING_AWARE_LARGE_VERDICT L_GREEN (9/9 clauses PASS)' \
    || fail "verdict not L_GREEN 9/9"
echo "L1_L9_LARGE_SAN_CONTRACT PASS"

# C9: canonical numbers cross-check (spec section 7 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_m SAN=' \
    || fail "SAN total machine suffering line missing"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAN gratuitous=0 FLOPs' \
    || fail "SAN gratuitous suffering not exactly zero"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'asymmetry 5.0x' \
    || fail "harm-matrix asymmetry anchor missing or wrong"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (larger architectures, real data, metering, theorems).
grep -Fq 'real dataset' "${SPEC}" || fail "spec missing real-dataset statement"
grep -Fq 'ResNet-50' "${SPEC}" || fail "spec missing ResNet-50"
grep -Fq 'ViT-large' "${SPEC}" || fail "spec missing ViT-large"
grep -Fq 'GPT' "${SPEC}" || fail "spec missing GPT leg"
grep -Fq 'CIFAR-10' "${SPEC}" || fail "spec missing CIFAR-10"
grep -Fq 'metering conservation' "${SPEC}" || fail "spec missing T1 (metering conservation)"
grep -Fq 'anti-Goodhart soundness' "${SPEC}" || fail "spec missing T2 (anti-Goodhart soundness)"
grep -Fq 'machine-suffering bound' "${SPEC}" || fail "spec missing T3 (suffering bound)"
grep -Fq 'scalability' "${SPEC}" || fail "spec missing scalability argument"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim,
# honest statement that the harm structures are synthetic over real labels.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-harm statement in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
echo "C11_SCOPE_GUARDS PASS"

echo "SUFFERING_AWARE_LARGE_GATE_OK"
