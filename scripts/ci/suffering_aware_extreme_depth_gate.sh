#!/usr/bin/env bash
# CI gate for the Mercyful Learning Suffering-Aware neural Network (SAN) at
# EXTREME depth: a 100-layer residual network (SAN-100) and a 24-block
# transformer (SAN-ViT-24) on real data (CIFAR-10).
#
# Spec:    docs/research/suffering_aware_extreme_depth_spec_2026-07-31.md
# Harness: scripts/research/suffering_aware_extreme_depth.py (X1..X9)
#
# Execution path: repo .venv Python (torch CPU + numpy) on CIFAR-10 (real
# dataset, fetched out-of-band — see below). No Sounio-native leg (Python
# reference implementation; scope note in spec section 9). Self-contained:
# intentionally NOT wired into .github/workflows/ci.yml yet (shared control
# file under active edit by other lanes on this branch); wiring is left to
# the integrator.
#
# Runtime: the full contract trains a 100-layer ResNet and a 24-block ViT on
# a 4000-sample CIFAR-10 subset on CPU; expect ~60-120 minutes wall-clock
# (SAN_XDEPTH_THREADS controls parallelism). A fast mechanics-only check is
# available as SAN_XDEPTH_SMOKE=1, but this gate runs the canonical full
# contract.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_extreme_depth.py"
SPEC="${REPO_ROOT}/docs/research/suffering_aware_extreme_depth_spec_2026-07-31.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"
DATA_DIR="${REPO_ROOT}/datasets/cifar-10-batches-py"

fail() {
    echo "SUFFERING_AWARE_EXTREME_DEPTH_GATE_FAIL: $*" >&2
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

# X1..X10: extreme-depth suffering-aware architecture contract (canonical run).
# The harness output is tee'd to a log (SAN_XDEPTH_GATE_LOG, default a mktemp
# file) so the canonical numbers are inspectable after the run.
PY_LOG="${SAN_XDEPTH_GATE_LOG:-$(mktemp -t xdepth_gate.XXXXXX.log)}"
"${PYTHON}" "${HARNESS}" 2>&1 | tee "${PY_LOG}" || fail "extreme-depth SAN harness failed to run"
PY_OUTPUT=$(cat "${PY_LOG}")
for clause in X1 X2 X3 X4 X5 X6 X7 X8 X9 X10; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}\[" || fail "clause ${clause} output missing"
done
# every per-family clause line must be PASS
if printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  X([1-8]|10)\[.*\]: FAIL"; then
    fail "an X1..X8/X10 family clause failed"
fi
if printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  X9\[.*\]: FAIL"; then
    fail "an X9 scalability-sweep clause failed"
fi
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SUFFERING_AWARE_EXTREME_DEPTH_VERDICT X_GREEN (10/10 clauses PASS)' \
    || fail "verdict not X_GREEN 10/10"
echo "X1_X10_EXTREME_DEPTH_SAN_CONTRACT PASS"

# C9: canonical numbers cross-check (spec section 7 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_m SAN=' \
    || fail "SAN total machine suffering line missing"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAN gratuitous=0 FLOPs' \
    || fail "SAN gratuitous suffering not exactly zero"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'asymmetry 5.0x' \
    || fail "harm-matrix asymmetry anchor missing or wrong"
# extreme-depth identity anchors: the 100-layer ResNet and the 24-block ViT
# must be the trained instances, and the sweep must cover them.
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAN-100 (100 weighted layers)' \
    || fail "SAN-100 identity line missing"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'X9[resnet-100layers]: PASS' \
    || fail "100-layer sweep configuration missing or failed"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'X9[vit-24blocks]: PASS' \
    || fail "24-block sweep configuration missing or failed"
# the certified scheduler-race inversion: the conv family must LOSE the
# race with the catastrophe signature, the transformer family must WIN it
# with the gradual signature (spec section 5, T7).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'X10[resnet]: PASS (scheduler race LOST' \
    || fail "X10 resnet certified-inversion anchor missing"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'X10[vit]: PASS (scheduler race WON' \
    || fail "X10 vit race-win anchor missing"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (extreme-depth scaling argument, real data, metering,
# theorems).
grep -Fq 'real dataset' "${SPEC}" || fail "spec missing real-dataset statement"
grep -Fq '100' "${SPEC}" || fail "spec missing the 100-layer instance"
grep -Fq 'SAN-100' "${SPEC}" || fail "spec missing SAN-100"
grep -Fq 'Transformer' "${SPEC}" || fail "spec missing transformer leg"
grep -Fq 'CIFAR-10' "${SPEC}" || fail "spec missing CIFAR-10"
grep -Fq 'extreme depth' "${SPEC}" || fail "spec missing extreme-depth framing"
grep -Fq 'metering conservation' "${SPEC}" || fail "spec missing T1'' (metering conservation)"
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

echo "SUFFERING_AWARE_EXTREME_DEPTH_GATE_OK"
