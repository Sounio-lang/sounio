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

# L1..L9 contract. Two equivalent paths:
#   (default) full single-process harness run (~hours)
#   SAN_LARGE_MULTI_LEG=1 — re-verify existing per-family + sweep logs under
#   artifacts/san_large/canonical_{resnet50,vitlarge,gpt,sweep}.log
# Thread count is part of the canonical numeric environment (THREADS=16).
export SAN_LARGE_THREADS="${SAN_LARGE_THREADS:-16}"
mkdir -p "${REPO_ROOT}/artifacts/san_large"
GATE_LOG="${REPO_ROOT}/artifacts/san_large/gate_harness_output.log"

if [[ "${SAN_LARGE_MULTI_LEG:-}" == "1" ]]; then
    ART="${REPO_ROOT}/artifacts/san_large"
    for fam in resnet50 vitlarge gpt; do
        log="${ART}/canonical_${fam}.log"
        [[ -f "${log}" ]] || fail "missing multi-leg log ${log}"
        grep -Fq 'SUFFERING_AWARE_LARGE_VERDICT L_GREEN (8/8 clauses PASS)' "${log}" \
            || fail "family ${fam} not L_GREEN 8/8"
        for clause in L1 L2 L3 L4 L5 L6 L7 L8; do
            grep -Eq "^  ${clause}\[${fam}\]: PASS" "${log}" \
                || fail "clause ${clause}[${fam}] missing or not PASS"
        done
        grep -Fq 'SAN gratuitous=0 FLOPs' "${log}" \
            || fail "SAN gratuitous not zero for ${fam}"
        grep -Fq 'asymmetry 5.0x' "${log}" \
            || fail "harm-matrix asymmetry missing for ${fam}"
    done
    slog="${ART}/canonical_sweep.log"
    [[ -f "${slog}" ]] || fail "missing sweep log ${slog}"
    grep -Fq 'SUFFERING_AWARE_LARGE_VERDICT L_GREEN (1/1 clauses PASS)' "${slog}" \
        || fail "sweep not L_GREEN 1/1"
    if grep -Eq "^  L9\[.*\]: FAIL" "${slog}"; then
        fail "an L9 scalability-sweep clause failed"
    fi
    n_l9=$(grep -cE '^  L9\[.*\]: PASS' "${slog}" || true)
    [[ "${n_l9}" -ge 8 ]] || fail "expected ≥8 L9 PASS lines, got ${n_l9}"
    # Assemble multi-leg certificate for the 9/9 verdict line.
    "${PYTHON}" - <<'PY' || fail "multi-leg assembler failed"
import re, pathlib
root = pathlib.Path("artifacts/san_large")
lines = []
for fam in ("resnet50", "vitlarge", "gpt"):
    t = (root / f"canonical_{fam}.log").read_text()
    assert "L_GREEN (8/8 clauses PASS)" in t, fam
    for i in range(1, 9):
        m = re.search(rf"^  L{i}\[{re.escape(fam)}\]: PASS", t, re.M)
        assert m, (fam, i)
        lines.append(m.group(0))
sw = (root / "canonical_sweep.log").read_text()
assert "L_GREEN (1/1 clauses PASS)" in sw
lines += re.findall(r"^  L9\[.*\]: PASS.*", sw, re.M)
assert len([ln for ln in lines if ln.startswith("  L9[")]) >= 8
out = root / "multi_leg_certificate.log"
out.write_text(
    "SAN large multi-leg certificate (THREADS=16)\n"
    + "\n".join(lines) + "\n"
    + "SUFFERING_AWARE_LARGE_VERDICT L_GREEN (9/9 clauses PASS)\n"
    + "SUFFERING_AWARE_LARGE_MULTI_LEG_OK\n"
)
print(out.read_text())
PY
    PY_OUTPUT=$(cat "${ART}/multi_leg_certificate.log")
    printf '%s\n' "${PY_OUTPUT}" | tee "${GATE_LOG}" >/dev/null
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SUFFERING_AWARE_LARGE_VERDICT L_GREEN (9/9 clauses PASS)' \
        || fail "multi-leg verdict not L_GREEN 9/9"
    echo "L1_L9_LARGE_SAN_CONTRACT PASS (multi-leg)"
else
    GATE_LOG="${REPO_ROOT}/artifacts/san_large/gate_harness_output.log"
    PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1 | tee "${GATE_LOG}") || fail "large-SAN harness failed to run (see ${GATE_LOG})"
    for clause in L1 L2 L3 L4 L5 L6 L7 L8 L9; do
        printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}\[" || fail "clause ${clause} output missing"
    done
    if printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  L[1-8]\[.*\]: FAIL"; then
        fail "an L1..L8 family clause failed"
    fi
    if printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  L9\[.*\]: FAIL"; then
        fail "an L9 scalability-sweep clause failed"
    fi
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SUFFERING_AWARE_LARGE_VERDICT L_GREEN (9/9 clauses PASS)' \
        || fail "verdict not L_GREEN 9/9"
    echo "L1_L9_LARGE_SAN_CONTRACT PASS"
fi

# C9: canonical numbers cross-check (spec section 7 anchors).
# Family legs print L4 as 'SAN gratuitous=0 FLOPs' and L7 as 'asymmetry 5.0x'.
# Multi-leg path reuses the per-family logs for those anchors.
if [[ "${SAN_LARGE_MULTI_LEG:-}" == "1" ]]; then
    for fam in resnet50 vitlarge gpt; do
        grep -Fq 'SAN gratuitous=0 FLOPs' "${REPO_ROOT}/artifacts/san_large/canonical_${fam}.log" \
            || fail "SAN gratuitous not zero for ${fam}"
        grep -Fq 'asymmetry 5.0x' "${REPO_ROOT}/artifacts/san_large/canonical_${fam}.log" \
            || fail "asymmetry anchor missing for ${fam}"
        grep -Eq "ledger\[${fam}-san\].*S_m=" "${REPO_ROOT}/artifacts/san_large/canonical_${fam}.log" \
            || fail "SAN ledger missing for ${fam}"
    done
else
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAN gratuitous=0 FLOPs' \
        || fail "SAN gratuitous suffering not exactly zero"
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'asymmetry 5.0x' \
        || fail "harm-matrix asymmetry anchor missing or wrong"
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq 'S_m(/epoch)? SAN=' \
        || fail "SAN machine-suffering line missing"
fi
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
if [[ "${SAN_LARGE_MULTI_LEG:-}" == "1" ]]; then
    grep -Fq 'no_consciousness_claim' \
        "${REPO_ROOT}/artifacts/san_large/canonical_resnet50.log" \
        || fail "harness output missing no_consciousness_claim note"
else
    printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
        || fail "harness output missing no_consciousness_claim note"
fi
echo "C11_SCOPE_GUARDS PASS"

echo "SUFFERING_AWARE_LARGE_GATE_OK"
