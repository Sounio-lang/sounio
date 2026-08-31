#!/usr/bin/env bash
# CI gate for the Mercyful Learning SAN expanded-ethics rung: environmental,
# social, and temporal suffering channels alongside patient + machine.
#
# Spec:    docs/research/suffering_aware_expanded_channels_spec_2026-07-31.md
# Harness: scripts/research/suffering_aware_expanded_channels.py (X1..X8)
#
# Execution path: repo .venv Python (torch CPU + numpy). Pure synthetic data;
# no Sounio-native leg (Python reference implementation; spec section 8).
# Self-contained: intentionally NOT wired into .github/workflows/ci.yml yet
# (shared control file under active edit by other lanes on this branch);
# wiring is left to the integrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_expanded_channels.py"
SPEC="${REPO_ROOT}/docs/research/suffering_aware_expanded_channels_spec_2026-07-31.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"

fail() {
    echo "SUFFERING_AWARE_EXPANDED_CHANNELS_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# X1..X8: expanded-channels contract.
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "expanded-channels harness failed to run"
for clause in X1 X2 X3 X4 X5 X6 X7 X8; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}: PASS" || fail "clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SUFFERING_AWARE_EXPANDED_CHANNELS_VERDICT X_GREEN (8/8 clauses PASS)' \
    || fail "verdict not X_GREEN 8/8"
echo "X1_X8_EXPANDED_CHANNELS_CONTRACT PASS"

# C9: canonical numbers cross-check (spec section 6 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_machine=0.645GF' \
    || fail "SAN total machine suffering missing or wrong (expected 0.645GF)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAN gratuitous all-zero=True' \
    || fail "SAN gratuitous suffering not exactly zero on all channels"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'shortcut train_acc=0.866' \
    || fail "X8 shortcut train accuracy missing or wrong"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq '70-point simplex grid feasible-only=True' \
    || fail "X3 simplex-grid selection not feasible-only"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (five channels, expanded gate, theorems, honesty notes).
grep -Fq 'environmental' "${SPEC}" || fail "spec missing environmental channel"
grep -Fq 'social' "${SPEC}" || fail "spec missing social channel"
grep -Fq 'temporal' "${SPEC}" || fail "spec missing temporal channel"
grep -Fq 'Rawls' "${SPEC}" || fail "spec missing Rawlsian justice term"
grep -Fq 'equity' "${SPEC}" || fail "spec missing equity gap"
grep -Fq 'expanded feasibility' "${SPEC}" || fail "spec missing expanded feasibility gate"
grep -Fq 'channel metering conservation' "${SPEC}" || fail "spec missing T1X"
grep -Fq 'expanded anti-Goodhart soundness' "${SPEC}" || fail "spec missing T2X"
grep -Fq 'derived vs. measured channels' "${SPEC}" || fail "spec missing derived/measured honesty note"
grep -Fq 'equity-gap tradeoff' "${SPEC}" || fail "spec missing honest negative finding"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim,
# proxies declared as proxies.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'operational proxies with declared constants' "${SPEC}" \
    || fail "missing proxy disclaimer in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'operational_proxies' \
    || fail "harness output missing operational_proxies note"
echo "C11_SCOPE_GUARDS PASS"

echo "SUFFERING_AWARE_EXPANDED_CHANNELS_GATE_OK"
