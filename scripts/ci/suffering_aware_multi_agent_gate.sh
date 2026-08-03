#!/usr/bin/env bash
# CI gate for the Mercyful Learning Suffering-Aware Multi-Agent (SAMA) system:
# multi-agent training that minimizes collective patient + machine suffering
# under strategic and adversarial agents.
#
# Spec:    docs/research/suffering_aware_multi_agent_spec_2026-07-30.md
# Harness: scripts/research/suffering_aware_multi_agent.py (G1..G8)
#
# Execution path: repo .venv Python (numpy only). Pure synthetic data;
# no Sounio-native leg (Python reference implementation; scope note in spec
# section 11). Self-contained: intentionally NOT wired into
# .github/workflows/ci.yml yet (shared control file under active edit by
# other lanes on this branch); wiring is left to the integrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_multi_agent.py"
SPEC="${REPO_ROOT}/docs/research/suffering_aware_multi_agent_spec_2026-07-30.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"

fail() {
    echo "SUFFERING_AWARE_MULTI_AGENT_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# G1..G8: suffering-aware multi-agent contract.
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "SAMA harness failed to run"
for clause in G1 G2 G3 G4 G5 G6 G7 G8; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}: PASS" || fail "clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SUFFERING_AWARE_MULTI_AGENT_VERDICT G_GREEN (8/8 clauses PASS)' \
    || fail "verdict not G_GREEN 8/8"
echo "G1_G8_SAMA_CONTRACT PASS"

# C9: canonical numbers cross-check (spec sections 7 / 9 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_m=5.897MF' \
    || fail "SAMA total machine suffering missing or wrong (expected 5.897MF)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAMA gratuitous=0 FLOPs' \
    || fail "SAMA gratuitous suffering not exactly zero"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'adv_phi=+0.3252' \
    || fail "adversarial attributed harm missing or wrong (expected +0.3252)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'harm share=-1.1658 cf=-1.1734' \
    || fail "G8 harm-share pair missing or wrong (expected -1.1658 vs -1.1734)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'eff_err=0.00e+00' \
    || fail "Shapley efficiency error missing or wrong (expected 0.00e+00)"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (agents, ledger, gate, attribution, theorems).
grep -Fq 'Suffering-aware agents' "${SPEC}" || fail "spec missing suffering-aware agents"
grep -Fq 'collective suffering ledger' "${SPEC}" || fail "spec missing collective suffering ledger"
grep -Fq 'Anti-Goodhart' "${SPEC}" || fail "spec missing anti-Goodhart gating"
grep -Fq 'Burden attribution' "${SPEC}" || fail "spec missing burden attribution"
grep -Fq 'convergence' "${SPEC}" || fail "spec missing T1 (convergence)"
grep -Fq 'anti-Goodhart soundness' "${SPEC}" || fail "spec missing T2 (anti-Goodhart soundness)"
grep -Fq 'attribution soundness' "${SPEC}" || fail "spec missing T3 (attribution soundness)"
grep -Fq 'strategic robustness' "${SPEC}" || fail "spec missing T4 (strategic robustness)"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
echo "C11_SCOPE_GUARDS PASS"

# C12: environment smoke mode for the 2..5-agent range (contract pinned to 5).
for n in 2 3 4; do
    SMOKE_OUTPUT=$(SAMA_N_AGENTS=$n "${PYTHON}" "${HARNESS}" 2>&1) \
        || fail "smoke mode N=${n} failed to run"
    printf '%s\n' "${SMOKE_OUTPUT}" | grep -Fq 'SUFFERING_AWARE_MULTI_AGENT_VERDICT SMOKE_OK' \
        || fail "smoke mode N=${n} did not complete"
done
echo "C12_ENV_SMOKE_2_TO_4 PASS"

echo "SUFFERING_AWARE_MULTI_AGENT_GATE_OK"
