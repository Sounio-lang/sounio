#!/usr/bin/env bash
# CI gate for SAMA at scale: 10, 100, and 1000 heterogeneous,
# hierarchically organized agents minimizing collective patient + machine
# suffering under strategic and adversarial agents.
#
# Spec:    docs/research/suffering_aware_multi_agent_scale_spec_2026-07-31.md
# Harness: scripts/research/suffering_aware_multi_agent_scale.py (S1..S8)
#
# Execution path: repo .venv Python (numpy only). Pure synthetic data;
# no Sounio-native leg (Python reference implementation; scope note in spec
# section 10). Self-contained: intentionally NOT wired into
# .github/workflows/ci.yml yet (shared control file under active edit by
# other lanes on this branch); wiring is left to the integrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/suffering_aware_multi_agent_scale.py"
SPEC="${REPO_ROOT}/docs/research/suffering_aware_multi_agent_scale_spec_2026-07-31.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"

fail() {
    echo "SAMA_SCALE_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# S1..S8 at every scale: 24 clauses.
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "SAMA scale harness failed to run"
for n in 10 100 1000; do
    for clause in S1 S2 S3 S4 S5 S6 S7 S8; do
        printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  N= *${n} ${clause}: PASS" \
            || fail "clause ${clause} at N=${n} did not pass"
    done
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAMA_SCALE_VERDICT S_GREEN (24/24 clauses PASS)' \
    || fail "verdict not S_GREEN 24/24"
echo "S1_S8_SCALE_CONTRACT PASS"

# C9: canonical numbers cross-check (spec section 6 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_m=7.582MF' \
    || fail "N=10 SAMA machine suffering missing or wrong (expected 7.582MF)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_m=46.238MF' \
    || fail "N=100 SAMA machine suffering missing or wrong (expected 46.238MF)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_m=467.111MF' \
    || fail "N=1000 SAMA machine suffering missing or wrong (expected 467.111MF)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_m=18684.432MF' \
    || fail "N=1000 baseline machine suffering missing or wrong (expected 18684.432MF)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'adv_phi_min=+0.2765' \
    || fail "N=10 exact per-agent adversarial phi missing or wrong (expected +0.2765)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'eff_err=0.00e+00' \
    || fail "N=1000 MC-Shapley efficiency error missing or wrong (expected 0.00e+00)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'harm leg VIOLATED (exact): share=-0.6170 cf=-0.5967' \
    || fail "N=10 S8 harm-leg violation report missing or wrong (spec section 7)"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (heterogeneity, hierarchy, attribution, theorems).
grep -Fq 'Heterogeneous' "${SPEC}" || fail "spec missing heterogeneity section"
grep -Fq 'Hierarchical organization' "${SPEC}" || fail "spec missing hierarchy section"
grep -Fq 'Monte-Carlo permutation Shapley' "${SPEC}" || fail "spec missing MC-Shapley section"
grep -Fq 'hierarchical robustness' "${SPEC}" || fail "spec missing T5 (hierarchical robustness)"
grep -Fq 'MC-Shapley soundness' "${SPEC}" || fail "spec missing T6 (MC-Shapley soundness)"
grep -Fq 'VIOLATED' "${SPEC}" || fail "spec missing the measured S8 harm-leg violation"
grep -Fq 'anti-Goodhart' "${SPEC}" || fail "spec missing anti-Goodhart gating"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
echo "C11_SCOPE_GUARDS PASS"

# C12: environment smoke mode (subset-of-scales run completes).
SMOKE_OUTPUT=$(SAMA_SCALE_NS=10 "${PYTHON}" "${HARNESS}" 2>&1) \
    || fail "smoke mode SAMA_SCALE_NS=10 failed to run"
printf '%s\n' "${SMOKE_OUTPUT}" | grep -Fq 'SAMA_SCALE_VERDICT SMOKE_OK' \
    || fail "smoke mode did not complete"
echo "C12_ENV_SMOKE PASS"

echo "SAMA_SCALE_GATE_OK"
