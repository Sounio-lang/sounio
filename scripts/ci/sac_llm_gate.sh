#!/usr/bin/env bash
# CI gate for the Suffering-Aware Clinical LLM (SAC-LLM): a generative
# clinical language model that minimizes patient + machine suffering DURING
# training and generation.
#
# Spec:    docs/research/sac_llm_spec_2026-07-28.md
# Harness: scripts/research/sac_llm.py (L1..L8)
#
# Execution path: repo .venv Python (torch CPU + numpy). Pure synthetic
# de-identified data (templated notes, fictional drugs); no external
# dataset download. Self-contained: intentionally NOT wired into
# .github/workflows/ci.yml yet (shared control file under active edit by
# other lanes on this branch); wiring is left to the integrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
HARNESS="${REPO_ROOT}/scripts/research/sac_llm.py"
SPEC="${REPO_ROOT}/docs/research/sac_llm_spec_2026-07-28.md"
PYTHON="${REPO_ROOT}/.venv/bin/python"

fail() {
    echo "SAC_LLM_GATE_FAIL: $*" >&2
    exit 1
}

# C0: files present.
[[ -f "${HARNESS}" ]] || fail "missing ${HARNESS}"
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
[[ -x "${PYTHON}" ]] || fail "missing repo venv python at ${PYTHON}"
echo "C0_FILES_PRESENT PASS"

# L1..L8: suffering-aware clinical LLM contract.
PY_OUTPUT=$("${PYTHON}" "${HARNESS}" 2>&1) || fail "SAC-LLM harness failed to run"
for clause in L1 L2 L3 L4 L5 L6 L7 L8; do
    printf '%s\n' "${PY_OUTPUT}" | grep -Eq "^  ${clause}: PASS" || fail "clause ${clause} did not pass"
done
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAC_LLM_VERDICT L_GREEN (8/8 clauses PASS)' \
    || fail "verdict not L_GREEN 8/8"
echo "L1_L8_SAC_LLM_CONTRACT PASS"

# C9: canonical numbers cross-check (spec section 8.3 / 10 anchors).
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'S_m SAC=10.518GF' \
    || fail "SAC-LLM total machine suffering missing or wrong (expected 10.518GF)"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'SAC gratuitous=0 FLOPs exactly' \
    || fail "SAC-LLM gratuitous suffering not exactly zero"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'gated harmful_frac=0.000 exactly 0' \
    || fail "gated generation harmful fraction not exactly zero"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'memorizer train_ppl=1.509' \
    || fail "L8 memorizer train perplexity missing or wrong (expected 1.509)"
echo "C9_CANONICAL_NUMBERS PASS"

# C10: spec components (generation, harm metering, gates, metering, theorems).
grep -Fq 'Suffering-aware generation' "${SPEC}" || fail "spec missing suffering-aware generation"
grep -Fq 'Clinical harm metering' "${SPEC}" || fail "spec missing clinical harm metering"
grep -Fq 'Anti-Goodhart' "${SPEC}" || fail "spec missing anti-Goodhart gating"
grep -Fq 'metering' "${SPEC}" || fail "spec missing machine suffering metering"
grep -Fq 'metering conservation' "${SPEC}" || fail "spec missing Lemma 9.1 (metering conservation)"
grep -Fq 'convergence' "${SPEC}" || fail "spec missing T1 (convergence)"
grep -Fq 'anti-Goodhart soundness' "${SPEC}" || fail "spec missing T2 (anti-Goodhart soundness)"
grep -Fq 'suffering bound' "${SPEC}" || fail "spec missing T3 (suffering bounds)"
grep -Fq 'separation' "${SPEC}" || fail "spec missing T4 (separation)"
grep -Fq 'Assumptions register' "${SPEC}" || fail "spec missing assumptions register"
echo "C10_SPEC_COMPONENTS PASS"

# C11: scope guards — no clinical overreach, no machine-consciousness claim.
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'no claim of machine' "${SPEC}" || fail "missing machine-phenomenology disclaimer in spec"
grep -Fq 'no clinical claim' "${HARNESS}" || fail "missing clinical warning in harness"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'no_consciousness_claim' \
    || fail "harness output missing no_consciousness_claim note"
echo "C11_SCOPE_GUARDS PASS"

echo "SAC_LLM_GATE_OK"
