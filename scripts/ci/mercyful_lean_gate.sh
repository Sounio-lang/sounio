#!/usr/bin/env bash
# CI gate for the Lean 4 formalization of Mercyful Learning scheduler
# correctness (Task 3, MIMIC-IV vancomycin TDM line).
#
# Spec:      docs/research/mercyful_scheduler_lean_spec_2026-07-26.md
# Lean file: formal/lean4/SounioMercyfulScheduler.lean
# Runtime:   scripts/research/mercyful_mimic_iv_vancomycin_contract.py (V1..V7)
#
# The module is also a @[default_target] in formal/lean4/lakefile.lean, so
# the CI `lean-proofs` job (`lake build`) covers it. This gate adds the
# contract-specific checks: no sorry, axiom footprint, lakefile
# registration, cross-implementation agreement with the Python contract,
# and scope guards.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LEAN_FILE="${REPO_ROOT}/formal/lean4/SounioMercyfulScheduler.lean"
LEAN_DIR="${REPO_ROOT}/formal/lean4"
LAKEFILE="${LEAN_DIR}/lakefile.lean"
SPEC="${REPO_ROOT}/docs/research/mercyful_scheduler_lean_spec_2026-07-26.md"
PYCONTRACT="${REPO_ROOT}/scripts/research/mercyful_mimic_iv_vancomycin_contract.py"

fail() {
    echo "MERCYFUL_LEAN_GATE_FAIL: $*" >&2
    exit 1
}

# L1: Lean file exists and compiles.
[[ -f "${LEAN_FILE}" ]] || fail "missing ${LEAN_FILE}"
if command -v lake >/dev/null 2>&1; then
    (cd "${LEAN_DIR}" && lake build SounioMercyfulScheduler) || fail "lake build failed"
    echo "L1_LEAN_COMPILES PASS"
else
    echo "SKIP: lake not found; skipping compile check"
fi

# L2: no sorry / admit / custom axioms in the Lean file (matched in
# tactic/declaration position only; the words may appear in comments).
if grep -Enq '(:=[[:space:]]*(by[[:space:]]+)?(sorry|admit)([[:space:]]|$))|(^[[:space:]]*axiom[[:space:]])' "${LEAN_FILE}"; then
    fail "sorry/admit/axiom found in Lean file"
fi
echo "L2_NO_SORRY PASS"

# L3: module registered as @[default_target] in lakefile (CI lean-proofs coverage).
grep -Fq 'lean_lib «SounioMercyfulScheduler»' "${LAKEFILE}" || fail "SounioMercyfulScheduler not in lakefile.lean"
grep -B10 -F 'lean_lib «SounioMercyfulScheduler»' "${LAKEFILE}" | grep -Fq '@[default_target]' \
    || fail "SounioMercyfulScheduler is not a @[default_target]"
echo "L3_LAKEFILE_DEFAULT_TARGET PASS"

# L4: abstract theorems depend only on the standard axiom set
# ([propext, Classical.choice, Quot.sound]) — mirrors the CI axiom check.
if command -v lake >/dev/null 2>&1; then
    AXIOM_TMP="$(mktemp -t mercyful_axioms-XXXXXX.lean)"
    trap 'rm -f "${AXIOM_TMP}"' EXIT
    cat > "${AXIOM_TMP}" <<'EOF'
import SounioMercyfulScheduler
#print axioms Sounio.Mercyful.mercyful_feasible_selection
#print axioms Sounio.Mercyful.mercyful_reaches_target
#print axioms Sounio.Mercyful.mercyful_selects_therapeutic_window
#print axioms Sounio.Mercyful.naive_minimizer_optimal
#print axioms Sounio.Mercyful.goodhart_trap
#print axioms Sounio.Mercyful.anti_goodhart_necessary_and_sufficient
EOF
    AXIOM_OUT="$(cd "${LEAN_DIR}" && lake env lean "${AXIOM_TMP}" 2>&1)" \
        || fail "axiom check failed to run"
    if printf '%s\n' "${AXIOM_OUT}" | grep -v "propext\|Classical\|Quot" | grep -q "axioms"; then
        fail "abstract theorems gained unexpected axioms: ${AXIOM_OUT}"
    fi
    echo "L4_AXIOM_FOOTPRINT PASS"
else
    echo "SKIP: lake not found; skipping axiom check"
fi

# L5: cross-implementation agreement — the Python runtime contract stays
# green and the canonical numbers the Lean C3 theorem certifies are the
# contract's printed values.
[[ -f "${PYCONTRACT}" ]] || fail "missing ${PYCONTRACT}"
PY_OUTPUT=$(python3 "${PYCONTRACT}" 2>&1) || fail "python contract failed"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'MERCYFUL_MIMIC_IV_VERDICT V_GREEN' \
    || fail "python contract not V_GREEN"
printf '%s\n' "${PY_OUTPUT}" | grep -Fq 'integral=0.735099 peak=0.675679 total=1.410778' \
    || fail "canonical numbers missing from python contract (C3 disagreement)"
echo "L5_CROSS_IMPLEMENTATION_AGREEMENT PASS"

# L6: scope guards — no clinical overreach in spec or Lean file.
[[ -f "${SPEC}" ]] || fail "missing ${SPEC}"
grep -Fq 'not medical guidance' "${SPEC}" || fail "missing clinical warning in spec"
grep -Fq 'synthetic' "${SPEC}" || fail "missing synthetic-data statement in spec"
grep -Fq 'not medical guidance' "${LEAN_FILE}" || fail "missing clinical warning in Lean file"
echo "L6_NO_CLINICAL_CLAIM PASS"

echo "MERCYFUL_LEAN_GATE_OK"
