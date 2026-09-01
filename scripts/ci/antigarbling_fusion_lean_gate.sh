#!/usr/bin/env bash
# Anti-Garbling FUSION theorem Lean gate — two certificates, one calculus.
#
# Spec:      docs/research/ANTIGARBLING_FUSION_THEOREM_2026-09-01.md
# Lean file: formal/lean4/EpistemicEffectsNSA.lean   (self-contained: no imports)
#
# Checks:
#   C1  the module compiles (lake build; falls back to a direct `lean --threads=1`
#       build when lake dies with the pod's "failed to create thread" limit)
#   C2  sorry-free; native_decide-free (the module must be kernel-decided only)
#   C3  axiom footprint of every load-bearing theorem ⊆ {propext, Quot.sound, Classical.choice}
#   C4  the load-bearing theorems exist by name (the companion doc cites them)
#   C5  registered as @[default_target] in lakefile.lean (CI `lean-proofs` coverage)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEAN_DIR="${REPO_ROOT}/formal/lean4"
LEAN_FILE="${LEAN_DIR}/EpistemicEffectsNSA.lean"
LAKEFILE="${LEAN_DIR}/lakefile.lean"
MODULE="EpistemicEffectsNSA"
ALLOWED_AXIOMS="propext Quot.sound Classical.choice"

fail() { echo "FAIL: $*" >&2; exit 1; }
[[ -f "${LEAN_FILE}" ]] || fail "missing ${LEAN_FILE}"

if [[ -d "${HOME}/.elan/bin" ]]; then export PATH="${HOME}/.elan/bin:${PATH}"; fi
command -v lean >/dev/null 2>&1 || { echo "SKIP: lean not found"; exit 0; }

# ---- C1: compile (and capture the #print axioms output for C3) -------------------------
OUT="$(mktemp -t nsa_lean_gate-XXXXXX.log)"
cleanup() { rm -f "${OUT}"; return 0; }
trap cleanup EXIT

direct_build() {
    (cd "${LEAN_DIR}" && lean --threads=1 "${MODULE}.lean") > "${OUT}" 2>&1 \
        || { cat "${OUT}" >&2; fail "direct build of ${MODULE} failed"; }
    echo "C1 PASS: ${MODULE} compiles (direct lean --threads=1)"
}

if command -v lake >/dev/null 2>&1; then
    if (cd "${LEAN_DIR}" && lake build "${MODULE}") > "${OUT}" 2>&1; then
        (cd "${LEAN_DIR}" && lake env lean "${MODULE}.lean") > "${OUT}" 2>&1 \
            || { cat "${OUT}" >&2; fail "lake env lean ${MODULE}.lean failed"; }
        echo "C1 PASS: ${MODULE} compiles (lake build)"
    elif grep -q 'failed to create thread' "${OUT}"; then
        echo "note: lake hit the host thread limit; falling back to direct lean --threads=1"
        direct_build
    else
        cat "${OUT}" >&2; fail "lake build ${MODULE} failed"
    fi
else
    direct_build
fi

grep -q 'error' "${OUT}" && { cat "${OUT}" >&2; fail "compile log contains errors"; }
grep -q 'warning' "${OUT}" && { cat "${OUT}" >&2; fail "compile log contains warnings"; }

# ---- C2: sorry-free, native_decide-free -------------------------------------------------
if grep -nE '\bsorry\b' "${LEAN_FILE}" | grep -v 'sorry-free' | grep -q .; then
    grep -nE '\bsorry\b' "${LEAN_FILE}" | grep -v 'sorry-free' >&2
    fail "sorry found in ${MODULE}"
fi
# Only PROOF uses count: a line whose first non-blank token is `native_decide` or `by native_decide`.
if grep -nE '(:=|by)\s*native_decide\b|^\s*native_decide\b' "${LEAN_FILE}" | grep -q .; then
    grep -nE 'native_decide' "${LEAN_FILE}" >&2
    fail "native_decide used in ${MODULE} (kernel-decided only)"
fi
echo "C2 PASS: sorry-free, native_decide-free"

# ---- C3: axiom footprint ----------------------------------------------------------------
AXLINES="$(grep 'depends on axioms' "${OUT}" || true)"
[[ -n "${AXLINES}" ]] || fail "no '#print axioms' output captured (expected ≥ 24 lines)"
NAX="$(printf '%s\n' "${AXLINES}" | wc -l | tr -d ' ')"
[[ "${NAX}" -ge 24 ]] || fail "expected ≥ 24 axiom reports, got ${NAX}"
while IFS= read -r line; do
    axs="$(printf '%s' "${line}" | sed -E 's/.*\[(.*)\].*/\1/' | tr ',' ' ')"
    for ax in ${axs}; do
        ok=0
        for allowed in ${ALLOWED_AXIOMS}; do [[ "${ax}" == "${allowed}" ]] && ok=1; done
        [[ "${ok}" -eq 1 ]] || fail "disallowed axiom '${ax}' in: ${line}"
    done
done <<< "${AXLINES}"
grep -q 'sorryAx' "${OUT}" && fail "sorryAx in axiom footprint"
echo "C3 PASS: axiom footprint ⊆ {${ALLOWED_AXIOMS}} across ${NAX} theorems"

# ---- C4: load-bearing theorem names -----------------------------------------------------
for thm in cdMul_add_left cdMul_add_right cdMul_smul_left cdMul_smul_right normSq_add \
           assoc_add1 assoc_add2 assoc_add3 assoc_slot1 assoc_slot2 assoc_slot3 \
           assoc_zero_of_qCoversL fano_lines_assoc quaternion_assoc non_fano_124 \
           assoc_zero_of_cert innerA_zero_of_ns trueVar_append innerA_disjoint \
           nsDisjoint_reassoc_invariant qCovers_cdMul \
           typed_agfree preservation exact_preservation soundness_star \
           reassoc_payload_gap reassoc_sensitivity_gap reassoc_forms_eq reassoc_sound \
           w1_typable w1_cert_refused w1_reassoc_changes_value w1_sensitivity_changes \
           "w1'_cert" "w1'_reassoc_sound" assocCert_level0 w2_untypable w2_understates \
           sed_shortcut_understates sed_x_typable oct_shortcut_exact sed_shortcut_overstates \
           shortcut_eq_sensitivity_level0 \
           lin_zero_of_basis polarBasis3 not_polarBasis4 polar_zero_of_polarBasis basis_bil_zero \
           bil_zero_of_polarBasis norm_mult_of_polarBasis octonion_norm_multiplicative \
           quaternion_norm_multiplicative sedenion_norm_not_multiplicative inner_mulR_eq inner_mulL_eq \
           trueVar_scaleR_eq trueVar_scaleL_eq shortcut_eq_sensitivity_of_polarBasis octonion_shortcut_exact; do
    grep -qE "^theorem ${thm}( |$|\()" "${LEAN_FILE}" || fail "theorem ${thm} not found"
done
echo "C4 PASS: all load-bearing theorems present"

# ---- C5: lakefile registration ---------------------------------------------------------
grep -Fq "lean_lib «${MODULE}»" "${LAKEFILE}" || fail "${MODULE} not in lakefile.lean"
grep -B1 -F "lean_lib «${MODULE}»" "${LAKEFILE}" | grep -Fq '@[default_target]' \
    || fail "${MODULE} not a @[default_target]"
echo "C5 PASS: registered as @[default_target]"

echo "ANTIGARBLING_FUSION_LEAN_GATE_PASS"
