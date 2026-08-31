#!/usr/bin/env bash
# NS metatheory Lean gate — Paper A §6 mechanization.
#
# Spec:      docs/research/paper_A_ns_metatheory_mechanized_2026-08-30.md
# Lean file: formal/lean4/EpistemicEffectsNS.lean
#
# Checks:
#   C1  the module compiles (lake build; falls back to a direct `lean --threads=1`
#       build ONLY when lake dies with the pod's "failed to create thread" limit)
#   C2  sorry-free
#   C3  axiom footprint of every load-bearing theorem ⊆ {propext, Quot.sound, Classical.choice}
#       — in particular NO sorryAx and NO custom axioms
#   C4  the load-bearing theorems exist by name (the paper cites them)
#   C5  registered as @[default_target] in lakefile.lean (CI `lean-proofs` coverage)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEAN_DIR="${REPO_ROOT}/formal/lean4"
LEAN_FILE="${LEAN_DIR}/EpistemicEffectsNS.lean"
LAKEFILE="${LEAN_DIR}/lakefile.lean"
MODULE="EpistemicEffectsNS"
DEP="EpistemicEffects"
ALLOWED_AXIOMS="propext Quot.sound Classical.choice"

fail() { echo "FAIL: $*" >&2; exit 1; }
[[ -f "${LEAN_FILE}" ]] || fail "missing ${LEAN_FILE}"

if [[ -d "${HOME}/.elan/bin" ]]; then export PATH="${HOME}/.elan/bin:${PATH}"; fi
command -v lean >/dev/null 2>&1 || { echo "SKIP: lean not found"; exit 0; }

# ---- C1: compile (and capture the #print axioms output for C3) -------------------------
OUT="$(mktemp -t ns_lean_gate-XXXXXX.log)"
TMPDIR_OLEAN=""
cleanup() {
    rm -f "${OUT}"
    if [[ -n "${TMPDIR_OLEAN}" ]]; then rm -rf "${TMPDIR_OLEAN}"; fi
    return 0
}
trap cleanup EXIT

direct_build() {
    TMPDIR_OLEAN="$(mktemp -d -t ns_lean_olean-XXXXXX)"
    (cd "${LEAN_DIR}" && lean --threads=1 -o "${TMPDIR_OLEAN}/${DEP}.olean" "${DEP}.lean") \
        || fail "direct build of dependency ${DEP} failed"
    (cd "${LEAN_DIR}" && LEAN_PATH="${TMPDIR_OLEAN}" lean --threads=1 "${MODULE}.lean") > "${OUT}" 2>&1 \
        || { cat "${OUT}" >&2; fail "direct build of ${MODULE} failed"; }
    echo "C1 PASS: ${MODULE} compiles (direct lean --threads=1; lake unavailable on this host)"
}

if command -v lake >/dev/null 2>&1; then
    if (cd "${LEAN_DIR}" && lake build "${MODULE}") > "${OUT}" 2>&1; then
        # lake's build log does not echo #print axioms for cached modules; re-run to capture.
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

# ---- C2: sorry-free ---------------------------------------------------------------------
if grep -nE '\bsorry\b' "${LEAN_FILE}" | grep -v 'sorry-free' | grep -q .; then
    grep -nE '\bsorry\b' "${LEAN_FILE}" | grep -v 'sorry-free' >&2
    fail "sorry found in ${MODULE}"
fi
echo "C2 PASS: sorry-free"

# ---- C3: axiom footprint ----------------------------------------------------------------
AXLINES="$(grep 'depends on axioms' "${OUT}" || true)"
[[ -n "${AXLINES}" ]] || fail "no '#print axioms' output captured (expected ≥ 13 lines)"
NAX="$(printf '%s\n' "${AXLINES}" | wc -l | tr -d ' ')"
[[ "${NAX}" -ge 13 ]] || fail "expected ≥ 13 axiom reports, got ${NAX}"
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
for thm in trueVar_append trueVar_mul inner_disjoint covers_union covers_scale \
           support_over_approx covers_coeff inner_zero_of_ns progress preservation \
           exact_preservation typed_agfree soundness_star \
           x_plus_x_understates x_plus_x_untypable x_plus_top_untypable x_plus_y_exact \
           measure_plus_measure_untypable opaque_y_typable let_x_plus_x_untypable; do
    grep -qE "^theorem ${thm}\b" "${LEAN_FILE}" || fail "theorem ${thm} not found"
done
echo "C4 PASS: all load-bearing theorems present"

# ---- C5: lakefile registration ---------------------------------------------------------
grep -Fq "lean_lib «${MODULE}»" "${LAKEFILE}" || fail "${MODULE} not in lakefile.lean"
grep -B1 -F "lean_lib «${MODULE}»" "${LAKEFILE}" | grep -Fq '@[default_target]' \
    || fail "${MODULE} not a @[default_target]"
echo "C5 PASS: registered as @[default_target]"

echo "NS_METATHEORY_LEAN_GATE_PASS"
