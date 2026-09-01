#!/usr/bin/env bash
# Anti-garbling axes 1 and 3 at the checker — E252 (order) and E251 (norm), with controls.
#
# Theory:  formal/lean4/EpistemicEffectsNSA.lean  (assocCert / reassoc_sound; not_polarBasis4 /
#          sed_shortcut_understates / shortcut_eq_sensitivity_of_polarBasis)
# Doc:     docs/research/ANTIGARBLING_FUSION_THEOREM_2026-09-01.md §10
# Sibling: scripts/ci/ns_antigarbling_gate.sh (axis 2, E230)
#
# Every fixture is checked with the SAME compiler, and each refusal is paired with the
# nearest accepted program, so a refusal cannot be a broken compiler and an acceptance
# cannot be a silent skip:
#   E251  Knowledge<Hyper<Sedenion>> * Knowledge<Hyper<Sedenion>>   refused (NS on and NS off)
#         Knowledge<Hyper<Octonion>> * ...  and sedenion `+`         accepted under NS off (Hurwitz boundary;
#         with NS on, ⊤-parameter operands are E230 by design, so the knob isolates E251)
#   E252  `reassociate: free` on an alternative product              refused
#         `reassociate: fano_selective` on Sedenion                   refused
#         clause omitted on an alternative product                    accepted  (derived, fail-closed)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
[[ -x "$SOUC" ]] || { echo "FAIL: no souc at $SOUC" >&2; exit 1; }
echo "souc=$SOUC"
"$SOUC" --version 2>&1 | grep -E '^provenance' || true

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "PASS: $*"; }

refuses() {  # refuses <code> <label> <file>
  local code="$1" label="$2" src="$3" out rc
  set +e; out=$("$SOUC" check "$src" 2>&1); rc=$?; set -e
  [[ $rc -ne 0 ]] || fail "$label expected refusal, got rc=0"
  echo "$out" | grep -E "$code" >/dev/null || fail "$label refused but not with $code: $out"
  pass "$label refused with $code"
}
accepts() {  # accepts <label> <file>
  local label="$1" src="$2" out rc
  set +e; out=$("$SOUC" check "$src" 2>&1); rc=$?; set -e
  [[ $rc -eq 0 ]] || fail "$label expected acceptance, got rc=$rc: $out"
  echo "$out" | grep -E 'E251|E252' >/dev/null && fail "$label accepted but printed E251/E252: $out"
  pass "$label accepted"
}

# E251 with NS on: the sedenion product is refused (E230 also fires: ⊤ parameters, by design).
refuses E251 "sedenion Knowledge product (third axis, NS on)" tests/compile-fail/e251_knowledge_sedenion_product_shortcut.sio
# E251 with NS OFF (SOUNIO_NS_DISABLE=1): E230 vanishes, E251 must SURVIVE -- it is not an NS effect.
refuses_nsoff() {  # refuses_nsoff <code> <label> <file>
  local code="$1" label="$2" src="$3" out rc
  set +e; out=$(SOUNIO_NS_DISABLE=1 "$SOUC" check "$src" 2>&1); rc=$?; set -e
  [[ $rc -ne 0 ]] || fail "$label expected refusal under SOUNIO_NS_DISABLE=1, got rc=0"
  echo "$out" | grep -E "$code" >/dev/null || fail "$label not refused with $code under SOUNIO_NS_DISABLE=1: $out"
  echo "$out" | grep -E 'E230' >/dev/null && fail "$label still raised E230 under SOUNIO_NS_DISABLE=1 (knob inert)"
  pass "$label survives SOUNIO_NS_DISABLE=1 with $code (causally separable from E230)"
}
accepts_nsoff() {  # accepts_nsoff <label> <file>
  local label="$1" src="$2" out rc
  set +e; out=$(SOUNIO_NS_DISABLE=1 "$SOUC" check "$src" 2>&1); rc=$?; set -e
  [[ $rc -eq 0 ]] || fail "$label expected acceptance under SOUNIO_NS_DISABLE=1, got rc=$rc: $out"
  echo "$out" | grep -E 'E251|E252' >/dev/null && fail "$label accepted but printed E251/E252: $out"
  pass "$label accepted under SOUNIO_NS_DISABLE=1"
}
refuses_nsoff E251 "sedenion Knowledge product (NS off)" tests/compile-fail/e251_knowledge_sedenion_product_shortcut.sio
accepts_nsoff      "octonion Knowledge product + sedenion sum (Hurwitz control, NS off)" tests/fixtures/antigarbling/e251_control_knowledge_octonion_product.sio
refuses E252 "reassociate: free on alternative"        tests/compile-fail/e252_reassociate_free_on_alternative.sio
refuses E252 "fano_selective on Sedenion"              tests/compile-fail/e252_fano_selective_on_sedenion.sio
accepts       "reassociate omitted on alternative (derived)" tests/run-pass/algebra_reassociate_omitted_fail_closed.sio
# Pre-existing octonion algebra fixtures must keep checking (fano_selective on Octonion).
accepts       "algebra_decl_basic (fano_selective on Octonion)"       tests/run-pass/algebra_decl_basic.sio
accepts       "octonion_hessian_fano_annotated (fano_selective)"     tests/run-pass/octonion_hessian_fano_annotated.sio

echo "ANTIGARBLING_THIRD_AXIS_GATE_PASS"
