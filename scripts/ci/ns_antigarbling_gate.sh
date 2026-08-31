#!/usr/bin/env bash
# NS anti-garbling (E230) acceptance — N3 gate.
#
# The same-source-built sabotage witness (synthesis §26): a compile-fail witness
# proves nothing by also failing under sabotage. The control that means something
# is a knob (SOUNIO_NS_DISABLE=1) that makes E230 DISAPPEAR while an unrelated
# refusal (E222 R-ORIGIN) STAYS -- both built from the SAME source tree. That
# proves the E230 is caused by the noise-symbol rule and nothing else.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUC_BIN || true
unset SOUNIO_SOUC_BIN || true

fail() { echo "NS_ANTIGARBLING_GATE_FAIL: $*" >&2; exit 1; }
pass() { echo "NS_ANTIGARBLING_GATE_OK: $*"; }

SOUC="${SOUC:-$ROOT/bin/souc}"
[[ -x "$SOUC" ]] || fail "no souc at $SOUC"
echo "souc=$SOUC"

# Refuses with E230 (unset knob).
refuses_e230() {
  local label="$1" src="$2" out rc
  set +e; out=$(SOUNIO_NS_DISABLE= "$SOUC" check "$src" 2>&1); rc=$?; set -e
  [[ $rc -ne 0 ]] || fail "$label expected refusal, got rc=0"
  echo "$out" | grep -E 'E230' >/dev/null || fail "$label refused but not with E230: $out"
  pass "$label refused with E230"
}

# Same file COMPILES with NS disabled (the refusal vanishes -> caused by NS).
vanishes_under_sabotage() {
  local label="$1" src="$2" out rc
  set +e; out=$(SOUNIO_NS_DISABLE=1 "$SOUC" check "$src" 2>&1); rc=$?; set -e
  echo "$out" | grep -E 'E230' >/dev/null && fail "$label E230 survived NS-disable (not caused by NS): $out"
  pass "$label E230 vanishes under SOUNIO_NS_DISABLE"
}

# A disjoint add is NOT refused by the NS rule (the rule is not a blanket ban on
# Knowledge arithmetic). Measured as "no E230", not as rc=0: on main the SAME
# kadd/kmul site also carries E245 (Knowledge x Knowledge arithmetic has no
# lowering on Madaros, #1706), so a disjoint `a + b` is refused by E245 while
# being accepted by E230. The two verdicts are independent and this gate only
# speaks for E230.
accepts() {
  local label="$1" src="$2" out rc
  set +e; out=$("$SOUC" check "$src" 2>&1); rc=$?; set -e
  echo "$out" | grep -E 'E230' >/dev/null && fail "$label wrongly refused a disjoint op: $out"
  pass "$label accepted by the NS rule (no E230; rc=$rc, other diagnostics are not this gate's)"
}

# An UNRELATED refusal must be UNAFFECTED by the NS knob. The wire's original
# control was E222 (R-ORIGIN), which lives on the lane branch and not on main;
# when that fixture is absent, use E245 -- a refusal that sits at the SAME
# binary-op site as E230 and must survive SOUNIO_NS_DISABLE=1 (this is also the
# coexistence proof: E230 off, E245 still on, same source build).
unrelated_refusal_survives_ns_disable() {
  local src="tests/compile-fail/r_origin_measured_on_sum.sio" code="E222" out rc
  if [[ ! -f "$src" ]]; then
    src="tests/compile-fail/knowledge_binary_add_has_no_lowering.sio"; code="E245"
  fi
  [[ -f "$src" ]] || { echo "NS_ANTIGARBLING_GATE_SKIP: no unrelated-refusal fixture"; return 0; }
  set +e; out=$(SOUNIO_NS_DISABLE=1 "$SOUC" check "$src" 2>&1); rc=$?; set -e
  [[ $rc -ne 0 ]] || fail "$code vanished under SOUNIO_NS_DISABLE (knob too broad)"
  echo "$out" | grep -E "$code" >/dev/null || fail "$code fixture refused under SOUNIO_NS_DISABLE but not with $code: $out"
  echo "$out" | grep -E 'E230' >/dev/null && fail "$code fixture also raised E230 under SOUNIO_NS_DISABLE (knob inert)"
  pass "$code survives SOUNIO_NS_DISABLE (knob is NS-specific; E230 and $code causally separable)"
}

refuses_e230        "x+x shared source"  tests/compile-fail/ns_add_shared_source_rejected.sio
vanishes_under_sabotage "x+x shared source" tests/compile-fail/ns_add_shared_source_rejected.sio
refuses_e230        "top operand"        tests/compile-fail/ns_add_unknown_conservative.sio
vanishes_under_sabotage "top operand"     tests/compile-fail/ns_add_unknown_conservative.sio
accepts             "disjoint x+y"       tests/run-pass/ns_union_add_trace.sio
accepts             "disjoint x*y"       tests/run-pass/ns_union_mul_trace.sio
refuses_e230        "ident(x)+x"         tests/compile-fail/ns_add_via_identity_rejected.sio
vanishes_under_sabotage "ident(x)+x"     tests/compile-fail/ns_add_via_identity_rejected.sio
unrelated_refusal_survives_ns_disable

echo "NS_ANTIGARBLING_GATE_OK: all controls passed"
