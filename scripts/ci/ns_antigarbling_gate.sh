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
  echo "$out" | grep -Eq 'E230' || fail "$label refused but not with E230: $out"
  pass "$label refused with E230"
}

# Same file COMPILES with NS disabled (the refusal vanishes -> caused by NS).
vanishes_under_sabotage() {
  local label="$1" src="$2" out rc
  set +e; out=$(SOUNIO_NS_DISABLE=1 "$SOUC" check "$src" 2>&1); rc=$?; set -e
  echo "$out" | grep -Eq 'E230' && fail "$label E230 survived NS-disable (not caused by NS): $out"
  pass "$label E230 vanishes under SOUNIO_NS_DISABLE"
}

# A disjoint add still COMPILES (the rule is not a blanket ban on Knowledge arithmetic).
accepts() {
  local label="$1" src="$2" out rc
  set +e; out=$("$SOUC" check "$src" 2>&1); rc=$?; set -e
  echo "$out" | grep -Eq 'E230' && fail "$label wrongly refused a disjoint op: $out"
  pass "$label accepted (disjoint)"
}

# An UNRELATED refusal (E222 R-ORIGIN) must be UNAFFECTED by the NS knob.
e222_survives_ns_disable() {
  local src="tests/compile-fail/r_origin_measured_on_sum.sio" out rc
  [[ -f "$src" ]] || { echo "NS_ANTIGARBLING_GATE_SKIP: no R-ORIGIN fixture"; return 0; }
  set +e; out=$(SOUNIO_NS_DISABLE=1 "$SOUC" check "$src" 2>&1); rc=$?; set -e
  [[ $rc -ne 0 ]] || fail "E222 vanished under SOUNIO_NS_DISABLE (knob too broad)"
  pass "E222 survives SOUNIO_NS_DISABLE (knob is NS-specific)"
}

refuses_e230        "x+x shared source"  tests/compile-fail/ns_add_shared_source_rejected.sio
vanishes_under_sabotage "x+x shared source" tests/compile-fail/ns_add_shared_source_rejected.sio
refuses_e230        "top operand"        tests/compile-fail/ns_add_unknown_conservative.sio
vanishes_under_sabotage "top operand"     tests/compile-fail/ns_add_unknown_conservative.sio
accepts             "disjoint x+y"       tests/run-pass/ns_union_add_trace.sio
e222_survives_ns_disable

echo "NS_ANTIGARBLING_GATE_OK: all controls passed"
