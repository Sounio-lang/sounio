#!/usr/bin/env bash
# NS N2 dataflow trace witnesses — observed under SOUNIO_NS_DISABLE.
#
# These five tests were written as N2 dataflow witnesses (top production, union,
# call-projection, source-cap overflow, loop widen). Each observes the analysis by
# performing an operation that N3 now (correctly) REFUSES with E230 -- a shared
# source (s+s) or a top operand (u+m). So under the default checker they no longer
# compile, and they carry `//@ known-failure` in the standard suite.
#
# Their N2 witness is still valid with the N3 refusal turned OFF: SOUNIO_NS_DISABLE=1
# disables ONLY the anti-garbling gate, leaving the noise-set dataflow (and its
# SOUNIO_NS_TRACE output) intact. This gate is their real home.
#
# WHAT IS MEASURED ON MAIN, AND WHAT IS NOT. The wire's original form of this
# gate ran each witness (`souc run`) and asserted rc=0 + its PASS line. On main
# every one of these bodies performs Knowledge x Knowledge arithmetic, which
# Madaros refuses with E245 (no lowering, #1706) INDEPENDENTLY of the NS knob --
# so no build/run verdict is available from this engine. What this gate can
# still measure, and does, is the checker-level dataflow: under
# SOUNIO_NS_DISABLE=1 SOUNIO_NS_TRACE=1 `souc check` the NS_TRACE lines are
# emitted and NO E230 is raised; under the default checker the SAME file IS
# refused with E230. The run-level assertion is deliberately not faked with a
# knob that would hide E245.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUC_BIN || true
unset SOUNIO_SOUC_BIN || true

fail() { echo "NS_TRACE_GATE_FAIL: $*" >&2; exit 1; }
pass() { echo "NS_TRACE_GATE_OK: $*"; }

SOUC="${SOUC:-$ROOT/bin/souc}"
[[ -x "$SOUC" ]] || fail "no souc at $SOUC"
echo "souc=$SOUC"

WITNESSES=(
  ns_env_join_overflow_trace
  ns_unknown_absorb_trace
  ns_call_projection_top_trace
  ns_source_cap_unknown_trace
  ns_loop_widen_top_trace
)

for w in "${WITNESSES[@]}"; do
  src="tests/run-pass/${w}.sio"
  [[ -f "$src" ]] || fail "$w: missing $src"
  # Big stack: these witness bodies overflow the default frame (pre-existing,
  # not an NS property). SOUNIO_NS_DISABLE=1 turns off ONLY the E230 refusal.
  out="$(SOUNIO_NS_DISABLE=1 SOUNIO_NS_TRACE=1 bash -c "ulimit -s unlimited 2>/dev/null; timeout 120 '$SOUC' check '$src' 2>&1")"
  n_trace=$(printf '%s\n' "$out" | grep -c '^NS_TRACE ')
  [[ $n_trace -gt 0 ]] || fail "$w: no NS_TRACE lines under SOUNIO_NS_DISABLE=1 SOUNIO_NS_TRACE=1 (dataflow not observable): $out"
  printf '%s\n' "$out" | grep -E 'E230' >/dev/null && fail "$w: E230 survived SOUNIO_NS_DISABLE (knob inert): $out"
  # And confirm the refusal is real: WITHOUT the knob the same file is refused (E230).
  ref="$("$SOUC" check "$src" 2>&1)"
  printf '%s\n' "$ref" | grep -E 'E230' >/dev/null || fail "$w: expected E230 under default checker (witness no longer flips): $ref"
  pass "$w: $n_trace NS_TRACE line(s) and no E230 under SOUNIO_NS_DISABLE; E230 under default"
done

echo "NS_TRACE_GATE_OK: all five N2 dataflow witnesses reconciled (check-level; see header for what is not measured)"
