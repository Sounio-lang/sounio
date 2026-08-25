#!/usr/bin/env bash
# NS N2 dataflow trace witnesses — run under SOUNIO_NS_DISABLE.
#
# These five tests were written as N2 dataflow witnesses (top production, union,
# call-projection, source-cap overflow, loop widen). Each observes the analysis by
# performing an operation that N3 now (correctly) REFUSES with E230 -- a shared
# source (s+s) or a top operand (u+m). So under the default checker they no longer
# compile, and they carry `//@ known-failure` in the standard suite.
#
# Their N2 witness is still valid with the N3 refusal turned OFF: SOUNIO_NS_DISABLE=1
# disables ONLY the anti-garbling gate, leaving the noise-set dataflow (and its
# SOUNIO_NS_TRACE output) intact. This gate is their real home -- it runs each with
# the refusal off and a large stack (these bodies overflow the default frame, a
# pre-existing runtime property unrelated to NS), and asserts rc=0 + the PASS line.
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
  want="$(grep -oE 'expect-stdout: .*' "$src" | sed 's/expect-stdout: //')"
  # Big stack: these witness bodies overflow the default frame (pre-existing,
  # not an NS property). SOUNIO_NS_DISABLE=1 turns off ONLY the E230 refusal.
  out="$(SOUNIO_NS_DISABLE=1 bash -c "ulimit -s unlimited 2>/dev/null; timeout 120 '$SOUC' run '$src' 2>&1")"
  rc=$?
  [[ $rc -eq 0 ]] || fail "$w: rc=$rc under SOUNIO_NS_DISABLE: $out"
  if [[ -n "$want" ]]; then
    echo "$out" | grep -Fq "$want" || fail "$w: missing '$want': $out"
  fi
  # And confirm the refusal is real: WITHOUT the knob the same file is refused (E230).
  ref="$("$SOUC" check "$src" 2>&1)"
  echo "$ref" | grep -Eq 'E230' || fail "$w: expected E230 under default checker (witness no longer flips): $ref"
  pass "$w: PASS under SOUNIO_NS_DISABLE; E230 under default"
done

echo "NS_TRACE_GATE_OK: all five N2 dataflow witnesses reconciled"
