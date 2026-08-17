#!/usr/bin/env bash
# Shared assertions for CI gates. Source it:
#
#     . "$(dirname "${BASH_SOURCE[0]}")/../lib/gate_assert.sh"
#     gate_name "my_gate"
#
# WHY THIS EXISTS. A census on 2026-08-04 over the 417 *_gate.{sh,py} in
# scripts/ci and scripts/dev found:
#
#     258 (67%)  no emptiness guard of any kind
#      46 (12%)  an explicit check that the gate is still reading real content
#     125        define their own private fail()
#       0        shared assertion library
#
# The 12% is concentrated in two families. Outside them, a gate that stops
# finding its subject usually reports success. Measured examples on that date:
# run_pass_output_gate.sh printed "PASS (strict): all 0 ... tests" if the
# //@ expect-stdout: marker were renamed; reproduce_artifact.sh printed
# "PASS: test suite" when the suite crashed on startup; the *anti-vacuity lane*
# of eisa_bridge_conformance_gate.sh printed "PASS anti-vacuity" when its own
# extraction came back empty.
#
# The failure is always the same shape: AN EMPTY ANSWER READ AS A MEANINGFUL
# ONE. These helpers make the empty case loud.

if [[ -n "${_SOUNIO_GATE_ASSERT_SOURCED:-}" ]]; then return 0; fi
_SOUNIO_GATE_ASSERT_SOURCED=1

_GATE_NAME="${_GATE_NAME:-gate}"

gate_name() { _GATE_NAME="$1"; }

gate_fail() {
  echo "$(printf '%s' "$_GATE_NAME" | tr '[:lower:]' '[:upper:]')_FAIL: $*" >&2
  exit 1
}

gate_pass() {
  echo "$(printf '%s' "$_GATE_NAME" | tr '[:lower:]' '[:upper:]')_OK${1:+: $1}"
}

require_file() {
  [[ -f "$1" ]] || gate_fail "${2:-missing file: $1}"
}

require_executable() {
  [[ -x "$1" ]] || gate_fail "${2:-not executable: $1}"
}

# The one that would have caught most of the census. Use it on every value
# pulled out of a tool's output before comparing it to anything.
require_nonempty() {
  [[ -n "${1:-}" ]] || gate_fail "${2:-a value the gate depends on came back empty — it is not measuring what it claims}"
}

require_nonempty_file() {
  require_file "$1" "${2:-}"
  [[ -s "$1" ]] || gate_fail "${2:-$1 is empty — whatever produced it produced nothing}"
}

# Counting that distinguishes "no match" from "the tool broke".
#
# `grep -c ... || true` — the dominant idiom in this repo — collapses rc 1
# (no match, count really is 0) and rc 2 (bad regex, unreadable file, tool
# missing) into the same 0. Modelled on scripts/ci/assert_no_rust_markers.sh:38-45,
# the only place in the tree that already gets this right.
#
#   count_matches <pattern> <file> [--fixed]
count_matches() {
  local pattern="$1" file="$2" mode="${3:-}" count rc
  require_file "$file" "count_matches: no such file: $file"
  set +e
  if [[ "$mode" == "--fixed" ]]; then
    count="$(grep -F -c -- "$pattern" "$file")"
  else
    count="$(grep -E -c -- "$pattern" "$file")"
  fi
  rc=$?
  set -e
  case "$rc" in
    0) ;;
    1) count=0 ;;
    *) gate_fail "failed to scan $file for '$pattern' (grep rc=$rc) — this is a tool error, not a zero count" ;;
  esac
  printf '%s' "$count"
}

# A floor on how much the gate saw. This is the anti-vacuity primitive: a sweep
# that finds nothing to check must be red, not green.
require_min_count() {
  local actual="$1" minimum="$2" subject="${3:-items}"
  require_nonempty "$actual" "count of $subject came back empty"
  [[ "$actual" -ge "$minimum" ]] \
    || gate_fail "only $actual $subject (expected at least $minimum) — the gate found nothing to measure, which is not the same as everything passing"
}

# Text anchors. Folds three near-identical private copies that were living in
# madaros_operational_contract_gate.sh:18-31,
# exact_bitwise_rebracket_source_ir_gate.sh:29-41 and
# ordered_path_provenance_source_ir_gate.sh:33-45.
require_text() {
  local pattern="$1" file="$2"
  require_file "$file"
  grep -Fq -- "$pattern" "$file" \
    || gate_fail "missing required anchor in $file: $pattern"
}

require_text_regex() {
  local pattern="$1" file="$2"
  require_file "$file"
  grep -Eq -- "$pattern" "$file" \
    || gate_fail "missing required anchor in $file: /$pattern/"
}

# Run a command and require it to FAIL. The other half of an instrument: a check
# that only ever confirms the good case has not been shown to discriminate.
require_rejects() {
  local why="$1"; shift
  if "$@" >/dev/null 2>&1; then
    gate_fail "the check accepted $why — it is not measuring that"
  fi
}
