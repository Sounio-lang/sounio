#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_epistemic_adapter.sh"
MODULE="$ROOT_DIR/stdlib/coordination/loom_epistemic_machine.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-epistemic-adapter.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-epistemic-adapter: FAIL: %s\n' "$*" >&2
  exit 1
}

zeros='0 0 0 0 0 0 0 0'
value='11 12 13 14 15 16 17 18'
error='21 22 23 24 25 26 27 28'
uncertainty='31 32 33 34 35 36 37 38'
confidence='41 42 43 44 45 46 47 48'
provenance='51 52 53 54 55 56 57 58'
evidence='61 62 63 64 65 66 67 68'
falsifier='71 72 73 74 75 76 77 78'
parent_head='81 82 83 84 85 86 87 88'
hypothesis='91 92 93 94 95 96 97 98'
adapter="$WORK/sounio-loom-epistemic-runtime"

SOUNIO_LOOM_EPISTEMIC_OUTPUT="$adapter" "$BUILD" >/dev/null

expect_accept() {
  local name="$1" frame="$2" transition="$3" output
  output="$(printf '%s\n' "$frame" | "$adapter")" || fail "$name refused"
  [[ "$output" == \
    "SOUNIO_EPISTEMIC_ACCEPT schema=loom-native-epistemic-v0 transition=$transition state=active" ]] || \
    fail "$name emitted: $output"
}

expect_refusal() {
  local name="$1" frame="$2" expected_rc="$3" expected="$4" output rc=0
  set +e
  output="$(printf '%s\n' "$frame" | "$adapter")"
  rc=$?
  set -e
  [[ "$rc" -eq "$expected_rc" ]] || fail "$name rc=$rc"
  [[ "$output" == "$expected" ]] || fail "$name emitted: $output"
}

expect_accept create \
  "9008 1 0 1 101 0 0 0 0 0 $zeros $zeros $zeros $zeros $zeros $zeros $zeros" \
  create
expect_accept observe \
  "9008 2 1 1 101 0 201 0 0 0 $value $error $uncertainty $confidence $provenance $zeros $zeros" \
  observe
expect_accept claim \
  "9008 3 1 1 101 0 301 201 0 0 $zeros $zeros $zeros $zeros $zeros $evidence $zeros" \
  claim
expect_accept challenge \
  "9008 4 1 1 101 0 401 301 0 0 $zeros $zeros $zeros $zeros $zeros $zeros $falsifier" \
  challenge
expect_accept capability-acquire \
  "9008 5 1 1 101 0 501 601 701 801 $zeros $zeros $zeros $zeros $zeros $zeros $zeros" \
  capability-acquire
expect_accept capability-release \
  "9008 6 1 1 101 0 501 601 701 801 $zeros $zeros $zeros $zeros $zeros $zeros $zeros" \
  capability-release
expect_accept fork \
  "9008 7 0 1 102 101 201 0 0 0 $zeros $zeros $zeros $zeros $parent_head $hypothesis $zeros" \
  fork

missing_axis_frame="9008 2 1 1 101 0 201 0 0 0 $value $error $zeros $confidence $provenance $zeros $zeros"
expect_refusal missing-uncertainty "$missing_axis_frame" 42 \
  'SOUNIO_EPISTEMIC_REFUSE reason=transition-policy'
expect_refusal challenge-without-falsifier \
  "9008 4 1 1 101 0 401 301 0 0 $zeros $zeros $zeros $zeros $zeros $zeros $zeros" \
  42 'SOUNIO_EPISTEMIC_REFUSE reason=transition-policy'
expect_refusal malformed-frame \
  "9008 1 0 1 101 0 0 0 0 0 $zeros" 64 \
  'SOUNIO_EPISTEMIC_REFUSE reason=malformed-frame'

mutated="$WORK/loom_epistemic_sabotaged.sio"
awk '
  BEGIN { in_rule=0; in_body=0; changed=0 }
  $0 == "fn knowledge_axes_are_bound(" {
    in_rule=1
    print
    next
  }
  in_rule && $0 == ") -> bool {" {
    print
    print "    true"
    in_body=1
    next
  }
  in_body && $0 == "}" {
    print
    in_rule=0
    in_body=0
    changed++
    next
  }
  in_body { next }
  { print }
  END { if (changed != 1) exit 42 }
' "$MODULE" > "$mutated" || fail 'could not apply the named-rule sabotage'

sabotaged="$WORK/sounio-loom-epistemic-sabotaged"
SOUNIO_LOOM_EPISTEMIC_MODULE="$mutated" \
  SOUNIO_LOOM_EPISTEMIC_OUTPUT="$sabotaged" "$BUILD" >/dev/null
sabotage_output="$(printf '%s\n' "$missing_axis_frame" | "$sabotaged")" || \
  fail 'named-rule sabotage did not admit the exact missing-axis frame'
[[ "$sabotage_output" == \
  'SOUNIO_EPISTEMIC_ACCEPT schema=loom-native-epistemic-v0 transition=observe state=active' ]] || \
  fail "named-rule sabotage emitted: $sabotage_output"

printf 'loom-epistemic-adapter: PASS frame=9008 transitions=7 missing_axis=refused challenge_without_falsifier=refused sabotage=accepted same_frame=1 named_rule=knowledge_axes_are_bound\n'
