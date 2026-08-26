#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_attention_adapter.sh"
MODULE="$ROOT_DIR/stdlib/coordination/loom_attention_compiler.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-attention-adapter.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-attention-adapter: FAIL: %s\n' "$*" >&2
  exit 1
}

zeros='0 0 0 0 0 0 0 0'
evidence='11 12 13 14 15 16 17 18'
falsifier='21 22 23 24 25 26 27 28'
candidates='31 32 33 34 35 36 37 38'
outcome='41 42 43 44 45 46 47 48'
adapter="$WORK/sounio-loom-attention-runtime"

SOUNIO_LOOM_ATTENTION_OUTPUT="$adapter" "$BUILD" >/dev/null

expect_accept() {
  local name="$1" frame="$2" expected="$3" output
  output="$(printf '%s\n' "$frame" | "$adapter")" || fail "$name refused"
  [[ "$output" == "$expected" ]] || fail "$name emitted: $output"
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

expect_accept information-first \
  "9009 1 1 100 101 201 202 301 401 900 500 400 50 100 800 900 900 50 100 $evidence $falsifier $candidates $zeros" \
  'SOUNIO_ATTENTION_ACCEPT schema=loom-native-attention-v0 transition=compile policy=information-first'
expect_accept falsification-first \
  "9009 1 2 100 101 201 202 301 401 400 900 500 50 100 900 800 900 50 100 $evidence $falsifier $candidates $zeros" \
  'SOUNIO_ATTENTION_ACCEPT schema=loom-native-attention-v0 transition=compile policy=falsification-first'
expect_accept counterfactual-first \
  "9009 1 3 100 101 201 202 301 401 400 500 900 50 100 900 900 800 50 100 $evidence $falsifier $candidates $zeros" \
  'SOUNIO_ATTENTION_ACCEPT schema=loom-native-attention-v0 transition=compile policy=counterfactual-first'
expect_accept infeasible-rival \
  "9009 1 1 100 101 201 202 301 401 100 100 100 50 100 900 900 900 101 0 $evidence $falsifier $candidates $zeros" \
  'SOUNIO_ATTENTION_ACCEPT schema=loom-native-attention-v0 transition=compile policy=information-first'
expect_accept completion \
  "9009 2 0 0 101 201 0 301 401 0 0 0 0 0 0 0 0 0 0 $zeros $zeros $zeros $outcome" \
  'SOUNIO_ATTENTION_ACCEPT schema=loom-native-attention-v0 transition=complete state=completed'

dominated_frame="9009 1 1 100 101 201 202 301 401 100 900 900 50 0 900 100 100 50 100 $evidence $falsifier $candidates $zeros"
expect_refusal dominated-selection "$dominated_frame" 42 \
  'SOUNIO_ATTENTION_REFUSE reason=decision-policy'
expect_refusal selected-over-budget \
  "9009 1 1 49 101 201 202 301 401 900 900 900 50 0 100 100 100 10 0 $evidence $falsifier $candidates $zeros" \
  42 'SOUNIO_ATTENTION_REFUSE reason=decision-policy'
expect_refusal completion-without-outcome \
  "9009 2 0 0 101 201 0 301 401 0 0 0 0 0 0 0 0 0 0 $zeros $zeros $zeros $zeros" \
  42 'SOUNIO_ATTENTION_REFUSE reason=decision-policy'
expect_refusal malformed-frame "9009 1 1 100 101" 64 \
  'SOUNIO_ATTENTION_REFUSE reason=malformed-frame'

mutated="$WORK/loom_attention_sabotaged.sio"
awk '
  BEGIN { in_rule=0; in_body=0; changed=0 }
  $0 == "fn attention_selected_not_dominated(" {
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
' "$MODULE" > "$mutated" || fail 'could not apply the named comparator sabotage'

sabotaged="$WORK/sounio-loom-attention-sabotaged"
SOUNIO_LOOM_ATTENTION_MODULE="$mutated" \
  SOUNIO_LOOM_ATTENTION_OUTPUT="$sabotaged" "$BUILD" >/dev/null
sabotage_output="$(printf '%s\n' "$dominated_frame" | "$sabotaged")" || \
  fail 'named comparator sabotage did not admit the exact dominated frame'
[[ "$sabotage_output" == \
  'SOUNIO_ATTENTION_ACCEPT schema=loom-native-attention-v0 transition=compile policy=information-first' ]] || \
  fail "named comparator sabotage emitted: $sabotage_output"

printf 'loom-attention-adapter: PASS frame=9009 policies=3 dominated=refused over_budget=refused completion_without_outcome=refused sabotage=accepted same_frame=1 named_rule=attention_selected_not_dominated\n'
