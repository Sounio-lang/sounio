#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_portfolio_attention_adapter.sh"
MODULE="$ROOT_DIR/stdlib/coordination/loom_portfolio_attention.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-portfolio-adapter.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-portfolio-adapter: FAIL: %s\n' "$*" >&2
  exit 1
}

zeros='0 0 0 0 0 0 0 0'
candidates='11 12 13 14 15 16 17 18'
frontier='21 22 23 24 25 26 27 28'
selected='31 32 33 34 35 36 37 38'
evidence='41 42 43 44 45 46 47 48'
falsifier='51 52 53 54 55 56 57 58'
outcome='61 62 63 64 65 66 67 68'
adapter="$WORK/sounio-loom-portfolio-runtime"

SOUNIO_LOOM_PORTFOLIO_OUTPUT="$adapter" "$BUILD" >/dev/null

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

base_digests="$candidates $frontier $selected $evidence $falsifier $zeros"
expect_accept information-first \
  "9010 1 1 100 100 10 10 101 201 202 301 401 900 500 400 40 50 50 5 5 800 900 900 50 50 50 5 5 $base_digests" \
  'SOUNIO_PORTFOLIO_ACCEPT schema=loom-native-portfolio-v0 transition=compile policy=information-first'
expect_accept falsification-first \
  "9010 1 2 100 100 10 10 101 201 202 301 401 400 900 500 40 50 50 5 5 900 800 900 50 50 50 5 5 $base_digests" \
  'SOUNIO_PORTFOLIO_ACCEPT schema=loom-native-portfolio-v0 transition=compile policy=falsification-first'
expect_accept counterfactual-first \
  "9010 1 3 100 100 10 10 101 201 202 301 401 400 500 900 40 50 50 5 5 900 900 800 50 50 50 5 5 $base_digests" \
  'SOUNIO_PORTFOLIO_ACCEPT schema=loom-native-portfolio-v0 transition=compile policy=counterfactual-first'
expect_accept completion \
  "9010 2 0 0 0 0 0 101 201 0 301 401 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 $zeros $zeros $zeros $zeros $zeros $outcome" \
  'SOUNIO_PORTFOLIO_ACCEPT schema=loom-native-portfolio-v0 transition=complete state=completed'

dominated_frame="9010 1 1 100 100 10 10 101 201 202 301 401 100 900 900 40 50 50 5 5 900 100 100 40 50 50 5 5 $base_digests"
expect_refusal dominated-selection "$dominated_frame" 42 \
  'SOUNIO_PORTFOLIO_REFUSE reason=decision-policy'
expect_refusal token-over-budget \
  "9010 1 1 49 100 10 10 101 201 202 301 401 900 900 900 40 50 50 5 5 100 100 100 10 10 10 1 1 $base_digests" \
  42 'SOUNIO_PORTFOLIO_REFUSE reason=decision-policy'
expect_refusal gpu-over-budget \
  "9010 1 1 100 100 4 10 101 201 202 301 401 900 900 900 40 50 50 5 5 100 100 100 10 10 10 1 1 $base_digests" \
  42 'SOUNIO_PORTFOLIO_REFUSE reason=decision-policy'
expect_refusal completion-without-outcome \
  "9010 2 0 0 0 0 0 101 201 0 301 401 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 $zeros $zeros $zeros $zeros $zeros $zeros" \
  42 'SOUNIO_PORTFOLIO_REFUSE reason=decision-policy'
expect_refusal malformed-frame "9010 1 1 100 100" 64 \
  'SOUNIO_PORTFOLIO_REFUSE reason=malformed-frame'
expect_refusal integer-overflow \
  "9010 999999999999999999999999999999999999999999" 64 \
  'SOUNIO_PORTFOLIO_REFUSE reason=malformed-frame'

mutated="$WORK/loom_portfolio_sabotaged.sio"
awk '
  BEGIN { in_rule=0; in_body=0; changed=0 }
  $0 == "fn portfolio_selected_not_dominated(" {
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
' "$MODULE" > "$mutated" || fail 'could not apply the named portfolio comparator sabotage'

sabotaged="$WORK/sounio-loom-portfolio-sabotaged"
SOUNIO_LOOM_PORTFOLIO_MODULE="$mutated" \
  SOUNIO_LOOM_PORTFOLIO_OUTPUT="$sabotaged" "$BUILD" >/dev/null
sabotage_output="$(printf '%s\n' "$dominated_frame" | "$sabotaged")" || \
  fail 'named portfolio comparator sabotage did not admit the exact dominated frame'
[[ "$sabotage_output" == \
  'SOUNIO_PORTFOLIO_ACCEPT schema=loom-native-portfolio-v0 transition=compile policy=information-first' ]] || \
  fail "named portfolio comparator sabotage emitted: $sabotage_output"

printf 'loom-portfolio-adapter: PASS frame=9010 policies=3 budgets=4 dominated=refused completion_without_outcome=refused overflow=refused sabotage=accepted same_frame=1 named_rule=portfolio_selected_not_dominated\n'
