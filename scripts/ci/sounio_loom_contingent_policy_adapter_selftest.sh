#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_contingent_policy_adapter.sh"
MODULE="$ROOT_DIR/stdlib/coordination/loom_contingent_policy.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-contingent-adapter.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-contingent-adapter: FAIL: %s\n' "$*" >&2
  exit 1
}

zeros='0 0 0 0 0 0 0 0'
actions='11 12 13 14 15 16 17 18'
outcomes='21 22 23 24 25 26 27 28'
frontier='31 32 33 34 35 36 37 38'
selected='41 42 43 44 45 46 47 48'
evidence='51 52 53 54 55 56 57 58'
falsifier='61 62 63 64 65 66 67 68'
branch='71 72 73 74 75 76 77 78'
outcome='81 82 83 84 85 86 87 88'
adapter="$WORK/sounio-loom-contingent-runtime"

SOUNIO_LOOM_CONTINGENT_OUTPUT="$adapter" "$BUILD" >/dev/null

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

compile_digests="$actions $outcomes $frontier $selected $evidence $falsifier $branch $zeros"
partition_digests="$zeros $outcomes $zeros $zeros $zeros $zeros $branch $zeros"
observe_digests="$zeros $zeros $zeros $selected $zeros $zeros $branch $outcome"

expect_accept information-first \
  "9011 1 1 0 100 100 10 10 101 201 202 301 0 0 0 401 501 900 500 400 40 50 50 5 5 800 900 900 50 50 50 5 5 $compile_digests" \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=compile policy=information-first'
expect_accept falsification-first \
  "9011 1 2 0 100 100 10 10 101 201 202 301 0 0 0 401 501 400 900 500 40 50 50 5 5 900 800 900 50 50 50 5 5 $compile_digests" \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=compile policy=falsification-first'
expect_accept counterfactual-first \
  "9011 1 3 0 100 100 10 10 101 201 202 301 0 0 0 401 501 400 500 900 40 50 50 5 5 900 900 800 50 50 50 5 5 $compile_digests" \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=compile policy=counterfactual-first'

expect_accept partition-one \
  "9011 3 0 0 0 0 0 0 101 0 0 301 0 0 0 401 501 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 $partition_digests" \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=partition state=total'
expect_accept partition-two \
  "9011 3 0 0 0 0 0 0 101 0 0 301 0 0 0 401 501 2 2 1 2 0 0 0 0 0 0 0 0 0 0 0 0 $partition_digests" \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=partition state=total'
expect_accept partition-three \
  "9011 3 0 0 0 0 0 0 101 0 0 301 0 0 0 401 501 3 3 1 2 3 0 0 0 0 0 0 0 0 0 0 0 $partition_digests" \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=partition state=total'

expect_accept observe-advance \
  "9011 2 0 0 0 0 0 0 101 201 0 301 601 302 302 401 501 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 $observe_digests" \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=observe state=advanced'
expect_accept observe-terminal \
  "9011 2 0 1 0 0 0 0 101 201 0 301 601 0 0 401 501 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 $observe_digests" \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=observe state=completed'

dominated_frame="9011 1 1 0 100 100 10 10 101 201 202 301 0 0 0 401 501 100 900 900 40 50 50 5 5 900 100 100 40 50 50 5 5 $compile_digests"
incomplete_partition_frame="9011 3 0 0 0 0 0 0 101 0 0 301 0 0 0 401 501 3 2 1 2 0 0 0 0 0 0 0 0 0 0 0 0 $partition_digests"
wrong_route_frame="9011 2 0 0 0 0 0 0 101 201 0 301 601 302 999 401 501 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 $observe_digests"

expect_refusal dominated-selection "$dominated_frame" 42 \
  'SOUNIO_CONTINGENT_REFUSE reason=decision-policy'
expect_refusal incomplete-partition "$incomplete_partition_frame" 42 \
  'SOUNIO_CONTINGENT_REFUSE reason=decision-policy'
expect_refusal wrong-next-action "$wrong_route_frame" 42 \
  'SOUNIO_CONTINGENT_REFUSE reason=decision-policy'
expect_refusal observe-without-outcome \
  "9011 2 0 1 0 0 0 0 101 201 0 301 601 0 0 401 501 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 $zeros $zeros $zeros $selected $zeros $zeros $branch $zeros" \
  42 'SOUNIO_CONTINGENT_REFUSE reason=decision-policy'
expect_refusal malformed-frame "9011 1 1 0 100" 64 \
  'SOUNIO_CONTINGENT_REFUSE reason=malformed-frame'
expect_refusal integer-overflow \
  "9011 999999999999999999999999999999999999999999" 64 \
  'SOUNIO_CONTINGENT_REFUSE reason=malformed-frame'

sabotage_rule() {
  local rule="$1" output="$2"
  awk -v target="fn $rule(" '
    BEGIN { in_rule=0; in_body=0; changed=0 }
    $0 == target {
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
  ' "$MODULE" > "$output" || fail "could not sabotage named rule $rule"
}

run_sabotage() {
  local label="$1" rule="$2" frame="$3" expected="$4"
  local source="$WORK/$label.sio" binary="$WORK/$label-runtime" output
  sabotage_rule "$rule" "$source"
  SOUNIO_LOOM_CONTINGENT_MODULE="$source" \
    SOUNIO_LOOM_CONTINGENT_OUTPUT="$binary" "$BUILD" >/dev/null
  output="$(printf '%s\n' "$frame" | "$binary")" || \
    fail "$label did not admit its exact control"
  [[ "$output" == "$expected" ]] || fail "$label emitted: $output"
}

run_sabotage comparator-sabotage contingent_policy_selected_not_dominated \
  "$dominated_frame" \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=compile policy=information-first'
run_sabotage partition-sabotage contingent_outcome_partition_is_total \
  "$incomplete_partition_frame" \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=partition state=total'
run_sabotage routing-sabotage contingent_transition_matches_branch \
  "$wrong_route_frame" \
  'SOUNIO_CONTINGENT_ACCEPT schema=loom-native-contingent-policy-v0 transition=observe state=advanced'

printf 'loom-contingent-adapter: PASS frame=9011 policies=3 budgets=4 partition=total dominated=refused wrong_route=refused overflow=refused sabotages=3 same_frame=1 named_rules=selection+partition+routing\n'
