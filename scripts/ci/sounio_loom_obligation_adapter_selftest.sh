#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_obligation_adapter.sh"
MODULE="$ROOT_DIR/stdlib/coordination/loom_obligation.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-obligation-adapter.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-obligation-adapter: FAIL: %s\n' "$*" >&2
  exit 1
}

zeros='0 0 0 0 0 0 0 0'
message='1 2 3 4 5 6 7 8'
outcome='11 12 13 14 15 16 17 18'
evidence='21 22 23 24 25 26 27 28'
adapter="$WORK/sounio-loom-obligation-runtime"

SOUNIO_LOOM_OBLIGATION_OUTPUT="$adapter" "$BUILD" >/dev/null

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

expect_accept open \
  "9007 1 0 1 101 0 0 0 0 $message $zeros $zeros" \
  'SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=open state=1'
expect_accept consume \
  "9007 2 1 2 101 201 301 0 400 $message $zeros $zeros" \
  'SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=consume state=2'
expect_accept interrupt-consumed \
  "9007 5 2 4 101 201 301 0 400 $message $zeros $zeros" \
  'SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=interrupt state=4'
expect_refusal interrupt-consumed-wrong-next \
  "9007 5 2 6 101 201 301 0 400 $message $zeros $zeros" 42 \
  'SOUNIO_OBLIGATION_REFUSE reason=transition-policy'
expect_accept claim \
  "9007 3 2 3 101 201 301 401 500 $message $zeros $zeros" \
  'SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=claim state=3'
expect_accept renew \
  "9007 4 3 3 101 201 301 401 600 $message $zeros $zeros" \
  'SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=renew state=3'
expect_accept interrupt \
  "9007 5 3 4 101 201 301 401 600 $message $zeros $zeros" \
  'SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=interrupt state=4'
expect_refusal interrupt-claimed-wrong-next \
  "9007 5 3 6 101 201 301 401 600 $message $zeros $zeros" 42 \
  'SOUNIO_OBLIGATION_REFUSE reason=transition-policy'
expect_accept recover \
  "9007 6 4 5 101 202 302 401 0 $message $zeros $zeros" \
  'SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=recover state=5'
expect_accept complete \
  "9007 7 3 6 101 202 302 402 700 $message $outcome $evidence" \
  'SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=complete state=6'

evidence_less_frame="9007 7 3 6 101 202 302 402 700 $message $outcome $zeros"
expect_refusal evidence-less-completion "$evidence_less_frame" 42 \
  'SOUNIO_OBLIGATION_REFUSE reason=transition-policy'
expect_refusal malformed-frame \
  "9007 1 0 1 101 0 0 0 0 $message $zeros" 64 \
  'SOUNIO_OBLIGATION_REFUSE reason=malformed-frame'

mutated="$WORK/loom_obligation_sabotaged.sio"
awk '
  BEGIN { in_rule=0; in_body=0; changed=0 }
  $0 == "fn completion_evidence_is_bound(" {
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

sabotaged="$WORK/sounio-loom-obligation-sabotaged"
SOUNIO_LOOM_OBLIGATION_MODULE="$mutated" \
  SOUNIO_LOOM_OBLIGATION_OUTPUT="$sabotaged" "$BUILD" >/dev/null
sabotage_output="$(printf '%s\n' "$evidence_less_frame" | "$sabotaged")" || \
  fail 'named-rule sabotage did not admit the exact evidence-less frame'
[[ "$sabotage_output" == \
  'SOUNIO_OBLIGATION_ACCEPT schema=loom-native-obligation-v1 transition=complete state=6' ]] || \
  fail "named-rule sabotage emitted: $sabotage_output"

treatment_log="${SOUNIO_LOOM_OBLIGATION_TREATMENT_LOG:-}"
control_log="${SOUNIO_LOOM_OBLIGATION_CONTROL_LOG:-}"
if [[ -n "$treatment_log" || -n "$control_log" ]]; then
  [[ -n "$treatment_log" && -n "$control_log" ]] || \
    fail 'treatment and control evidence paths must be provided together'
  [[ "$treatment_log" != "$control_log" ]] || \
    fail 'treatment and control evidence paths must be distinct'
  mkdir -p "$(dirname "$treatment_log")" "$(dirname "$control_log")"
  treatment_rc=0
  set +e
  treatment_output="$(printf '%s\n' "$evidence_less_frame" | "$adapter")"
  treatment_rc=$?
  set -e
  [[ "$treatment_rc" -eq 42 ]] || fail "evidence receipt treatment rc=$treatment_rc"
  frame_sha256="$(printf '%s\n' "$evidence_less_frame" | sha256sum | awk '{print $1}')"
  {
    printf 'experiment=exp-loom-obligation-evidence-v1\n'
    printf 'arm=treatment\n'
    printf 'frame_sha256=%s\n' "$frame_sha256"
    printf 'module_sha256=%s\n' "$(sha256sum "$MODULE" | awk '{print $1}')"
    printf 'named_rule=completion_evidence_is_bound\n'
    printf 'mutation=none\n'
    printf 'return_code=%s\n' "$treatment_rc"
    printf 'output=%s\n' "$treatment_output"
  } > "$treatment_log"
  {
    printf 'experiment=exp-loom-obligation-evidence-v1\n'
    printf 'arm=sabotage-control\n'
    printf 'frame_sha256=%s\n' "$frame_sha256"
    printf 'module_sha256=%s\n' "$(sha256sum "$mutated" | awk '{print $1}')"
    printf 'named_rule=completion_evidence_is_bound\n'
    printf 'mutation=replace_only_named_rule_body_with_true\n'
    printf 'mutation_count=1\n'
    printf 'return_code=0\n'
    printf 'output=%s\n' "$sabotage_output"
  } > "$control_log"
fi

printf 'loom-obligation-adapter: PASS frame=9007 transitions=7 malformed=refused evidence_less=refused sabotage=accepted same_frame=1\n'
