#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_outcome_authority_adapter.sh"
MODULE="$ROOT_DIR/stdlib/coordination/loom_outcome_authority.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-outcome-adapter.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-outcome-authority-adapter: FAIL: %s\n' "$*" >&2
  exit 1
}

one='1 1 1 1 1 1 1 1'
two='2 2 2 2 2 2 2 2'
three='3 3 3 3 3 3 3 3'
four='4 4 4 4 4 4 4 4'
five='5 5 5 5 5 5 5 5'
six='6 6 6 6 6 6 6 6'
digests="$one $two $two $three $three $four $four $five"
adapter="$WORK/sounio-loom-outcome-authority-runtime"

SOUNIO_LOOM_OUTCOME_AUTHORITY_OUTPUT="$adapter" "$BUILD" >/dev/null

expect_accept() {
  local name="$1" frame="$2" output
  output="$(printf '%s\n' "$frame" | "$adapter")" || fail "$name refused"
  [[ "$output" == \
    'SOUNIO_OUTCOME_AUTHORITY_ACCEPT schema=loom-native-outcome-authority-v0 transition=consume state=verified' ]] || \
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

valid="9012 1 1 101 201 301 401 501 601 701 750 801 901 1001 1101 101 201 301 401 701 750 1201 101 201 301 401 501 701 750 1201 $digests"
role_collapse="9012 1 1 101 201 301 401 501 601 701 750 801 801 1001 1101 101 201 301 401 701 750 1201 101 201 301 401 501 701 750 1201 $digests"
measurement_drift="9012 1 1 101 201 301 401 501 601 701 750 801 901 1001 1101 999 201 301 401 701 750 1201 101 201 301 401 501 701 750 1201 $digests"
head_drift="9012 1 1 101 201 301 401 501 601 701 750 801 901 1001 1101 101 201 301 401 701 751 1201 101 201 301 401 501 701 750 1201 $digests"
classifier_subject_drift="9012 1 1 101 201 301 401 501 601 701 750 801 901 1001 1101 101 201 301 401 701 750 1201 101 201 301 401 501 701 750 1201 $one $two $six $three $three $four $four $five"
spec_drift="9012 1 1 101 201 301 401 501 601 701 750 801 901 1001 1101 101 201 301 401 701 750 1201 101 201 301 401 501 701 750 1201 $one $two $two $three $six $four $four $five"
partition_drift="9012 1 1 101 201 301 401 501 601 701 750 801 901 1001 1101 101 201 301 401 701 750 1201 101 201 301 401 501 701 750 1201 $one $two $two $three $three $four $six $five"
nonce_drift="9012 1 1 101 201 301 401 501 601 701 750 801 901 1001 1101 101 201 301 401 701 750 1201 101 201 301 401 501 701 750 1202 $digests"
measurement_unsigned="9012 0 1 101 201 301 401 501 601 701 750 801 901 1001 1101 101 201 301 401 701 750 1201 101 201 301 401 501 701 750 1201 $digests"
classification_unsigned="9012 1 0 101 201 301 401 501 601 701 750 801 901 1001 1101 101 201 301 401 701 750 1201 101 201 301 401 501 701 750 1201 $digests"

expect_accept valid-consume "$valid"
for entry in \
  "role-collapse|$role_collapse" \
  "measurement-drift|$measurement_drift" \
  "journal-head-drift|$head_drift" \
  "classifier-subject-drift|$classifier_subject_drift" \
  "classifier-spec-drift|$spec_drift" \
  "partition-drift|$partition_drift" \
  "nonce-drift|$nonce_drift" \
  "measurement-unsigned|$measurement_unsigned" \
  "classification-unsigned|$classification_unsigned"; do
  name="${entry%%|*}"
  frame="${entry#*|}"
  expect_refusal "$name" "$frame" 42 \
    'SOUNIO_OUTCOME_AUTHORITY_REFUSE reason=evidence-policy'
done
expect_refusal malformed '9012 1 1 101' 64 \
  'SOUNIO_OUTCOME_AUTHORITY_REFUSE reason=malformed-frame'
expect_refusal overflow \
  '9012 9999999999999999999999999999999999999999' 64 \
  'SOUNIO_OUTCOME_AUTHORITY_REFUSE reason=malformed-frame'

sabotage_rule() {
  local rule="$1" output="$2"
  awk -v target="fn $rule(" '
    BEGIN { in_rule=0; in_body=0; changed=0 }
    $0 == target { in_rule=1; print; next }
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
  local label="$1" rule="$2" frame="$3"
  local source="$WORK/$label.sio" binary="$WORK/$label-runtime" output
  sabotage_rule "$rule" "$source"
  SOUNIO_LOOM_OUTCOME_AUTHORITY_MODULE="$source" \
    SOUNIO_LOOM_OUTCOME_AUTHORITY_OUTPUT="$binary" "$BUILD" >/dev/null
  output="$(printf '%s\n' "$frame" | "$binary")" || \
    fail "$label did not admit its exact control"
  [[ "$output" == \
    'SOUNIO_OUTCOME_AUTHORITY_ACCEPT schema=loom-native-outcome-authority-v0 transition=consume state=verified' ]] || \
    fail "$label emitted: $output"
}

run_sabotage role-sabotage outcome_authority_roles_are_separated \
  "$role_collapse"
run_sabotage measurement-sabotage outcome_measurement_receipt_is_bound \
  "$measurement_drift"
run_sabotage classifier-sabotage outcome_classifier_receipt_is_bound \
  "$spec_drift"
run_sabotage freshness-sabotage outcome_authority_receipt_is_fresh \
  "$nonce_drift"

printf 'loom-outcome-authority-adapter: PASS frame=9012 signatures=2 roles=3 bindings=policy+cursor+head+partition+spec+outcome replay_nonce=correlation-only refusals=9 sabotages=4 same_frame=1 named_rules=roles+measurement+classification+freshness\n'
