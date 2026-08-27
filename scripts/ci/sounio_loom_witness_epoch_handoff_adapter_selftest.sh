#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_witness_epoch_handoff_adapter.sh"
MODULE="$ROOT_DIR/stdlib/coordination/loom_witness_epoch_handoff.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-epoch-adapter.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-witness-epoch-handoff-adapter: FAIL: %s\n' "$*" >&2
  exit 1
}

zero='0 0 0 0 0 0 0 0'
one='1 1 1 1 1 1 1 1'
two='2 2 2 2 2 2 2 2'
three='3 3 3 3 3 3 3 3'
four='4 4 4 4 4 4 4 4'
five='5 5 5 5 5 5 5 5'
six='6 6 6 6 6 6 6 6'
seven='7 7 7 7 7 7 7 7'
eight='8 8 8 8 8 8 8 8'
adapter="$WORK/sounio-loom-witness-epoch-handoff-runtime"

SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_OUTPUT="$adapter" "$BUILD" >/dev/null

expect_accept() {
  local name="$1" frame="$2" output
  output="$(printf '%s\n' "$frame" | "$adapter")" || fail "$name refused"
  [[ "$output" == \
    'SOUNIO_WITNESS_EPOCH_HANDOFF_ACCEPT schema=loom-native-witness-epoch-handoff-v0 transition=joint-quorum state=prepared' ]] || \
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

valid="9015 1 1 1 2 3 3 4 4 501 501 7 1 12 12 $one $two $three $four $five $five $six $seven $zero"
unverified_quorum="9015 0 1 1 2 3 3 4 4 501 501 7 1 12 12 $one $two $three $four $five $five $six $seven $zero"
epoch_skip="9015 1 1 1 3 3 3 4 4 501 501 7 1 12 12 $one $two $three $four $five $five $six $seven $zero"
membership_reuse="9015 1 1 1 2 3 3 4 4 501 501 7 1 12 12 $one $one $three $four $five $five $six $seven $zero"
root_reuse="9015 1 1 1 2 3 3 4 4 501 501 7 1 12 12 $one $two $three $three $five $five $six $seven $zero"
checkpoint_drift="9015 1 1 1 2 3 3 4 4 501 501 7 1 12 12 $one $two $three $four $five $eight $six $seven $zero"
certificate_reuse="9015 1 1 1 2 3 3 4 4 501 501 7 1 12 12 $one $two $three $four $five $five $six $six $zero"
missing_predecessor="9015 1 1 2 3 3 3 4 4 501 501 7 1 12 12 $one $two $three $four $five $five $six $seven $zero"
epoch_out_of_range="9015 1 1 65 66 3 3 4 4 501 501 7 1 12 12 $one $two $three $four $five $five $six $seven $eight"

expect_accept valid-handoff "$valid"
for entry in \
  "unverified-quorum|$unverified_quorum" \
  "epoch-skip|$epoch_skip" \
  "membership-reuse|$membership_reuse" \
  "root-reuse|$root_reuse" \
  "checkpoint-drift|$checkpoint_drift" \
  "certificate-reuse|$certificate_reuse" \
  "missing-predecessor|$missing_predecessor" \
  "epoch-out-of-range|$epoch_out_of_range"; do
  name="${entry%%|*}"
  frame="${entry#*|}"
  expect_refusal "$name" "$frame" 42 \
    'SOUNIO_WITNESS_EPOCH_HANDOFF_REFUSE reason=handoff-policy'
done
expect_refusal malformed '9015 1 1' 64 \
  'SOUNIO_WITNESS_EPOCH_HANDOFF_REFUSE reason=malformed-frame'
expect_refusal overflow \
  '9015 9999999999999999999999999999999999999999' 64 \
  'SOUNIO_WITNESS_EPOCH_HANDOFF_REFUSE reason=malformed-frame'

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
  SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_MODULE="$source" \
    SOUNIO_LOOM_WITNESS_EPOCH_HANDOFF_OUTPUT="$binary" "$BUILD" >/dev/null
  output="$(printf '%s\n' "$frame" | "$binary")" || \
    fail "$label did not admit its exact control"
  [[ "$output" == \
    'SOUNIO_WITNESS_EPOCH_HANDOFF_ACCEPT schema=loom-native-witness-epoch-handoff-v0 transition=joint-quorum state=prepared' ]] || \
    fail "$label emitted: $output"
}

run_sabotage quorum-sabotage witness_epoch_quorums_are_verified \
  "$unverified_quorum"
run_sabotage epoch-sabotage witness_epoch_epochs_are_adjacent \
  "$epoch_skip"
run_sabotage membership-sabotage witness_epoch_memberships_change \
  "$membership_reuse"
run_sabotage root-sabotage witness_epoch_roots_are_separated \
  "$root_reuse"
run_sabotage checkpoint-sabotage witness_epoch_checkpoint_agrees \
  "$checkpoint_drift"
run_sabotage certificate-sabotage witness_epoch_certificates_are_bound \
  "$certificate_reuse"
run_sabotage chain-sabotage witness_epoch_chain_is_well_formed \
  "$missing_predecessor"

printf 'loom-witness-epoch-handoff-adapter: PASS frame=9015 joint_quorum=3/4+3/4 refusals=8 sabotages=7 same_frame=1 max_transitions=64 named_rules=quorums+epochs+memberships+roots+checkpoint+certificates+chain\n'
