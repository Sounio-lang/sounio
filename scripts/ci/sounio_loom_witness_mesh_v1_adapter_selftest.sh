#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_witness_mesh_v1_adapter.sh"
MODULE="$ROOT_DIR/stdlib/coordination/loom_witness_mesh_v1.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-v1-adapter.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-witness-mesh-v1-adapter: FAIL: %s\n' "$*" >&2
  exit 1
}

one='1 1 1 1 1 1 1 1'
two='2 2 2 2 2 2 2 2'
three='3 3 3 3 3 3 3 3'
zero='0 0 0 0 0 0 0 0'
adapter="$WORK/sounio-loom-witness-mesh-v1-runtime"

SOUNIO_LOOM_WITNESS_MESH_V1_OUTPUT="$adapter" "$BUILD" >/dev/null

expect_accept() {
  local name="$1" frame="$2" output
  output="$(printf '%s\n' "$frame" | "$adapter")" || fail "$name refused"
  [[ "$output" == \
    'SOUNIO_WITNESS_MESH_V1_ACCEPT schema=loom-native-witness-mesh-v1 transition=anchor state=quorum-verified' ]] || \
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

valid="9014 3 1 1 1 0 101 201 301 401 101 201 301 0 501 501 501 501 0 0 1 1 1 1 0 0 3 3 3 3 0 $one $one $one $one $zero $two $two $two $two $zero"
member_collapse="9014 3 1 1 1 0 101 101 301 401 101 101 301 0 501 501 501 501 0 0 1 1 1 1 0 0 3 3 3 3 0 $one $one $one $one $zero $two $two $two $two $zero"
two_share="9014 3 1 1 0 0 101 201 301 401 101 201 0 0 501 501 501 0 0 0 1 1 1 0 0 0 3 3 3 0 0 $one $one $one $zero $zero $two $two $two $zero $zero"
membership_drift="9014 3 1 1 1 0 101 201 301 401 101 201 301 0 501 501 501 501 0 0 1 1 1 1 0 0 3 3 3 3 0 $one $one $three $one $zero $two $two $two $two $zero"
checkpoint_drift="9014 3 1 1 1 0 101 201 301 401 101 201 301 0 501 501 501 501 0 0 1 1 1 1 0 0 3 3 3 3 0 $one $one $one $one $zero $two $two $three $two $zero"
non_advance="9014 3 1 1 1 0 101 201 301 401 101 201 301 0 501 501 501 501 0 1 1 1 1 1 0 0 3 3 3 3 0 $one $one $one $one $zero $two $two $two $two $zero"

expect_accept valid-anchor "$valid"
for entry in \
  "member-collapse|$member_collapse" \
  "two-share|$two_share" \
  "membership-drift|$membership_drift" \
  "checkpoint-drift|$checkpoint_drift" \
  "non-advance|$non_advance"; do
  name="${entry%%|*}"
  frame="${entry#*|}"
  expect_refusal "$name" "$frame" 42 \
    'SOUNIO_WITNESS_MESH_V1_REFUSE reason=checkpoint-policy'
done
expect_refusal malformed '9014 3 1' 64 \
  'SOUNIO_WITNESS_MESH_V1_REFUSE reason=malformed-frame'
expect_refusal overflow \
  '9014 9999999999999999999999999999999999999999' 64 \
  'SOUNIO_WITNESS_MESH_V1_REFUSE reason=malformed-frame'

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
  SOUNIO_LOOM_WITNESS_MESH_V1_MODULE="$source" \
    SOUNIO_LOOM_WITNESS_MESH_V1_OUTPUT="$binary" "$BUILD" >/dev/null
  output="$(printf '%s\n' "$frame" | "$binary")" || \
    fail "$label did not admit its exact control"
  [[ "$output" == \
    'SOUNIO_WITNESS_MESH_V1_ACCEPT schema=loom-native-witness-mesh-v1 transition=anchor state=quorum-verified' ]] || \
    fail "$label emitted: $output"
}

run_sabotage member-sabotage witness_mesh_v1_members_are_separated \
  "$member_collapse"
run_sabotage quorum-sabotage witness_mesh_v1_quorum_is_satisfied \
  "$two_share"
run_sabotage membership-sabotage witness_mesh_v1_receipts_match_membership \
  "$membership_drift"
run_sabotage checkpoint-sabotage witness_mesh_v1_checkpoint_agrees \
  "$checkpoint_drift"
run_sabotage advance-sabotage witness_mesh_v1_checkpoint_advances \
  "$non_advance"

printf 'loom-witness-mesh-v1-adapter: PASS frame=9014 quorum=3/4 intersection_min=2 f=1 refusals=5 sabotages=5 same_frame=1 named_rules=members+quorum+membership+checkpoint+advance\n'
