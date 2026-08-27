#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_witness_epoch_transparency_adapter.sh"
MODULE="$ROOT_DIR/stdlib/coordination/loom_witness_epoch_transparency.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-witness-epoch-transparency-adapter.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'loom-witness-epoch-transparency-adapter: FAIL: %s\n' "$*" >&2
  exit 1
}

one='1 1 1 1 1 1 1 1'
two='2 2 2 2 2 2 2 2'
three='3 3 3 3 3 3 3 3'
four='4 4 4 4 4 4 4 4'
five='5 5 5 5 5 5 5 5'
six='6 6 6 6 6 6 6 6'
seven='7 7 7 7 7 7 7 7'
zero='0 0 0 0 0 0 0 0'

valid="9016 1 1 1 1 1 1 1 1 1 3 4 1 2 0 1 1 1 1 101 202 301 302 303 304 $one $one $zero $zero $two $two $three $three $four $five $six $seven"
unreachable="9016 1 0 1 1 1 1 1 1 1 3 4 1 2 0 1 1 1 1 101 202 301 302 303 304 $one $one $zero $zero $two $two $three $three $four $five $six $seven"
unverified_handoff="9016 0 1 1 1 1 1 1 1 1 3 4 1 2 0 1 1 1 1 101 202 301 302 303 304 $one $one $zero $zero $two $two $three $three $four $five $six $seven"
quorum_short="9016 1 1 1 1 1 1 1 1 1 2 4 1 2 0 1 1 1 1 101 202 301 302 303 304 $one $one $zero $zero $two $two $three $three $four $five $six $seven"
operator_collapse="9016 1 1 1 1 1 1 1 1 1 3 4 1 2 0 1 1 1 1 301 202 301 302 303 304 $one $one $zero $zero $two $two $three $three $four $five $six $seven"
non_monotonic_root="9016 1 1 1 1 1 1 1 1 1 3 4 1 2 0 1 1 1 1 101 202 301 302 303 304 $one $one $zero $zero $two $two $two $two $four $five $six $seven"
split_view="9016 1 1 1 1 1 1 1 1 1 3 4 1 2 0 1 1 1 1 101 202 301 302 303 304 $one $one $zero $zero $two $two $three $four $four $five $six $seven"
forged_inclusion="9016 1 1 1 0 1 1 1 1 1 3 4 1 2 0 1 1 1 1 101 202 301 302 303 304 $one $one $zero $zero $two $two $three $three $four $five $six $seven"
reordered_handoff="9016 1 1 1 1 1 1 1 1 1 3 4 1 2 0 1 1 1 1 101 202 301 302 303 304 $one $two $zero $zero $two $two $three $three $four $five $six $seven"
stale_sth="9016 1 1 1 1 1 1 1 0 1 3 4 1 2 0 1 1 1 1 101 202 301 302 303 304 $one $one $zero $zero $two $two $three $three $four $five $six $seven"

adapter="$WORK/sounio-loom-witness-epoch-transparency-runtime"
SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_OUTPUT="$adapter" "$BUILD" >/dev/null

accept='SOUNIO_WITNESS_EPOCH_TRANSPARENCY_ACCEPT schema=loom-native-witness-epoch-transparency-v0 rollback_bound=latest-quorum-witnessed-epoch state=verified'
refuse='SOUNIO_WITNESS_EPOCH_TRANSPARENCY_REFUSE reason=transparency-policy'

run_frame() {
  local binary="$1" frame="$2" output rc=0
  set +e
  output="$(printf '%s\n' "$frame" | "$binary")"
  rc=$?
  set -e
  printf '%s\n%s\n' "$rc" "$output"
}

positive="$(run_frame "$adapter" "$valid")"
[[ "$positive" == $'0\n'"$accept" ]] || fail "valid frame was not accepted: $positive"

expect_refusal() {
  local label="$1" frame="$2" result
  result="$(run_frame "$adapter" "$frame")"
  [[ "$result" == $'42\n'"$refuse" ]] || fail "$label was not refused: $result"
}

expect_refusal unreachable-log "$unreachable"
expect_refusal unverified-handoff "$unverified_handoff"
expect_refusal quorum-short "$quorum_short"
expect_refusal operator-collapse "$operator_collapse"
expect_refusal non-monotonic-sth "$non_monotonic_root"
expect_refusal split-view "$split_view"
expect_refusal forged-inclusion "$forged_inclusion"
expect_refusal reordered-handoff "$reordered_handoff"
expect_refusal stale-sth "$stale_sth"

malformed="$(run_frame "$adapter" '9016 1 1')"
[[ "$malformed" == $'64\nSOUNIO_WITNESS_EPOCH_TRANSPARENCY_REFUSE reason=malformed-frame' ]] || \
  fail "malformed frame was not refused: $malformed"

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

expect_sabotage_admits() {
  local label="$1" rule="$2" frame="$3"
  local sabotaged_module="$WORK/$label-module.sio"
  local sabotaged_binary="$WORK/$label-runtime"
  local result
  sabotage_rule "$rule" "$sabotaged_module"
  SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_MODULE="$sabotaged_module" \
    SOUNIO_LOOM_WITNESS_EPOCH_TRANSPARENCY_OUTPUT="$sabotaged_binary" \
    "$BUILD" >/dev/null
  result="$(run_frame "$sabotaged_binary" "$frame")"
  [[ "$result" == $'0\n'"$accept" ]] || \
    fail "$label did not isolate rule necessity: $result"
}

expect_sabotage_admits handoff-rule \
  epoch_transparency_handoff_is_verified "$unverified_handoff"
expect_sabotage_admits reachability-rule \
  epoch_transparency_log_is_reachable_and_signed "$unreachable"
expect_sabotage_admits quorum-rule \
  epoch_transparency_quorum_is_verified "$quorum_short"
expect_sabotage_admits independence-rule \
  epoch_transparency_operator_is_independent "$operator_collapse"
expect_sabotage_admits monotonic-rule \
  epoch_transparency_append_is_monotonic "$non_monotonic_root"
expect_sabotage_admits proof-rule \
  epoch_transparency_proofs_are_verified "$split_view"
expect_sabotage_admits leaf-binding-rule \
  epoch_transparency_leaf_binds_handoff "$reordered_handoff"
expect_sabotage_admits latest-rule \
  epoch_transparency_latest_epoch_matches "$stale_sth"

printf 'loom-witness-epoch-transparency-adapter: PASS frame=9016 positive=1 refusals=10 same_frame_sabotages=8 unreachable=REFUSED split_view=REFUSED forged_inclusion=REFUSED reordered_handoff=REFUSED stale_sth=REFUSED independence_collapse=REFUSED freeze_claim=NONE\n'
