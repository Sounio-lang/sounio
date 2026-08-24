#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
BUILD="$ROOT_DIR/scripts/dev/build_sounio_loom_continuity_adapter.sh"
ADAPTER="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-continuity-runtime"
MODULE="$ROOT_DIR/stdlib/coordination/loom_continuity.sio"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-continuity-adapter.XXXXXX")"

cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-continuity-adapter-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

expect_accept() {
  local label="$1" facts="$2" output
  output="$(printf '%s\n' "$facts" | "$ADAPTER")" || \
    fail "$label was refused"
  [[ "$output" == 'SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v1' ]] || \
    fail "$label returned a non-canonical verdict: $output"
}

expect_accept_signed() {
  local label="$1" facts="$2" output
  output="$(printf '%s\n' "$facts" | "$ADAPTER")" || \
    fail "$label was refused"
  [[ "$output" == \
    'SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v2 authenticity=ed25519' ]] || \
    fail "$label returned a non-canonical signed verdict: $output"
}

expect_refuse() {
  local label="$1" facts="$2" expected_rc="$3" output rc=0
  set +e
  output="$(printf '%s\n' "$facts" | "$ADAPTER")"
  rc=$?
  set -e
  [[ "$rc" -eq "$expected_rc" ]] || \
    fail "$label returned rc=$rc instead of rc=$expected_rc"
  [[ "$output" == SOUNIO_CONTINUITY_REFUSE* ]] || \
    fail "$label omitted the Sounio refusal verdict"
}

"$BUILD" >/dev/null
expect_accept initial '101 111 201 301 401 501 0 0 0 0 1 0 0'
expect_accept clean '101 111 202 302 402 502 601 701 801 901 2 1 0'
expect_accept pod '101 111 203 303 403 503 602 702 802 902 3 2 1'
expect_accept_signed signed-initial \
  '101 111 211 311 411 511 0 0 0 0 1 0 0 0 1'
expect_accept_signed signed-pod \
  '101 111 213 313 413 513 612 712 812 912 3 2 1 1001 1'
expect_refuse predecessor-missing \
  '101 111 203 303 403 503 602 702 0 902 3 2 1' 42
expect_refuse pod-count-zero \
  '101 111 203 303 403 503 602 702 802 902 3 2 0' 42
expect_refuse wrong-transition-kind \
  '101 111 203 303 403 503 602 702 802 902 2 2 2' 42
expect_refuse signed-predecessor-receipt-missing \
  '101 111 213 313 413 513 612 712 812 912 3 2 1 0 1' 42
expect_refuse signed-initial-has-predecessor \
  '101 111 211 311 411 511 0 0 0 0 1 0 0 1001 1' 42
expect_refuse extra-field \
  '101 111 203 303 403 503 602 702 802 902 3 2 1 99' 64

mutated="$WORK/loom_continuity_mutated.sio"
mutation_count="$(rg -c '^    if observed\.predecessor_semantic_head_token == 0 \{ return None \}$' "$MODULE")"
[[ "$mutation_count" -eq 1 ]] || \
  fail "expected one predecessor guard before sabotage, got $mutation_count"
awk '
  BEGIN { changed=0 }
  !changed && $0 == "    if observed.predecessor_semantic_head_token == 0 { return None }" {
    print "    if false { return None }"
    changed=1
    next
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$MODULE" > "$mutated" || fail 'could not apply predecessor-guard sabotage'

mutant="$WORK/sounio-loom-continuity-mutant"
SOUNIO_LOOM_CONTINUITY_PREBUILT= \
SOUNIO_LOOM_CONTINUITY_MODULE="$mutated" \
SOUNIO_LOOM_CONTINUITY_OUTPUT="$mutant" \
  "$BUILD" >/dev/null
mutant_output="$(
  printf '%s\n' '101 111 203 303 403 503 602 702 0 902 3 2 1' | "$mutant"
)" || fail 'removing the predecessor guard did not expose the negative witness'
[[ "$mutant_output" == 'SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v1' ]] || \
  fail 'mutated policy did not accept the formerly refused witness'

signed_mutated="$WORK/loom_continuity_signed_mutated.sio"
signed_mutation_count="$(
  rg -c '^    if observed\.authenticity_mode != 1 \|\| observed\.predecessor_receipt_token <= 0 \{$' \
    "$MODULE"
)"
[[ "$signed_mutation_count" -eq 2 ]] || \
  fail "expected two signed predecessor guards before sabotage, got $signed_mutation_count"
awk '
  BEGIN { seen=0; changed=0 }
  $0 == "    if observed.authenticity_mode != 1 || observed.predecessor_receipt_token <= 0 {" {
    seen++
    if (seen == 2) {
      print "    if observed.authenticity_mode != 1 {"
      changed=1
      next
    }
  }
  { print }
  END { if (changed != 1) exit 42 }
' "$MODULE" > "$signed_mutated" || fail 'could not apply signed predecessor sabotage'

signed_mutant="$WORK/sounio-loom-continuity-signed-mutant"
SOUNIO_LOOM_CONTINUITY_PREBUILT= \
SOUNIO_LOOM_CONTINUITY_MODULE="$signed_mutated" \
SOUNIO_LOOM_CONTINUITY_OUTPUT="$signed_mutant" \
  "$BUILD" >/dev/null
signed_mutant_output="$(
  printf '%s\n' '101 111 213 313 413 513 612 712 812 912 3 2 1 0 1' | \
    "$signed_mutant"
)" || fail 'removing the signed predecessor guard did not expose the negative witness'
[[ "$signed_mutant_output" == \
  'SOUNIO_CONTINUITY_ACCEPT schema=loom-native-continuity-v2 authenticity=ed25519' ]] || \
  fail 'mutated signed policy did not accept the formerly refused witness'

echo 'sounio-loom-continuity-adapter-selftest: PASS language=Sounio engine=lean_single transport=stdin initial=accept clean=accept pod=accept signed=accept predecessor=refused signed_predecessor=refused count=refused kind=refused canonical=refused sabotage_predecessor_guard=exposed sabotage_signed_predecessor=exposed'
