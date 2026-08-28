#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-resident-authority.XXXXXX")"
RUNTIME="$TEST_ROOT/resident-authority"
MODULE="$ROOT_DIR/stdlib/coordination/loom_resident_authority.sio"
ENTRYPOINT="$ROOT_DIR/tools/loom/resident_authority_main.sio"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-resident-authority-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

SOUNIO_LOOM_RESIDENT_AUTHORITY_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_resident_authority.sh" >/dev/null

selftest="$(printf '0\n' | "$RUNTIME")"
[[ "$selftest" == 'SOUNIO_RESIDENT_AUTHORITY_SELFTEST PASS cases=18' ]] ||
  fail "unexpected Sounio selftest: $selftest"

one='1 1 1 1 1 1 1 1'
zero='0 0 0 0 0 0 0 0'

valid_start="9024 3 1 1 1 0 0 0 0 1 1 1 0 $one $one $zero $zero $one"
valid_request="9024 3 2 1 1 1 0 1 0 1 1 1 0 $one $one $one $zero $one"
valid_response="9024 3 3 1 1 1 0 1 1 1 1 1 0 $one $one $one $one $one"
valid_stop="9024 3 4 1 1 1 1 0 0 1 1 1 0 $one $one $zero $zero $one"
replayed_request="9024 3 2 1 1 1 1 1 0 1 1 1 0 $one $one $one $zero $one"
skipped_request="9024 3 2 1 1 3 1 1 0 1 1 1 0 $one $one $one $zero $one"
orphan_request="9024 3 2 0 1 1 0 1 0 1 1 1 0 $one $one $one $zero $one"
uncorrelated_response="9024 3 3 1 1 1 0 1 1 0 1 1 0 $one $one $one $one $one"
no_deadline="9024 3 2 1 1 1 0 1 0 1 0 1 0 $one $one $one $zero $zero"
poisoned_generation="9024 3 2 1 1 1 0 1 0 1 1 1 1 $one $one $one $zero $one"

assert_output() {
  local label="$1" frame="$2" expected="$3"
  local actual
  actual="$(printf '%s\n' "$frame" | "$RUNTIME" || true)"
  [[ "$actual" == "$expected" ]] || fail "$label: $actual"
}

assert_output valid-start "$valid_start" \
  'SOUNIO_RESIDENT_AUTHORITY_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output valid-request "$valid_request" \
  'SOUNIO_RESIDENT_AUTHORITY_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output valid-response "$valid_response" \
  'SOUNIO_RESIDENT_AUTHORITY_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output valid-stop "$valid_stop" \
  'SOUNIO_RESIDENT_AUTHORITY_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN'
assert_output replay "$replayed_request" \
  'SOUNIO_RESIDENT_AUTHORITY_DENY code=442 reason=sequence-invalid stage=SEMANTICS_FROZEN'
assert_output skip "$skipped_request" \
  'SOUNIO_RESIDENT_AUTHORITY_DENY code=442 reason=sequence-invalid stage=SEMANTICS_FROZEN'
assert_output orphan "$orphan_request" \
  'SOUNIO_RESIDENT_AUTHORITY_DENY code=440 reason=parent-not-frozen stage=SEMANTICS_FROZEN'
assert_output correlation "$uncorrelated_response" \
  'SOUNIO_RESIDENT_AUTHORITY_DENY code=443 reason=binding-incomplete stage=SEMANTICS_FROZEN'
assert_output deadline "$no_deadline" \
  'SOUNIO_RESIDENT_AUTHORITY_DENY code=444 reason=transport-unhealthy stage=SEMANTICS_FROZEN'
assert_output poisoned "$poisoned_generation" \
  'SOUNIO_RESIDENT_AUTHORITY_DENY code=445 reason=generation-poisoned stage=SEMANTICS_FROZEN'
assert_output malformed '9024 3' \
  'SOUNIO_RESIDENT_AUTHORITY_DENY code=424 reason=malformed-frame stage=INVALID'

sabotage() {
  local label="$1" rule="$2" frame="$3"
  local sabotaged_module="$TEST_ROOT/$label.sio"
  local combined="$TEST_ROOT/$label-combined.sio"
  local sabotaged_runtime="$TEST_ROOT/$label-runtime"
  grep -Fqx "$rule" "$MODULE" || fail "$label rule is absent or changed"
  grep -Fvx "$rule" "$MODULE" > "$sabotaged_module"
  sed -n '1,$p' "$sabotaged_module" "$ENTRYPOINT" > "$combined"
  SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" compile "$combined" \
    -o "$sabotaged_runtime" >/dev/null
  chmod 0755 "$sabotaged_runtime"
  local actual
  actual="$(printf '%s\n' "$frame" | "$sabotaged_runtime")"
  [[ "$actual" == 'SOUNIO_RESIDENT_AUTHORITY_ALLOW code=0 reason=allow stage=SEMANTICS_FROZEN' ]] ||
    fail "$label sabotage did not admit its unchanged witness: $actual"
}

sabotage frozen-parent \
  '    if facts.parent_frozen != 1 || !loom_resident_digest_nonzero(bindings.parent_manifest_hash) { return 440 }' \
  "$orphan_request"
sabotage strict-sequence \
  '    if (facts.event_kind == 2 || facts.event_kind == 3) && facts.sequence != facts.previous_sequence + 1 { return 442 }' \
  "$replayed_request"

printf '%s\n' \
  'sounio-loom-resident-authority-selftest: PASS producer=Sounio role=SEMANTIC_AUTHORITY action=9024 cases=18 start=ALLOW request=ALLOW response=ALLOW stop=ALLOW replay=DENY442 skip=DENY442 orphan=DENY440 correlation=DENY443 no_deadline=DENY444 poisoned=DENY445 malformed=DENY424 causal_sabotage=ALLOWx2'
