#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-intent-envelope.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-exec-intent-envelope-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

RUNTIME_ONE="$TEST_ROOT/runtime-one"
RUNTIME_TWO="$TEST_ROOT/runtime-two"
for output in "$RUNTIME_ONE" "$RUNTIME_TWO"; do
  SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_intent_envelope_fixture.sh" >/dev/null
done
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio builds differ'

project='9034 4194303 264191'
command_mismatch='9034 4128767 264191'
fields='LOOM_EXEC_INTENT_ENVELOPE_FIELDS_V1
LOOM_EXEC_INTENT/1
113b0f7d2a1f7adc5b68a92ef37be3689accfe4bbb00cb4e8c15fc3ab7b70013
07ffb41877176a60b04949dd78c91313954dd759593f20677a1b2fcfeea31a60'
project_result="$(printf '%s\n' "$project" | "$RUNTIME_ONE")"
set +e
mismatch_result="$(printf '%s\n' "$command_mismatch" | "$RUNTIME_ONE")"
mismatch_code=$?
malformed_result="$(printf '9034 3\n' | "$RUNTIME_ONE")"
malformed_code=$?
set -e
[[ "$project_result" == "SOUNIO_EXEC_INTENT_ENVELOPE PROJECT semantic_authority=Sounio action=9034
$fields" ]] || fail "projection fixture diverged: $project_result"
[[ $mismatch_code -eq 42 && "$mismatch_result" == \
   'SOUNIO_EXEC_INTENT_ENVELOPE DENY555 semantic_authority=Sounio action=9034' ]] ||
  fail "command-binding sabotage diverged: $mismatch_result"
[[ $malformed_code -eq 42 && "$malformed_result" == \
   'SOUNIO_EXEC_INTENT_ENVELOPE DENY424 reason=malformed-frame semantic_authority=Sounio action=9034' ]] ||
  fail "malformed-frame control diverged: $malformed_result"

MUTANT_MODULE="$TEST_ROOT/mutant.sio"
sed 's/if !loom_exec_intent_command_binding_rule(/if false \&\& loom_exec_intent_command_binding_rule(/' \
  "$ROOT_DIR/stdlib/coordination/loom_exec_intent_envelope_authority.sio" > "$MUTANT_MODULE"
[[ "$(cmp -l "$ROOT_DIR/stdlib/coordination/loom_exec_intent_envelope_authority.sio" "$MUTANT_MODULE" 2>/dev/null | wc -l)" -gt 0 ]] ||
  fail 'causal mutation did not change the Sounio source'
MUTANT_RUNTIME="$TEST_ROOT/mutant-runtime"
SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_MODULE="$MUTANT_MODULE" \
SOUNIO_LOOM_EXEC_INTENT_ENVELOPE_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_intent_envelope_fixture.sh" >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'mutant Sounio runtime did not build'
mutant_result="$(printf '%s\n' "$command_mismatch" | "$MUTANT_RUNTIME")"
[[ "$mutant_result" == "SOUNIO_EXEC_INTENT_ENVELOPE PROJECT semantic_authority=Sounio action=9034
$fields" ]] ||
  fail "load-bearing mutation did not admit the unchanged witness: $mutant_result"

oracle_executed="$TEST_ROOT/oracle-executed"
for name in python3 rustc; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$oracle_executed" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
PATH="$TEST_ROOT:$PATH" printf '%s\n' "$project" | "$RUNTIME_ONE" >/dev/null
[[ ! -e "$oracle_executed" ]] || fail 'a prohibited oracle executed'
dependencies="$(ldd "$RUNTIME_ONE" 2>&1 || true)"
printf '%s\n' "$dependencies" | grep -Eqi 'python|rust' &&
  fail 'Sounio runtime has a prohibited dependency'

printf 'sounio-loom-exec-intent-envelope-selftest: PASS semantic_authority=Sounio action=9034 stage=SOUNIO_EXECUTABLE cases=12 treatment=PROJECT command_mismatch=DENY555 malformed=DENY424 causal_sabotage=PASS schema=LOOM_EXEC_INTENT/1 semantic_event_sha256=113b0f7d2a1f7adc5b68a92ef37be3689accfe4bbb00cb4e8c15fc3ab7b70013 command_sha256=07ffb41877176a60b04949dd78c91313954dd759593f20677a1b2fcfeea31a60 raw_event_separate=true expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false runtime_dependencies=clean source_sha256=%s executable_sha256=%s ocaml_projection_attached=false provider_lifecycle_attached=false arbitrary_command_projection=false exec_attached=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_exec_intent_envelope_authority.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME_ONE" | cut -d ' ' -f 1)"
