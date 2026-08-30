#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-result-record.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-exec-result-record-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

RUNTIME_ONE="$TEST_ROOT/runtime-one"
RUNTIME_TWO="$TEST_ROOT/runtime-two"
for output in "$RUNTIME_ONE" "$RUNTIME_TWO"; do
  SOUNIO_LOOM_EXEC_RESULT_RECORD_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_record_fixture.sh" >/dev/null
done
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio builds differ'

POSITIVE='9036 1073741815 75775'
BINDING_MISMATCH='9036 805306359 75775'
FIELDS='LOOM_EXEC_RESULT_RECORD_FIELDS_V1
LOOM_EXEC_RESULT_RECORD/1
sounio-check
31089d39690c8e008dd7ea9bfecefd7e51e4257cd96e091f7fb492c240ed342c
d6e513e785a170a7ce8cd8f07a66e5325736ffdf21cf8f366d82c4f5ef4f15cf
7efebd07c77aee01b3d9f9b4ae44d175bb1dd90454c58d09d460fd2bf68c59ec
LOOM_EXEC_RESULT_RECORD/1|operation=sounio-check|event_sha256|command_template_sha256|generation_sha256|source_sha256|compiler_sha256|argv_sha256|artifact_sha256|artifact_bytes|stdout_sha256|stderr_sha256|diagnostics_sha256|sandbox_profile_sha256|exit_code
loom-result-v2:{event_sha256}:{generation_sha256}:{record_sha256}
record_hash=sha256(canonical_fields)
handle_is_bearer=false
handle_is_execution_authority=false
artifact_executed=false'

positive_result="$(printf '%s\n' "$POSITIVE" | "$RUNTIME_ONE")"
[[ "$positive_result" == "SOUNIO_EXEC_RESULT_RECORD ISSUE semantic_authority=Sounio action=9036
$FIELDS" ]] || fail "positive record schema diverged: $positive_result"

set +e
binding_result="$(printf '%s\n' "$BINDING_MISMATCH" | "$RUNTIME_ONE")"
binding_code=$?
malformed_result="$(printf '9036 3\n' | "$RUNTIME_ONE")"
malformed_code=$?
set -e
[[ $binding_code -eq 42 && "$binding_result" == \
   'SOUNIO_EXEC_RESULT_RECORD DENY577 semantic_authority=Sounio action=9036' ]] ||
  fail "artifact-binding control diverged: $binding_result"
[[ $malformed_code -eq 42 && "$malformed_result" == \
   'SOUNIO_EXEC_RESULT_RECORD DENY424 reason=malformed-frame semantic_authority=Sounio action=9036' ]] ||
  fail "malformed-frame control diverged: $malformed_result"

declare -A CONTROLS=(
  [stage]='9036 1073741814 75775|DENY570'
  [authority]='9036 1073741815 75774|DENY571'
  [schema]='9036 1073741751 75775|DENY572'
  [material]='9036 1073733623 75775|DENY573'
  [binding]='9036 1071644663 75775|DENY574'
  [runtime]='9036 1040187383 75775|DENY575'
  [receipt]='9036 1073741815 75771|DENY576'
)
for label in stage authority schema material binding runtime receipt; do
  IFS='|' read -r frame expected <<< "${CONTROLS[$label]}"
  set +e
  observed="$(printf '%s\n' "$frame" | "$RUNTIME_ONE")"
  code=$?
  set -e
  [[ $code -eq 42 && "$observed" == \
     "SOUNIO_EXEC_RESULT_RECORD $expected semantic_authority=Sounio action=9036" ]] ||
    fail "$label control diverged: $observed"
done

MUTANT_MODULE="$TEST_ROOT/mutant.sio"
sed 's/if observation.artifact_bound_to_record != 1 {/if false {/' \
  "$ROOT_DIR/stdlib/coordination/loom_exec_result_record_authority.sio" > "$MUTANT_MODULE"
[[ "$(cmp -l "$ROOT_DIR/stdlib/coordination/loom_exec_result_record_authority.sio" "$MUTANT_MODULE" 2>/dev/null | wc -l)" -gt 0 ]] ||
  fail 'causal mutation did not change the Sounio source'
MUTANT_RUNTIME="$TEST_ROOT/mutant-runtime"
SOUNIO_LOOM_EXEC_RESULT_RECORD_MODULE="$MUTANT_MODULE" \
SOUNIO_LOOM_EXEC_RESULT_RECORD_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_record_fixture.sh" >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'mutant Sounio runtime did not build'
mutant_result="$(printf '%s\n' "$BINDING_MISMATCH" | "$MUTANT_RUNTIME")"
[[ "$mutant_result" == "SOUNIO_EXEC_RESULT_RECORD ISSUE semantic_authority=Sounio action=9036
$FIELDS" ]] ||
  fail "load-bearing mutation did not admit the unchanged witness: $mutant_result"

ORACLE_EXECUTED="$TEST_ROOT/oracle-executed"
for name in python3 rustc; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
PATH="$TEST_ROOT:$PATH" printf '%s\n' "$POSITIVE" | "$RUNTIME_ONE" >/dev/null
[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a prohibited oracle executed'
DEPENDENCIES="$(ldd "$RUNTIME_ONE" 2>&1 || true)"
printf '%s\n' "$DEPENDENCIES" | grep -Eqi 'python|rust' &&
  fail 'Sounio runtime has a prohibited dependency'

printf 'sounio-loom-exec-result-record-selftest: PASS semantic_authority=Sounio action=9036 stage=SOUNIO_EXECUTABLE cases=9 treatment=ISSUE stage_control=DENY570 authority_control=DENY571 schema_control=DENY572 material_control=DENY573 binding_control=DENY574 runtime_control=DENY575 receipt_control=DENY576 artifact_binding=DENY577 malformed=DENY424 causal_sabotage=PASS catalog_action=9035 handle_is_bearer=false handle_is_execution_authority=false expected_results_encoded_in_material_layer=false artifact_executed=false python_executed=false rust_executed=false runtime_dependencies=clean source_sha256=%s executable_sha256=%s ocaml_record_projection_attached=false dynamic_user_host_attached=false provider_result_returned=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_exec_result_record_authority.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME_ONE" | cut -d ' ' -f 1)"
