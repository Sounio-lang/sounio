#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-result-handle.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-exec-result-handle-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

RUNTIME_ONE="$TEST_ROOT/runtime-one"
RUNTIME_TWO="$TEST_ROOT/runtime-two"
for output in "$RUNTIME_ONE" "$RUNTIME_TWO"; do
  SOUNIO_LOOM_EXEC_RESULT_HANDLE_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_handle_fixture.sh" >/dev/null
done
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two Sounio builds differ'

# Masks are a closed V1 projection: each bit selects the exact Sounio-owned
# digest/condition or an explicit absent value. No material expected result is
# encoded here.
publish='9033 268435439 262143 1'
resolve='9033 335544311 262143 1'
command_mismatch='9033 268433391 262143 1'

publish_result="$(printf '%s\n' "$publish" | "$RUNTIME_ONE")"
resolve_result="$(printf '%s\n' "$resolve" | "$RUNTIME_ONE")"
set +e
mismatch_result="$(printf '%s\n' "$command_mismatch" | "$RUNTIME_ONE")"
mismatch_code=$?
malformed_result="$(printf '9033 3\n' | "$RUNTIME_ONE")"
malformed_code=$?
set -e
handle='loom-result-v1:113b0f7d2a1f7adc5b68a92ef37be3689accfe4bbb00cb4e8c15fc3ab7b70013:1:5805e6579b6420ba0dd693d385715943955d0e69e657f44e94e23d20a20d27d1'
handle_fields='LOOM_EXEC_RESULT_HANDLE_FIELDS_V1
113b0f7d2a1f7adc5b68a92ef37be3689accfe4bbb00cb4e8c15fc3ab7b70013
1
5805e6579b6420ba0dd693d385715943955d0e69e657f44e94e23d20a20d27d1'
[[ "$publish_result" == "SOUNIO_EXEC_RESULT_HANDLE PUBLISH semantic_authority=Sounio action=9033
$handle_fields" ]] ||
  fail "publish fixture diverged: $publish_result"
[[ "$resolve_result" == "SOUNIO_EXEC_RESULT_HANDLE RESOLVE semantic_authority=Sounio action=9033
$handle_fields" ]] ||
  fail "resolve fixture diverged: $resolve_result"
[[ $mismatch_code -eq 42 && "$mismatch_result" == 'SOUNIO_EXEC_RESULT_HANDLE DENY534 semantic_authority=Sounio action=9033' ]] ||
  fail "command-binding sabotage diverged: $mismatch_result"
[[ $malformed_code -eq 42 && "$malformed_result" == 'SOUNIO_EXEC_RESULT_HANDLE DENY424 reason=malformed-frame semantic_authority=Sounio action=9033' ]] ||
  fail "malformed-frame control diverged: $malformed_result"

MUTANT_MODULE="$TEST_ROOT/mutant.sio"
sed 's/if !loom_exec_result_command_binding_rule(/if false \&\& loom_exec_result_command_binding_rule(/' \
  "$ROOT_DIR/stdlib/coordination/loom_exec_result_handle_authority.sio" > "$MUTANT_MODULE"
[[ "$(cmp -l "$ROOT_DIR/stdlib/coordination/loom_exec_result_handle_authority.sio" "$MUTANT_MODULE" 2>/dev/null | wc -l)" -gt 0 ]] ||
  fail 'causal mutation did not change the Sounio source'
MUTANT_RUNTIME="$TEST_ROOT/mutant-runtime"
SOUNIO_LOOM_EXEC_RESULT_HANDLE_MODULE="$MUTANT_MODULE" \
SOUNIO_LOOM_EXEC_RESULT_HANDLE_OUTPUT="$MUTANT_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_handle_fixture.sh" >/dev/null || true
[[ -x "$MUTANT_RUNTIME" ]] || fail 'mutant Sounio runtime did not build'
mutant_result="$(printf '%s\n' "$command_mismatch" | "$MUTANT_RUNTIME")"
[[ "$mutant_result" == "SOUNIO_EXEC_RESULT_HANDLE PUBLISH semantic_authority=Sounio action=9033
$handle_fields" ]] ||
  fail "load-bearing mutation did not admit the unchanged witness: $mutant_result"

oracle_executed="$TEST_ROOT/oracle-executed"
for name in python3 rustc; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$oracle_executed" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
PATH="$TEST_ROOT:$PATH" printf '%s\n' "$publish" | "$RUNTIME_ONE" >/dev/null
[[ ! -e "$oracle_executed" ]] || fail 'a prohibited oracle executed'
dependencies="$(ldd "$RUNTIME_ONE" 2>&1 || true)"
printf '%s\n' "$dependencies" | grep -Eqi 'python|rust' &&
  fail 'Sounio runtime has a prohibited dependency'

printf 'sounio-loom-exec-result-handle-selftest: PASS semantic_authority=Sounio action=9033 stage=SOUNIO_EXECUTABLE cases=16 treatment=PUBLISH+RESOLVE command_mismatch=DENY534 malformed=DENY424 causal_sabotage=PASS handle=%s expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false runtime_dependencies=clean source_sha256=%s executable_sha256=%s material_execution=false exec_cell_attached=false result_store_attached=false provider_hook_switched=false production_activation=false parity_open=false claim_ready=false\n' \
  "$handle" \
  "$(sha256sum "$ROOT_DIR/stdlib/coordination/loom_exec_result_handle_authority.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME_ONE" | cut -d ' ' -f 1)"
