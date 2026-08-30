#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-causal-workflow-journal.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT
MANIFEST="$ROOT_DIR/tools/loom/causal_workflow_journal.runtime.v1"

fail() {
  printf 'sounio-loom-causal-workflow-journal-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

manifest_value() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

expect_value() {
  local key="$1" expected="$2"
  [[ "$(manifest_value "$key")" == "$expected" ]] ||
    fail "manifest field $key diverged"
}

expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" && ! -L "$path" ]] || fail "$path is absent or linked"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "$path hash drifted"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'runtime manifest is absent or linked'
expect_value schema loom-causal-workflow-journal-runtime-v1
expect_value stage OCAML_DURABLE_JOURNAL
expect_value semantic_authority Sounio
expect_value semantic_action 9037
expect_value semantic_role SEMANTIC_AUTHORITY
expect_value operational_language OCaml
expect_value operational_role EFFECT_PARITY
expect_value run_ticket_is_bearer false
expect_value run_ticket_is_execution_authority false
expect_value launch_authority action-9030
expect_value hostguardian_pidfd_attached false
expect_value dynamic_user_workflow_attached false
expect_value material_execution false
expect_value pod_loss_measured false
expect_value production_activation false
expect_value parity_open false
expect_value claim_ready false
for key in semantics_manifest module fixture build_script selftest evidence; do
  expect_hash "$ROOT_DIR/$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done
for key in ocamlopt ocamlfind cryptokit_cmxa cryptokit_archive; do
  expect_hash "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

SOUNIO_RUNTIME="$TEST_ROOT/sounio-authority"
SOUNIO_LOOM_CAUSAL_WORKFLOW_OUTPUT="$SOUNIO_RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_kernel_fixture.sh" >/dev/null

RUNTIME_ONE="$TEST_ROOT/journal-runtime-one"
RUNTIME_TWO="$TEST_ROOT/journal-runtime-two"
for output in "$RUNTIME_ONE" "$RUNTIME_TWO"; do
  SOUNIO_LOOM_CAUSAL_WORKFLOW_JOURNAL_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_loom_causal_workflow_journal_fixture.sh" >/dev/null
done
cmp "$RUNTIME_ONE" "$RUNTIME_TWO" || fail 'two OCaml builds differ'

ORACLE_EXECUTED="$TEST_ROOT/oracle-executed"
for name in python python3 rustc cargo; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done

STATE_ROOT="$TEST_ROOT/state"
PHASE_A_OUTPUT="$TEST_ROOT/phase-a.out"
SOUNIO_LOOM_CAUSAL_WORKFLOW_RUNTIME="$SOUNIO_RUNTIME" \
PATH="$TEST_ROOT:$PATH" \
  "$RUNTIME_ONE" phase-a-wait "$ROOT_DIR" "$STATE_ROOT" >"$PHASE_A_OUTPUT" 2>&1 &
controller_pid=$!
ready=false
for _ in $(seq 1 200); do
  if grep -Fq 'PHASE_A_READY phase=COMPILED_CLOSED sequence=4 compile_count=1 ticket_count=0 launch_count=0' "$PHASE_A_OUTPUT"; then
    ready=true
    break
  fi
  kill -0 "$controller_pid" 2>/dev/null || break
  sleep 0.01
done
[[ "$ready" == true ]] || fail "phase A did not become durable: $(cat "$PHASE_A_OUTPUT")"
kill -9 "$controller_pid"
set +e
wait "$controller_pid" 2>/dev/null
kill_code=$?
set -e
[[ $kill_code -eq 137 ]] || fail "controller SIGKILL returned $kill_code"

set +e
wrong_recovery="$(SOUNIO_LOOM_CAUSAL_WORKFLOW_RUNTIME="$SOUNIO_RUNTIME" \
  PATH="$TEST_ROOT:$PATH" "$RUNTIME_TWO" wrong-recovery "$ROOT_DIR" "$STATE_ROOT" 2>&1)"
wrong_code=$?
set -e
[[ $wrong_code -eq 70 && "$wrong_recovery" == \
   'CAUSAL_WORKFLOW_FIXTURE_ERROR causal-workflow-recovery-custody-mismatch' ]] ||
  fail "wrong Guardian recovery diverged: $wrong_recovery"

before_recovery="$(SOUNIO_LOOM_CAUSAL_WORKFLOW_RUNTIME="$SOUNIO_RUNTIME" \
  PATH="$TEST_ROOT:$PATH" "$RUNTIME_TWO" status "$ROOT_DIR" "$STATE_ROOT")"
[[ "$before_recovery" == STATUS\ phase=COMPILED_CLOSED\ sequence=4\ compile_count=1\ ticket_count=0\ launch_count=0* ]] ||
  fail "refused recovery changed the journal: $before_recovery"

phase_b="$(SOUNIO_LOOM_CAUSAL_WORKFLOW_RUNTIME="$SOUNIO_RUNTIME" \
  PATH="$TEST_ROOT:$PATH" "$RUNTIME_TWO" phase-b "$ROOT_DIR" "$STATE_ROOT")"
[[ "$phase_b" == PHASE_B_COMPLETE\ recompile=REFUSED\ duplicate_ticket=REFUSED\ duplicate_launch=REFUSED\ phase=ATTESTED_CLOSED\ sequence=12\ compile_count=1\ ticket_count=1\ launch_count=1* ]] ||
  fail "phase B diverged: $phase_b"

status="$(SOUNIO_LOOM_CAUSAL_WORKFLOW_RUNTIME="$SOUNIO_RUNTIME" \
  PATH="$TEST_ROOT:$PATH" "$RUNTIME_ONE" status "$ROOT_DIR" "$STATE_ROOT")"
[[ "$status" == STATUS\ phase=ATTESTED_CLOSED\ sequence=12\ compile_count=1\ ticket_count=1\ launch_count=1* ]] ||
  fail "final replay diverged: $status"

digest_of() {
  printf '%s' "$1" | sha256sum | cut -d ' ' -f 1
}

material() {
  local command="$1"
  shift
  SOUNIO_LOOM_CAUSAL_WORKFLOW_RUNTIME="$SOUNIO_RUNTIME" \
    PATH="$TEST_ROOT:$PATH" "$RUNTIME_TWO" "$command" "$ROOT_DIR" \
    "$MATERIAL_STATE_ROOT" "$MATERIAL_WORKFLOW_ID" "$@"
}

MATERIAL_STATE_ROOT="$TEST_ROOT/material-state"
MATERIAL_WORKFLOW_ID=material-controller-loss-at-running
WORKFLOW_GENERATION="$(digest_of workflow-generation)"
GUARDIAN_GENERATION="$(digest_of guardian-generation)"
JOURNAL_ID="$(digest_of journal-id)"
STORE_ID="$(digest_of store-id)"
CONTROLLER_ONE="$(digest_of controller-one)"
CONTROLLER_TWO="$(digest_of controller-two)"
SOURCE_SHA256="$(digest_of source)"
COMPILE_RECEIPT="$(digest_of compile-receipt)"
ARTIFACT_RECORD="$(digest_of artifact-record)"
ARTIFACT_HANDLE="$(digest_of artifact-handle)"
RUN_TICKET="$(digest_of run-ticket)"
RUN_GRANT="$(digest_of run-grant)"
RUN_GENERATION="$(digest_of run-generation)"
START_RECEIPT="$(digest_of start-receipt)"
RUN_PID_IDENTITY="$(digest_of run-pid-identity)"
EMPTY_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
RESULT_RECORD="$(digest_of result-record)"
RESULT_HANDLE="$(digest_of result-handle)"
ATTEST_RECORD="$(digest_of attestation-record)"
ATTEST_HANDLE="$(digest_of attestation-handle)"

material material-open "$WORKFLOW_GENERATION" "$GUARDIAN_GENERATION" \
  "$JOURNAL_ID" "$STORE_ID" "$CONTROLLER_ONE" "$SOURCE_SHA256" >/dev/null
material material-arm-compile >/dev/null
material material-start-compile >/dev/null
material material-close-compile "$COMPILE_RECEIPT" "$ARTIFACT_RECORD" \
  "$ARTIFACT_HANDLE" >/dev/null
material material-arm-run "$RUN_TICKET" "$RUN_GRANT" "$RUN_GENERATION" >/dev/null

MATERIAL_RUNNING_OUTPUT="$TEST_ROOT/material-running.out"
SOUNIO_LOOM_CAUSAL_WORKFLOW_RUNTIME="$SOUNIO_RUNTIME" \
PATH="$TEST_ROOT:$PATH" "$RUNTIME_TWO" material-mark-running-wait "$ROOT_DIR" \
  "$MATERIAL_STATE_ROOT" "$MATERIAL_WORKFLOW_ID" "$START_RECEIPT" \
  "$RUN_PID_IDENTITY" >"$MATERIAL_RUNNING_OUTPUT" 2>&1 &
material_controller_pid=$!
material_ready=false
for _ in $(seq 1 200); do
  if grep -Fq 'MATERIAL_RUNNING phase=RUNNING sequence=6 compile_count=1 ticket_count=1 launch_count=1' \
      "$MATERIAL_RUNNING_OUTPUT"; then
    material_ready=true
    break
  fi
  kill -0 "$material_controller_pid" 2>/dev/null || break
  sleep 0.01
done
[[ "$material_ready" == true ]] ||
  fail "material controller did not durably enter RUNNING: $(cat "$MATERIAL_RUNNING_OUTPUT")"
kill -9 "$material_controller_pid"
set +e
wait "$material_controller_pid" 2>/dev/null
material_kill_code=$?
set -e
[[ $material_kill_code -eq 137 ]] ||
  fail "material controller SIGKILL returned $material_kill_code"

material material-recover "$CONTROLLER_TWO" "$GUARDIAN_GENERATION" \
  "$JOURNAL_ID" "$STORE_ID" >/dev/null
material_running_status="$(material material-status)"
[[ "$material_running_status" == MATERIAL_STATUS\ phase=RUNNING\ sequence=7\ compile_count=1\ ticket_count=1\ launch_count=1* ]] ||
  fail "RUNNING recovery replay diverged: $material_running_status"

set +e
material_recompile="$(material material-recompile 2>&1)"
material_recompile_code=$?
material_relaunch="$(material material-mark-running "$START_RECEIPT" "$RUN_PID_IDENTITY" 2>&1)"
material_relaunch_code=$?
set -e
[[ $material_recompile_code -eq 70 ]] ||
  fail "material recompile was not refused: $material_recompile"
[[ $material_relaunch_code -eq 70 ]] ||
  fail "material relaunch was not refused: $material_relaunch"

material material-record-result 0 "$EMPTY_SHA256" "$EMPTY_SHA256" \
  "$RESULT_RECORD" "$RESULT_HANDLE" >/dev/null
set +e
incomplete_extinction="$(material material-close-run true true false true true 2>&1)"
incomplete_extinction_code=$?
set -e
[[ $incomplete_extinction_code -eq 70 ]] ||
  fail "incomplete extinction was not refused: $incomplete_extinction"
material material-close-run true true true true true >/dev/null
material material-arm-attest >/dev/null
material material-start-attest >/dev/null
material material-close-attest "$ATTEST_RECORD" "$ATTEST_HANDLE" >/dev/null
material_final_status="$(material material-status)"
[[ "$material_final_status" == MATERIAL_STATUS\ phase=ATTESTED_CLOSED\ sequence=12\ compile_count=1\ ticket_count=1\ launch_count=1* ]] ||
  fail "material recovery completion diverged: $material_final_status"

mapfile -t journals < <(find "$STATE_ROOT/causal-workflows" -maxdepth 1 -type f -name '*.journal' -print)
[[ ${#journals[@]} -eq 1 ]] || fail 'journal cardinality diverged'
JOURNAL="${journals[0]}"
[[ "$(wc -l < "$JOURNAL")" -eq 12 ]] || fail 'journal sequence count diverged'
[[ "$(stat -c '%a' "$JOURNAL")" == 600 ]] || fail 'journal mode is not 0600'
[[ "$(stat -c '%a' "$STATE_ROOT/causal-workflows")" == 700 ]] ||
  fail 'workflow store mode is not 0700'

TAMPER_ROOT="$TEST_ROOT/tamper-state"
cp -a "$STATE_ROOT" "$TAMPER_ROOT"
TAMPER_JOURNAL="$(find "$TAMPER_ROOT/causal-workflows" -maxdepth 1 -type f -name '*.journal' -print)"
printf 'x' >> "$TAMPER_JOURNAL"
set +e
tamper="$(SOUNIO_LOOM_CAUSAL_WORKFLOW_RUNTIME="$SOUNIO_RUNTIME" \
  PATH="$TEST_ROOT:$PATH" "$RUNTIME_ONE" status "$ROOT_DIR" "$TAMPER_ROOT" 2>&1)"
tamper_code=$?
set -e
[[ $tamper_code -eq 70 && "$tamper" == CAUSAL_WORKFLOW_FIXTURE_ERROR\ causal-workflow-journal-* ]] ||
  fail "journal tamper was not refused: $tamper"

[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a prohibited oracle executed'
DEPENDENCIES="$(ldd "$RUNTIME_ONE" 2>&1 || true)"
printf '%s\n' "$DEPENDENCIES" | grep -Eqi 'python|rust' &&
  fail 'OCaml journal runtime has a prohibited dependency'

result="$(printf 'sounio-loom-causal-workflow-journal-selftest: PASS semantic_authority=Sounio action=9037 operational_language=OCaml operational_role=EFFECT_PARITY controller_sigkill=true controller_sigkill_at=RUNNING controller_recovery_at_running=true tmux_used=false journal_replay_after_sigkill=true state_before=RUNNING state_after=ATTESTED_CLOSED sequence=12 compile_count=1 ticket_count=1 launch_count=1 recompile=REFUSED duplicate_ticket=REFUSED duplicate_launch=REFUSED incomplete_extinction=REFUSED wrong_guardian=REFUSED journal_tamper=REFUSED journal_mode=0600 store_mode=0700 hash_chain=verified fsync_before_reply=true sounio_rechecked_each_transition=true run_ticket_bearer=false run_ticket_execution_authority=false launch_authority=action-9030 python_executed=false rust_executed=false runtime_dependencies=clean module_sha256=%s fixture_sha256=%s executable_sha256=%s hostguardian_pidfd_attached=false dynamic_user_workflow_attached=false material_execution=false pod_loss_measured=false production_activation=false parity_open=false claim_ready=false' \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_causal_workflow.ml" | cut -d ' ' -f 1)" \
  "$(sha256sum "$ROOT_DIR/tools/loom/causal_workflow_journal_fixture.ml" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME_ONE" | cut -d ' ' -f 1)")"
[[ "$result" == "$(manifest_value result)" ]] || fail 'runtime result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] || fail 'runtime result hash drifted'
printf '%s\n' "$result" | cmp - "$ROOT_DIR/$(manifest_value evidence_path)" ||
  fail 'runtime evidence is not the exact result'
printf '%s\n' "$result"
