#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-result-record-projection.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-exec-result-record-projection-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

digest() {
  printf '%s' "$1" | sha256sum | cut -d ' ' -f 1
}

bash "$ROOT_DIR/scripts/ci/sounio_loom_exec_result_record_freeze_selftest.sh" >/dev/null
dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_operation_catalog_fixture.sh" >/dev/null
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_record_fixture.sh" >/dev/null

LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
CATALOG_RUNTIME="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-exec-operation-catalog"
RECORD_RUNTIME="$ROOT_DIR/tools/loom/_build/default/src/sounio-loom-exec-result-record"
MANIFEST="$ROOT_DIR/tools/loom/exec_result_record.freeze.v1"
SOURCE="tests/verify-ir/call_b.sio"
SOURCE_SHA256="$(sha256sum "$ROOT_DIR/$SOURCE" | cut -d ' ' -f 1)"
OUTPUT="$TEST_ROOT/loom-sounio-check-${SOURCE_SHA256:0:16}.elf"
EVENT='6017e4c6e745560696f78836f9cc07ec71a9106f13ad1bfdb16d7e342f0840a9'
GENERATION="$(digest projection-generation)"
PRINCIPAL="$(digest projection-principal)"
DESCRIPTOR="$(digest projection-descriptor)"
GRANT="$(digest projection-grant)"

run_issue() {
  SOUNIO_LOOM_HOOK_TEST_MODE=1 "$LOOM" exec-result-record-probe \
    --root "$ROOT_DIR" --mode issue --source "$SOURCE" --output "$OUTPUT" \
    --event "$EVENT" --generation "$GENERATION" --principal "$PRINCIPAL" \
    --descriptor-binding "$DESCRIPTOR" --grant-receipt "$GRANT"
}

FIRST="$(run_issue)"
[[ "$FIRST" == *'semantic_authority=Sounio action=9036 operational_kernel=OCaml operation=sounio-check'* &&
   "$FIRST" == *"event_sha256=$EVENT generation_sha256=$GENERATION source_sha256=$SOURCE_SHA256"* &&
   "$FIRST" == *'artifact_sha256=eff2ac0ef28b34d6cc4f008cfb08a30ba18a0874c8654c06a3c62ec2f48a249c'* &&
   "$FIRST" == *'handle_is_bearer=false handle_is_execution_authority=false artifact_executed=false ocaml_record_projection_attached=true dynamic_user_host_attached=false'* &&
   "$FIRST" == *"principal_sha256=$PRINCIPAL"* &&
   "$FIRST" == *"descriptor_binding_sha256=$DESCRIPTOR"* &&
   "$FIRST" == *"grant_receipt_sha256=$GRANT"* ]] ||
  fail "record projection diverged: $FIRST"
RECORD_SHA256="$(printf '%s\n' "$FIRST" | sed -n '1s/.* record_sha256=\([^ ]*\) .*/\1/p')"
HANDLE="$(printf '%s\n' "$FIRST" | sed -n '1s/.* handle=\([^ ]*\) .*/\1/p')"
[[ "$RECORD_SHA256" =~ ^[0-9a-f]{64}$ &&
   "$HANDLE" == "loom-result-v2:$EVENT:$GENERATION:$RECORD_SHA256" ]] ||
  fail 'record handle was not derived from event, generation, and record'

rm -f "$OUTPUT"
SECOND="$(run_issue)"
SECOND_RECORD_SHA256="$(printf '%s\n' "$SECOND" | sed -n '1s/.* record_sha256=\([^ ]*\) .*/\1/p')"
[[ "$SECOND_RECORD_SHA256" == "$RECORD_SHA256" ]] ||
  fail 'same execution measurements produced a different canonical record'

control="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 "$LOOM" exec-result-record-probe \
  --root "$ROOT_DIR" --mode artifact-binding)"
[[ "$control" == *'action=9036 mode=artifact-binding'* &&
   "$control" == *'decision=SOUNIO_EXEC_RESULT_RECORD DENY577'* &&
   "$control" == *'control_refused=true material_mutation=false'* ]] ||
  fail "artifact-binding control diverged: $control"

mkdir -p "$TEST_ROOT/wrong-event"
WRONG_OUTPUT="$TEST_ROOT/wrong-event/loom-sounio-check-${SOURCE_SHA256:0:16}.elf"
set +e
wrong_event="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 "$LOOM" exec-result-record-probe \
  --root "$ROOT_DIR" --mode issue --source "$SOURCE" --output "$WRONG_OUTPUT" \
  --event "$(digest wrong-event)" --generation "$GENERATION" \
  --principal "$PRINCIPAL" --descriptor-binding "$DESCRIPTOR" \
  --grant-receipt "$GRANT" 2>&1)"
wrong_event_code=$?
set -e
[[ $wrong_event_code -eq 1 && "$wrong_event" == \
   'error: exec-result-record-catalog-binding-mismatch' ]] ||
  fail "wrong semantic event was not refused: $wrong_event"

TAMPERED_MANIFEST="$TEST_ROOT/tampered.freeze.v1"
cp "$MANIFEST" "$TAMPERED_MANIFEST"
printf 'tampered=true\n' >> "$TAMPERED_MANIFEST"
set +e
manifest_tamper="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_RESULT_RECORD_MANIFEST="$TAMPERED_MANIFEST" \
  "$LOOM" exec-result-record-probe --root "$ROOT_DIR" \
  --mode artifact-binding 2>&1)"
manifest_tamper_code=$?
set -e
[[ $manifest_tamper_code -eq 1 && "$manifest_tamper" == \
   'error: exec-result-record-manifest-hash-mismatch' ]] ||
  fail "manifest tamper did not fail closed: $manifest_tamper"

TAMPERED_RUNTIME="$TEST_ROOT/tampered-runtime"
cp "$RECORD_RUNTIME" "$TAMPERED_RUNTIME"
printf 'tamper' >> "$TAMPERED_RUNTIME"
chmod 0755 "$TAMPERED_RUNTIME"
set +e
runtime_tamper="$(SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  SOUNIO_LOOM_EXEC_RESULT_RECORD_RUNTIME="$TAMPERED_RUNTIME" \
  "$LOOM" exec-result-record-probe --root "$ROOT_DIR" \
  --mode artifact-binding 2>&1)"
runtime_tamper_code=$?
set -e
[[ $runtime_tamper_code -eq 1 && "$runtime_tamper" == \
   'error: exec-result-record-runtime-hash-mismatch' ]] ||
  fail "runtime tamper did not fail closed: $runtime_tamper"

ORACLE_EXECUTED="$TEST_ROOT/oracle-executed"
for name in python3 rustc; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
PATH="$TEST_ROOT:$PATH" SOUNIO_LOOM_HOOK_TEST_MODE=1 \
  "$LOOM" exec-result-record-probe --root "$ROOT_DIR" \
  --mode artifact-binding >/dev/null
[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a prohibited oracle executed'
DEPENDENCIES="$(ldd "$LOOM" 2>&1 || true; ldd "$CATALOG_RUNTIME" 2>&1 || true; ldd "$RECORD_RUNTIME" 2>&1 || true)"
printf '%s\n' "$DEPENDENCIES" | grep -Eqi 'python|rust' &&
  fail 'record-projection runtime has a prohibited dependency'

RESULT="$(printf 'sounio-loom-exec-result-record-projection-selftest: PASS semantic_authority=Sounio action=9036 stage=SEMANTICS_FROZEN operational_kernel=OCaml operation=sounio-check source_sha256=%s artifact_sha256=eff2ac0ef28b34d6cc4f008cfb08a30ba18a0874c8654c06a3c62ec2f48a249c canonical_record_replay=stable handle_recipe=event+generation+record artifact_binding=DENY577 wrong_event=REFUSED manifest_tamper=REFUSED runtime_tamper=REFUSED handle_is_bearer=false handle_is_execution_authority=false artifact_executed=false python_executed=false rust_executed=false runtime_dependencies=clean ocaml_record_projection_attached=true dynamic_user_host_attached=false provider_result_returned=false production_activation=false operational_source_sha256=%s loom_executable_sha256=%s' \
  "$SOURCE_SHA256" \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_exec_result_record.ml" | cut -d ' ' -f 1)" \
  "$(sha256sum "$LOOM" | cut -d ' ' -f 1)")"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-exec-result-record-projection-v1-20260830.txt"
[[ "$(cat "$EVIDENCE")" == "$RESULT" ]] || fail 'checked-in evidence drifted'
printf '%s\n' "$RESULT"
