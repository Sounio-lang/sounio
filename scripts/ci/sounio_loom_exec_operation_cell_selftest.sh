#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-operation-cell.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT
LOOM="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
SOURCE="tests/verify-ir/call_b.sio"
UNIT="loom-exec-operation-cell-local.service"
GENERATION="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
EVENT="6017e4c6e745560696f78836f9cc07ec71a9106f13ad1bfdb16d7e342f0840a9"
GRANT="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
CLOSE_GRANT="dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
WRONG="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
EXPECTED_ARTIFACT="eff2ac0ef28b34d6cc4f008cfb08a30ba18a0874c8654c06a3c62ec2f48a249c"

fail() {
  printf 'sounio-loom-exec-operation-cell-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field_value() {
  local line="$1" key="$2" field
  for field in $line; do
    case "$field" in
      "$key"=*) printf '%s' "${field#*=}"; return 0 ;;
    esac
  done
  return 1
}

digest() {
  [[ "$1" =~ ^[0-9a-f]{64}$ ]]
}

mkdir -p "$TEST_ROOT/fake-bin"
ORACLE_EXECUTED="$TEST_ROOT/oracle-executed"
for name in python3 rustc; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" > "$TEST_ROOT/fake-bin/$name"
  chmod 0755 "$TEST_ROOT/fake-bin/$name"
done

(cd "$ROOT_DIR/tools/loom" && dune build src/loom.exe) >/dev/null
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_operation_catalog_fixture.sh" >/dev/null
bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_record_fixture.sh" >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_exec_operation_grant_fixture_freeze_selftest.sh" >/dev/null

POSITIVE_DIR="$TEST_ROOT/positive"
mkdir -m 0700 "$POSITIVE_DIR"
coproc POSITIVE {
  exec env PATH="$TEST_ROOT/fake-bin:$PATH" \
    SOUNIO_LOOM_EXEC_OPERATION_CELL_TEST_MODE=1 \
    "$LOOM" _exec-operation-cell --root "$ROOT_DIR" --source "$SOURCE" \
      --output-dir "$POSITIVE_DIR" --unit "$UNIT" --mode test \
      2>"$TEST_ROOT/positive.err"
}
positive_pid="$POSITIVE_PID"
positive_read="${POSITIVE[0]}"
positive_write="${POSITIVE[1]}"
IFS= read -r positive_ready <&"$positive_read" || fail 'positive READY frame absent'
[[ "$positive_ready" == LOOM_EXEC_OPERATION_CELL_READY_V1* ]] ||
  fail 'positive READY frame malformed'
principal="$(field_value "$positive_ready" principal_sha256)"
descriptor="$(field_value "$positive_ready" descriptor_binding_sha256)"
ready_pid="$(field_value "$positive_ready" pid)"
ready_uid="$(field_value "$positive_ready" uid)"
ready_gid="$(field_value "$positive_ready" gid)"
digest "$principal" || fail 'self-measured principal digest malformed'
digest "$descriptor" || fail 'descriptor binding digest malformed'
[[ "$ready_pid" == "$positive_pid" && "$ready_uid" != 0 && "$ready_gid" != 0 &&
   "$(field_value "$positive_ready" host_mode)" == false &&
   "$(field_value "$positive_ready" inherited_descriptor)" == true &&
   "$(field_value "$positive_ready" material_execution)" == false ]] ||
  fail 'positive READY identity or pre-ARM state diverged'
printf 'ARM %s %s %s %s %s\n' \
  "$GENERATION" "$EVENT" "$principal" "$descriptor" "$GRANT" >&"$positive_write"
IFS= read -r positive_header <&"$positive_read" || fail 'positive result header absent'
positive_record=''
for line_number in $(seq 1 18); do
  IFS= read -r record_line <&"$positive_read" ||
    fail "positive result record line $line_number absent"
  positive_record+="$record_line"$'\n'
done
record_sha256="$(printf '%s' "$positive_record" | sha256sum | cut -d ' ' -f 1)"
printf 'CLOSE %s %s %s %s\n' \
  "$GENERATION" "$EVENT" "$record_sha256" "$CLOSE_GRANT" >&"$positive_write"
exec {positive_write}>&-
IFS= read -r positive_closed <&"$positive_read" || fail 'positive CLOSED frame absent'
printf -v positive_result '%s\n%s' "$positive_header" "$positive_record"
wait "$positive_pid" ||
  fail "positive cell refused after CLOSE: $(tr '\n' ' ' <"$TEST_ROOT/positive.err")"
[[ "$positive_result" == LOOM_EXEC_OPERATION_CELL_RESULT_V1* ]] ||
  fail 'positive result frame absent'
for expected in \
  'grant_action=9030' 'catalog_action=9035' 'result_action=9036' \
  "generation_sha256=$GENERATION" "event_sha256=$EVENT" \
  "principal_sha256=$principal" "descriptor_binding_sha256=$descriptor" \
  "grant_receipt_sha256=$GRANT" 'grant_arm_received=true' \
  'principal_self_measured=true' 'inherited_descriptor=true' \
  'host_mode=false' 'material_execution=true' 'artifact_executed=false' \
  'handle_is_bearer=false' 'handle_is_execution_authority=false' \
  'LOOM_EXEC_RESULT_RECORD/1' 'operation=sounio-check' 'exit_code=0'; do
  grep -Fq "$expected" <<<"$positive_result" ||
    fail "positive result omitted $expected"
done
[[ "$positive_closed" == LOOM_EXEC_OPERATION_CELL_CLOSED_V1* &&
   "$(field_value "$positive_closed" generation_sha256)" == "$GENERATION" &&
   "$(field_value "$positive_closed" event_sha256)" == "$EVENT" &&
   "$(field_value "$positive_closed" record_sha256)" == "$record_sha256" &&
   "$(field_value "$positive_closed" close_receipt_sha256)" == "$CLOSE_GRANT" &&
   "$(field_value "$positive_closed" authority_extinction)" == armed ]] ||
  fail 'positive CLOSE binding diverged'
artifact="$POSITIVE_DIR/loom-sounio-check-899d05ffe60528a6.elf"
[[ -f "$artifact" && ! -L "$artifact" ]] || fail 'material artifact absent'
for capture in "$artifact.stdout" "$artifact.stderr"; do
  [[ -f "$capture" && ! -L "$capture" && "$(stat -c '%a' "$capture")" == 600 ]] ||
    fail "material capture is absent or non-private: $capture"
done
[[ "$(find "$POSITIVE_DIR" -mindepth 1 -maxdepth 1 -type f | wc -l)" == 3 ]] ||
  fail 'positive operation cell did not retain exactly three material files'
artifact_sha256="$(sha256sum "$artifact" | cut -d ' ' -f 1)"
[[ "$artifact_sha256" == "$EXPECTED_ARTIFACT" ]] || fail 'material artifact drifted'
result_artifact_sha256="$(
  field_value "$(head -n1 <<<"$positive_result")" artifact_sha256
)"
[[ "$result_artifact_sha256" == "$artifact_sha256" ]] ||
  fail 'result header did not bind material artifact'

CLOSE_CONTROL_DIR="$TEST_ROOT/close-mismatch"
mkdir -m 0700 "$CLOSE_CONTROL_DIR"
coproc CLOSE_CONTROL {
  exec env SOUNIO_LOOM_EXEC_OPERATION_CELL_TEST_MODE=1 \
    "$LOOM" _exec-operation-cell --root "$ROOT_DIR" --source "$SOURCE" \
      --output-dir "$CLOSE_CONTROL_DIR" --unit "$UNIT" --mode test \
      2>"$TEST_ROOT/close-mismatch.err"
}
close_pid="$CLOSE_CONTROL_PID"
close_read="${CLOSE_CONTROL[0]}"
close_write="${CLOSE_CONTROL[1]}"
IFS= read -r close_ready <&"$close_read" || fail 'close control READY absent'
close_principal="$(field_value "$close_ready" principal_sha256)"
close_descriptor="$(field_value "$close_ready" descriptor_binding_sha256)"
printf 'ARM %s %s %s %s %s\n' \
  "$GENERATION" "$EVENT" "$close_principal" "$close_descriptor" "$GRANT" \
  >&"$close_write"
IFS= read -r close_header <&"$close_read" || fail 'close control result header absent'
for line_number in $(seq 1 18); do
  IFS= read -r close_record_line <&"$close_read" ||
    fail "close control record line $line_number absent"
done
printf 'CLOSE %s %s %s %s\n' \
  "$GENERATION" "$EVENT" "$WRONG" "$CLOSE_GRANT" >&"$close_write"
exec {close_write}>&-
cat <&"$close_read" >"$TEST_ROOT/close-mismatch.out" || true
set +e
wait "$close_pid"
close_status=$?
set -e
[[ $close_status -ne 0 ]] || fail 'record-mismatched CLOSE was admitted'
grep -Fq 'exec-operation-cell-close-record-mismatch' \
  "$TEST_ROOT/close-mismatch.err" || fail 'CLOSE mismatch reason diverged'
[[ ! -s "$TEST_ROOT/close-mismatch.out" ]] ||
  fail 'record-mismatched CLOSE emitted a closure frame'
[[ -f "$CLOSE_CONTROL_DIR/loom-sounio-check-899d05ffe60528a6.elf" ]] ||
  fail 'CLOSE control did not cross the material boundary'

PRINCIPAL_DIR="$TEST_ROOT/principal-mismatch"
mkdir -m 0700 "$PRINCIPAL_DIR"
coproc PRINCIPAL_CONTROL {
  exec env SOUNIO_LOOM_EXEC_OPERATION_CELL_TEST_MODE=1 \
    "$LOOM" _exec-operation-cell --root "$ROOT_DIR" --source "$SOURCE" \
      --output-dir "$PRINCIPAL_DIR" --unit "$UNIT" --mode test \
      2>"$TEST_ROOT/principal.err"
}
principal_pid="$PRINCIPAL_CONTROL_PID"
principal_read="${PRINCIPAL_CONTROL[0]}"
principal_write="${PRINCIPAL_CONTROL[1]}"
IFS= read -r principal_ready <&"$principal_read" || fail 'principal control READY absent'
principal_descriptor="$(field_value "$principal_ready" descriptor_binding_sha256)"
printf 'ARM %s %s %s %s %s\n' \
  "$GENERATION" "$EVENT" "$WRONG" "$principal_descriptor" "$GRANT" \
  >&"$principal_write"
exec {principal_write}>&-
cat <&"$principal_read" >"$TEST_ROOT/principal.out" || true
set +e
wait "$principal_pid"
principal_status=$?
set -e
[[ $principal_status -ne 0 ]] || fail 'principal mismatch was admitted'
grep -Fq 'exec-operation-cell-arm-principal-mismatch' "$TEST_ROOT/principal.err" ||
  fail 'principal mismatch reason diverged'
[[ -z "$(find "$PRINCIPAL_DIR" -mindepth 1 -print -quit)" ]] ||
  fail 'principal mismatch created material artifact'

DESCRIPTOR_DIR="$TEST_ROOT/descriptor-mismatch"
mkdir -m 0700 "$DESCRIPTOR_DIR"
coproc DESCRIPTOR_CONTROL {
  exec env SOUNIO_LOOM_EXEC_OPERATION_CELL_TEST_MODE=1 \
    "$LOOM" _exec-operation-cell --root "$ROOT_DIR" --source "$SOURCE" \
      --output-dir "$DESCRIPTOR_DIR" --unit "$UNIT" --mode test \
      2>"$TEST_ROOT/descriptor.err"
}
descriptor_pid="$DESCRIPTOR_CONTROL_PID"
descriptor_read="${DESCRIPTOR_CONTROL[0]}"
descriptor_write="${DESCRIPTOR_CONTROL[1]}"
IFS= read -r descriptor_ready <&"$descriptor_read" || fail 'descriptor control READY absent'
descriptor_principal="$(field_value "$descriptor_ready" principal_sha256)"
printf 'ARM %s %s %s %s %s\n' \
  "$GENERATION" "$EVENT" "$descriptor_principal" "$WRONG" "$GRANT" \
  >&"$descriptor_write"
exec {descriptor_write}>&-
cat <&"$descriptor_read" >"$TEST_ROOT/descriptor.out" || true
set +e
wait "$descriptor_pid"
descriptor_status=$?
set -e
[[ $descriptor_status -ne 0 ]] || fail 'descriptor mismatch was admitted'
grep -Fq 'exec-operation-cell-arm-descriptor-mismatch' "$TEST_ROOT/descriptor.err" ||
  fail 'descriptor mismatch reason diverged'
[[ -z "$(find "$DESCRIPTOR_DIR" -mindepth 1 -print -quit)" ]] ||
  fail 'descriptor mismatch created material artifact'

INVALID_DIR="$TEST_ROOT/invalid-source"
mkdir -m 0700 "$INVALID_DIR"
set +e
env SOUNIO_LOOM_EXEC_OPERATION_CELL_TEST_MODE=1 \
  "$LOOM" _exec-operation-cell --root "$ROOT_DIR" --source '../call_b.sio' \
    --output-dir "$INVALID_DIR" --unit "$UNIT" --mode test \
    </dev/null >"$TEST_ROOT/invalid-source.out" 2>"$TEST_ROOT/invalid-source.err"
invalid_status=$?
set -e
[[ $invalid_status -ne 0 ]] || fail 'invalid source was admitted'
grep -Fq 'SOUNIO_EXEC_OPERATION_CATALOG DENY563' \
  "$TEST_ROOT/invalid-source.err" || fail 'invalid source did not exercise DENY563'
[[ ! -s "$TEST_ROOT/invalid-source.out" ]] || fail 'invalid source reached READY'
[[ -z "$(find "$INVALID_DIR" -mindepth 1 -print -quit)" ]] ||
  fail 'invalid source created material artifact'

[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a prohibited oracle executed'
DEPENDENCIES="$(ldd "$LOOM" 2>&1 || true)"
printf '%s\n' "$DEPENDENCIES" | grep -Eqi 'python|rust' &&
  fail 'operation cell has a prohibited runtime dependency'

printf 'sounio-loom-exec-operation-cell-selftest: PASS semantic_authority=Sounio grant_action=9030 catalog_action=9035 result_action=9036 operational_language=OCaml operational_role=EFFECT_PARITY protocol=READY+ARM+CLOSE close_receipt_bound=true close_causal_rule=record_sha256_equal close_record_sabotage=REFUSED close_sabotage_closed_frame=false principal_self_measured=true descriptor_binding=inherited-pipe material_files=artifact+stdout+stderr material_files_measured=3 positive_material_execution=true artifact_sha256=%s artifact_executed=false dynamic_user_host_attached=false principal_mismatch=REFUSED descriptor_mismatch=REFUSED invalid_source=DENY563 sabotage_material_created=false handle_is_bearer=false handle_is_execution_authority=false python_executed=false rust_executed=false runtime_dependencies=clean provider_lifecycle_attached=false production_activation=false parity_open=false claim_ready=false\n' \
  "$artifact_sha256"
