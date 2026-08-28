#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/process_witness_payload.freeze.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-payload-v1-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-process-witness-payload-freeze.XXXXXX")"
PAYLOAD_ONE="$TEST_ROOT/one/payload"
PAYLOAD_TWO="$TEST_ROOT/two/payload"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-payload-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in ${path#$ROOT_DIR/}"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

field() {
  record_field "$MANIFEST" "$1"
}

evidence_field() {
  record_field "$EVIDENCE" "$1"
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

stream_hash() {
  local sum
  sum="$(sha256sum)"
  printf '%s' "${sum%% *}"
}

expect_field() {
  local key="$1" expected="$2" actual
  actual="$(field "$key")"
  [[ "$actual" == "$expected" ]] || fail "$key drifted: expected=$expected actual=$actual"
}

expect_evidence() {
  local key="$1" expected="$2" actual
  actual="$(evidence_field "$key")"
  [[ "$actual" == "$expected" ]] || fail "evidence $key drifted: expected=$expected actual=$actual"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'payload manifest is missing or linked'
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'payload evidence is missing or linked'
expect_field schema loom-process-witness-payload-freeze-v1
expect_field stage SOUNIO_PAYLOAD_FROZEN
expect_field producing_language Sounio
expect_field language_role SEMANTIC_PAYLOAD
expect_field semantic_authority Sounio
expect_field action 9030
expect_field payload_generation 1
expect_field deterministic_build true
expect_field deterministic_result true
expect_field empty_environment true
expect_field empty_stderr true
expect_field descendants_expected 0
expect_field write_set_expected empty
expect_field shell_mediation false
expect_field material_grant true
expect_field material_execution false
for boundary in launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_field "$boundary" false
done

SOURCE_COMMIT="$(field source_commit)"
FREEZE_COMMIT="$(field freeze_gate_commit)"
git -C "$ROOT_DIR" cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'payload source commit is absent'
git -C "$ROOT_DIR" cat-file -e "${FREEZE_COMMIT}^{commit}" || fail 'payload freeze-gate commit is absent'
for pair in \
  garden_path:garden_sha256 \
  source_path:source_sha256 \
  semantic_manifest_path:semantic_manifest_sha256 \
  host_grant_manifest_path:host_grant_manifest_sha256 \
  build_script_path:build_script_sha256 \
  selftest_path:selftest_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$ROOT_DIR/$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git -C "$ROOT_DIR" show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the payload source commit"
done
FREEZE_PATH="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$FREEZE_PATH")" == "$(field freeze_selftest_sha256)" ]] || fail 'freeze gate drifted'
[[ "$(git -C "$ROOT_DIR" show "$FREEZE_COMMIT:$FREEZE_PATH" | stream_hash)" == "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze gate differs from its commit'

SEMANTIC_MANIFEST="$ROOT_DIR/$(field semantic_manifest_path)"
HOST_GRANT_MANIFEST="$ROOT_DIR/$(field host_grant_manifest_path)"
[[ "$(record_field "$SEMANTIC_MANIFEST" stage)" == SEMANTICS_FROZEN &&
   "$(record_field "$SEMANTIC_MANIFEST" producing_language)" == Sounio &&
   "$(record_field "$SEMANTIC_MANIFEST" language_role)" == SEMANTIC_AUTHORITY &&
   "$(record_field "$SEMANTIC_MANIFEST" action)" == 9030 ]] ||
  fail 'Sounio semantic root drifted'
[[ "$(record_field "$HOST_GRANT_MANIFEST" stage)" == MATERIAL_GRANT_FROZEN &&
   "$(record_field "$HOST_GRANT_MANIFEST" material_grant)" == true &&
   "$(record_field "$HOST_GRANT_MANIFEST" material_execution)" == false ]] ||
  fail 'host material grant boundary drifted'

[[ "$(file_hash "$(field toolchain_wrapper_path)")" == "$(field toolchain_wrapper_sha256)" ]] ||
  fail 'Sounio compiler wrapper drifted'
[[ "$(file_hash "$(field toolchain_compiler_path)")" == "$(field toolchain_compiler_sha256)" ]] ||
  fail 'Sounio lean_single compiler drifted'

mkdir -p "$(dirname "$PAYLOAD_ONE")" "$(dirname "$PAYLOAD_TWO")"
for output in "$PAYLOAD_ONE" "$PAYLOAD_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_PAYLOAD_OUTPUT="$output" \
    bash "$ROOT_DIR/$(field build_script_path)" >/dev/null
done
cmp "$PAYLOAD_ONE" "$PAYLOAD_TWO" || fail 'two frozen payload rebuilds differ'
[[ "$(file_hash "$PAYLOAD_ONE")" == "$(field executable_sha256)" ]] || fail 'payload executable hash drifted'

run_payload() {
  local payload="$1" cwd="$2" stdout="$3" stderr="$4" status_path="$5" status
  set +e
  (
    cd "$cwd"
    env -i "$payload" > "$stdout" 2> "$stderr"
  )
  status=$?
  set -e
  printf '%s' "$status" > "$status_path"
}
run_payload "$PAYLOAD_ONE" "$(dirname "$PAYLOAD_ONE")" "$TEST_ROOT/stdout-one" "$TEST_ROOT/stderr-one" "$TEST_ROOT/status-one"
run_payload "$PAYLOAD_TWO" "$(dirname "$PAYLOAD_TWO")" "$TEST_ROOT/stdout-two" "$TEST_ROOT/stderr-two" "$TEST_ROOT/status-two"
cmp "$TEST_ROOT/stdout-one" "$TEST_ROOT/stdout-two" || fail 'frozen payload stdout differs by cwd'
cmp "$TEST_ROOT/stderr-one" "$TEST_ROOT/stderr-two" || fail 'frozen payload stderr differs by cwd'
cmp "$TEST_ROOT/status-one" "$TEST_ROOT/status-two" || fail 'frozen payload status differs by cwd'
[[ "$(file_hash "$TEST_ROOT/stdout-one")" == "$(field stdout_sha256)" ]] || fail 'payload stdout hash drifted'
[[ "$(file_hash "$TEST_ROOT/stderr-one")" == "$(field stderr_sha256)" ]] || fail 'payload stderr hash drifted'
[[ "$(cat "$TEST_ROOT/status-one")" == "$(field exit_status)" ]] || fail 'payload exit status drifted'
[[ "$(cat "$TEST_ROOT/stdout-one")" == "$(field stdout)" ]] || fail 'payload stdout bytes drifted'
[[ ! -s "$TEST_ROOT/stderr-one" ]] || fail 'payload stderr is not empty'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_process_witness_payload_selftest.sh' ]] || fail 'payload gate command drifted'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'payload command hash drifted'
result="$(bash "$ROOT_DIR/$(field selftest_path)")"
[[ "$result" == "$(field result)" ]] || fail 'payload selftest result drifted'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'payload selftest result hash drifted'

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] || fail 'payload evidence hash drifted'
for key in schema stage semantic_authority semantic_action producing_language language_role \
  garden_sha256 source_sha256 semantic_manifest_sha256 host_grant_manifest_sha256 \
  executable_sha256 stdout_sha256 stderr_sha256 exit_status command command_sha256 result result_sha256 \
  material_grant material_execution launch_open recycle_open exec_attached commit_attached ci_attached \
  parity_open claim_ready; do
  manifest_key="$key"
  [[ "$key" == semantic_action ]] && manifest_key=action
  expect_evidence "$key" "$(field "$manifest_key")"
done

printf 'sounio-loom-process-witness-payload-freeze-selftest: PASS semantic_authority=Sounio producer=Sounio role=SEMANTIC_PAYLOAD action=9030 stage=SOUNIO_PAYLOAD_FROZEN manifest_sha256=%s evidence_sha256=%s garden_commit=%s source_commit=%s deterministic_build=true deterministic_result=true executable_sha256=%s stdout_sha256=%s stderr_sha256=%s exit_status=0 empty_environment=true empty_stderr=true descendants_expected=0 write_set_expected=empty shell_mediation=false host_material_grant=frozen material_grant=true material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)" "$(field garden_commit)" \
  "$SOURCE_COMMIT" "$(field executable_sha256)" "$(field stdout_sha256)" "$(field stderr_sha256)"
