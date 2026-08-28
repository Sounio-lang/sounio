#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/process_witness_handshake_payload.freeze.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-process-witness-handshake-payload-v1-20260828.txt"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-process-witness-handshake-freeze.XXXXXX")"
PAYLOAD_ONE="$TEST_ROOT/one/payload"
PAYLOAD_TWO="$TEST_ROOT/two/payload"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-handshake-payload-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

record_field() {
  local path="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$path" || true)"
  [[ "$count" == 1 ]] || fail "$key occurs $count times in ${path#$ROOT_DIR/}"
  line="$(grep -m1 "^${key}=" "$path")"
  printf '%s' "${line#*=}"
}

field() { record_field "$MANIFEST" "$1"; }
evidence_field() { record_field "$EVIDENCE" "$1"; }

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

run_case() {
  local label="$1" payload="$2" cwd="$3" input="$4" status
  set +e
  (
    cd "$cwd"
    env -i "$payload" < "$input" > "$TEST_ROOT/$label.stdout" 2> "$TEST_ROOT/$label.stderr"
  )
  status=$?
  set -e
  printf '%s' "$status" > "$TEST_ROOT/$label.status"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'handshake manifest is missing or linked'
[[ -f "$EVIDENCE" && ! -L "$EVIDENCE" ]] || fail 'handshake evidence is missing or linked'
expect_field schema loom-process-witness-handshake-payload-freeze-v1
expect_field stage SOUNIO_HANDSHAKE_PAYLOAD_FROZEN
expect_field producing_language Sounio
expect_field language_role SEMANTIC_PAYLOAD
expect_field semantic_authority Sounio
expect_field action 9030
expect_field two_phase true
expect_field source_precedes_expectations true
expect_field ready_before_close true
expect_field exact_close done
expect_field eof refused
expect_field wrong_close refused
expect_field extra_bytes refused
expect_field positive_status 0
expect_field refusal_status 70
expect_field material_grant true
expect_field material_execution false
for boundary in launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  expect_field "$boundary" false
done

SOURCE_COMMIT="$(field source_commit)"
FREEZE_COMMIT="$(field freeze_gate_commit)"
git -C "$ROOT_DIR" cat-file -e "${SOURCE_COMMIT}^{commit}" || fail 'handshake source commit is absent'
git -C "$ROOT_DIR" cat-file -e "${FREEZE_COMMIT}^{commit}" || fail 'handshake freeze-gate commit is absent'
for pair in \
  garden_path:garden_sha256 \
  source_path:source_sha256 \
  semantic_manifest_path:semantic_manifest_sha256 \
  host_grant_manifest_path:host_grant_manifest_sha256 \
  parent_payload_manifest_path:parent_payload_manifest_sha256 \
  build_script_path:build_script_sha256 \
  selftest_path:selftest_sha256; do
  path_key="${pair%%:*}"
  hash_key="${pair#*:}"
  path="$(field "$path_key")"
  expected="$(field "$hash_key")"
  [[ "$(file_hash "$ROOT_DIR/$path")" == "$expected" ]] || fail "$path drifted"
  [[ "$(git -C "$ROOT_DIR" show "$SOURCE_COMMIT:$path" | stream_hash)" == "$expected" ]] ||
    fail "$path differs from the handshake source commit"
done
FREEZE_PATH="$(field freeze_selftest_path)"
[[ "$(file_hash "$ROOT_DIR/$FREEZE_PATH")" == "$(field freeze_selftest_sha256)" ]] || fail 'freeze gate drifted'
[[ "$(git -C "$ROOT_DIR" show "$FREEZE_COMMIT:$FREEZE_PATH" | stream_hash)" == "$(field freeze_selftest_sha256)" ]] ||
  fail 'freeze gate differs from its commit'

SEMANTIC_MANIFEST="$ROOT_DIR/$(field semantic_manifest_path)"
HOST_GRANT_MANIFEST="$ROOT_DIR/$(field host_grant_manifest_path)"
PARENT_PAYLOAD_MANIFEST="$ROOT_DIR/$(field parent_payload_manifest_path)"
[[ "$(record_field "$SEMANTIC_MANIFEST" stage)" == SEMANTICS_FROZEN &&
   "$(record_field "$SEMANTIC_MANIFEST" producing_language)" == Sounio &&
   "$(record_field "$SEMANTIC_MANIFEST" language_role)" == SEMANTIC_AUTHORITY &&
   "$(record_field "$SEMANTIC_MANIFEST" action)" == 9030 ]] || fail 'Sounio semantic root drifted'
[[ "$(record_field "$HOST_GRANT_MANIFEST" stage)" == MATERIAL_GRANT_FROZEN &&
   "$(record_field "$HOST_GRANT_MANIFEST" material_grant)" == true &&
   "$(record_field "$HOST_GRANT_MANIFEST" material_execution)" == false ]] ||
  fail 'host material grant boundary drifted'
[[ "$(record_field "$PARENT_PAYLOAD_MANIFEST" stage)" == SOUNIO_PAYLOAD_FROZEN &&
   "$(record_field "$PARENT_PAYLOAD_MANIFEST" producing_language)" == Sounio ]] ||
  fail 'parent Sounio payload freeze drifted'
[[ "$(file_hash "$(field toolchain_wrapper_path)")" == "$(field toolchain_wrapper_sha256)" ]] ||
  fail 'Sounio compiler wrapper drifted'
[[ "$(file_hash "$(field toolchain_compiler_path)")" == "$(field toolchain_compiler_sha256)" ]] ||
  fail 'Sounio lean_single compiler drifted'

mkdir -p "$(dirname "$PAYLOAD_ONE")" "$(dirname "$PAYLOAD_TWO")"
for output in "$PAYLOAD_ONE" "$PAYLOAD_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_HANDSHAKE_OUTPUT="$output" \
    bash "$ROOT_DIR/$(field build_script_path)" >/dev/null
done
cmp "$PAYLOAD_ONE" "$PAYLOAD_TWO" || fail 'two frozen handshake rebuilds differ'
[[ "$(file_hash "$PAYLOAD_ONE")" == "$(field executable_sha256)" ]] || fail 'handshake executable hash drifted'

printf 'CLOSE\n' > "$TEST_ROOT/positive.input"
: > "$TEST_ROOT/eof.input"
printf 'CLOZE\n' > "$TEST_ROOT/wrong.input"
printf 'CLOSE\nX' > "$TEST_ROOT/extra.input"
run_case positive_one "$PAYLOAD_ONE" "$(dirname "$PAYLOAD_ONE")" "$TEST_ROOT/positive.input"
run_case positive_two "$PAYLOAD_TWO" "$(dirname "$PAYLOAD_TWO")" "$TEST_ROOT/positive.input"
run_case eof "$PAYLOAD_ONE" "$TEST_ROOT" "$TEST_ROOT/eof.input"
run_case wrong "$PAYLOAD_ONE" "$TEST_ROOT" "$TEST_ROOT/wrong.input"
run_case extra "$PAYLOAD_ONE" "$TEST_ROOT" "$TEST_ROOT/extra.input"

for label in positive_one positive_two; do
  [[ "$(file_hash "$TEST_ROOT/$label.stdout")" == "$(field positive_stdout_sha256)" ]] ||
    fail "$label stdout hash drifted"
  [[ "$(cat "$TEST_ROOT/$label.stdout")" == "$(field positive_stdout)" ]] ||
    fail "$label stdout bytes drifted"
  [[ "$(cat "$TEST_ROOT/$label.status")" == "$(field positive_status)" ]] ||
    fail "$label status drifted"
  [[ ! -s "$TEST_ROOT/$label.stderr" ]] || fail "$label stderr is not empty"
done
for label in eof wrong extra; do
  [[ "$(file_hash "$TEST_ROOT/$label.stdout")" == "$(field refusal_stdout_sha256)" ]] ||
    fail "$label refusal hash drifted"
  [[ "$(cat "$TEST_ROOT/$label.stdout")" == "$(field refusal_stdout)" ]] ||
    fail "$label refusal bytes drifted"
  [[ "$(cat "$TEST_ROOT/$label.status")" == "$(field refusal_status)" ]] ||
    fail "$label refusal status drifted"
  [[ ! -s "$TEST_ROOT/$label.stderr" ]] || fail "$label refusal stderr is not empty"
done
[[ "$(file_hash "$TEST_ROOT/positive_one.stderr")" == "$(field stderr_sha256)" ]] || fail 'stderr hash drifted'

command="$(field command)"
[[ "$command" == 'bash scripts/ci/sounio_loom_process_witness_handshake_payload_selftest.sh' ]] ||
  fail 'handshake gate command drifted'
[[ "$(printf '%s\n' "$command" | stream_hash)" == "$(field command_sha256)" ]] || fail 'command hash drifted'
result="$(bash "$ROOT_DIR/$(field selftest_path)")"
[[ "$result" == "$(field result)" ]] || fail 'handshake selftest result drifted'
[[ "$(printf '%s\n' "$result" | stream_hash)" == "$(field result_sha256)" ]] || fail 'result hash drifted'

[[ "$(file_hash "$EVIDENCE")" == "$(field evidence_sha256)" ]] || fail 'handshake evidence hash drifted'
expect_evidence schema loom-process-witness-handshake-payload-evidence-v1
expect_evidence stage SOUNIO_EXECUTABLE
for key in semantic_authority semantic_action producing_language language_role garden_sha256 source_sha256 \
  semantic_manifest_sha256 host_grant_manifest_sha256 parent_payload_manifest_sha256 executable_sha256 \
  positive_stdout positive_stdout_sha256 refusal_stdout refusal_stdout_sha256 stderr_sha256 positive_status \
  refusal_status command command_sha256 result result_sha256 material_grant material_execution launch_open \
  recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  manifest_key="$key"
  [[ "$key" == semantic_action ]] && manifest_key=action
  expect_evidence "$key" "$(field "$manifest_key")"
done

printf 'sounio-loom-process-witness-handshake-payload-freeze-selftest: PASS semantic_authority=Sounio producer=Sounio role=SEMANTIC_PAYLOAD action=9030 stage=SOUNIO_HANDSHAKE_PAYLOAD_FROZEN manifest_sha256=%s evidence_sha256=%s source_precedes_expectations=true two_phase=true ready_before_close=true exact_close=done eof=refused wrong_close=refused extra_bytes=refused executable_sha256=%s positive_stdout_sha256=%s refusal_stdout_sha256=%s positive_status=0 refusal_status=70 empty_environment=true empty_stderr=true shell_mediation=false host_material_grant=frozen material_grant=true material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(file_hash "$MANIFEST")" "$(field evidence_sha256)" "$(field executable_sha256)" \
  "$(field positive_stdout_sha256)" "$(field refusal_stdout_sha256)"
