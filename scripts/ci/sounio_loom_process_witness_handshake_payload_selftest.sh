#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-process-witness-handshake-selftest.XXXXXX")"
PAYLOAD_ONE="$TEST_ROOT/one/payload"
PAYLOAD_TWO="$TEST_ROOT/two/payload"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-handshake-payload-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

run_case() {
  local label="$1" payload="$2" cwd="$3" input="$4"
  local stdout="$TEST_ROOT/$label.stdout" stderr="$TEST_ROOT/$label.stderr"
  local status_path="$TEST_ROOT/$label.status" status
  set +e
  (
    cd "$cwd"
    env -i "$payload" < "$input" > "$stdout" 2> "$stderr"
  )
  status=$?
  set -e
  printf '%s' "$status" > "$status_path"
}

mkdir -p "$(dirname "$PAYLOAD_ONE")" "$(dirname "$PAYLOAD_TWO")"
for output in "$PAYLOAD_ONE" "$PAYLOAD_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_HANDSHAKE_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_handshake_payload.sh" >/dev/null
done
cmp "$PAYLOAD_ONE" "$PAYLOAD_TWO" || fail 'two Sounio handshake builds differ'
[[ "$(file_hash "$PAYLOAD_ONE")" == 7249748c322ede756c779904cb2d87f561ba2e17d0691314a09feaf16ca2ed4d ]] ||
  fail 'Sounio handshake executable hash drifted'
[[ "$(stat -c '%a' "$PAYLOAD_ONE")" == 755 && ! -u "$PAYLOAD_ONE" && ! -g "$PAYLOAD_ONE" ]] ||
  fail 'handshake payload mode is unsafe'

printf 'CLOSE\n' > "$TEST_ROOT/positive.input"
: > "$TEST_ROOT/eof.input"
printf 'CLOZE\n' > "$TEST_ROOT/wrong.input"
printf 'CLOSE\nX' > "$TEST_ROOT/extra.input"
run_case positive_one "$PAYLOAD_ONE" "$(dirname "$PAYLOAD_ONE")" "$TEST_ROOT/positive.input"
run_case positive_two "$PAYLOAD_TWO" "$(dirname "$PAYLOAD_TWO")" "$TEST_ROOT/positive.input"
run_case eof "$PAYLOAD_ONE" "$TEST_ROOT" "$TEST_ROOT/eof.input"
run_case wrong "$PAYLOAD_ONE" "$TEST_ROOT" "$TEST_ROOT/wrong.input"
run_case extra "$PAYLOAD_ONE" "$TEST_ROOT" "$TEST_ROOT/extra.input"

cmp "$TEST_ROOT/positive_one.stdout" "$TEST_ROOT/positive_two.stdout" ||
  fail 'positive stdout differs by build or cwd'
cmp "$TEST_ROOT/positive_one.stderr" "$TEST_ROOT/positive_two.stderr" ||
  fail 'positive stderr differs by build or cwd'
[[ "$(file_hash "$TEST_ROOT/positive_one.stdout")" == 5805e6579b6420ba0dd693d385715943955d0e69e657f44e94e23d20a20d27d1 ]] ||
  fail 'positive Sounio result hash drifted'
[[ "$(cat "$TEST_ROOT/positive_one.status")" == 0 && "$(cat "$TEST_ROOT/positive_two.status")" == 0 ]] ||
  fail 'positive Sounio status drifted'
[[ ! -s "$TEST_ROOT/positive_one.stderr" ]] || fail 'positive Sounio stderr is not empty'

for label in eof wrong extra; do
  [[ "$(file_hash "$TEST_ROOT/$label.stdout")" == e05f66cf22acbfa9123f2f9b095bcd4fa4198cee12fc64b3d90f33810522ce71 ]] ||
    fail "$label refusal result hash drifted"
  [[ "$(cat "$TEST_ROOT/$label.status")" == 70 ]] || fail "$label refusal status drifted"
  [[ ! -s "$TEST_ROOT/$label.stderr" ]] || fail "$label refusal wrote stderr"
done
[[ "$(file_hash "$TEST_ROOT/positive_one.stdout")" != "$(file_hash "$TEST_ROOT/eof.stdout")" ]] ||
  fail 'positive and refusal results collapsed'

dependencies="$(ldd "$PAYLOAD_ONE" 2>&1 || true)"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'handshake payload has a prohibited runtime dependency'
fi

printf 'sounio-loom-process-witness-handshake-payload-selftest: PASS semantic_authority=Sounio producer=Sounio role=SEMANTIC_PAYLOAD action=9030 garden=preregistered source_precedes_expectations=true builds=2 deterministic_binary=true two_phase=true ready_before_close=true exact_close=done eof=refused wrong_close=refused extra_bytes=refused positive_status=0 refusal_status=70 source_sha256=%s executable_sha256=%s positive_stdout_sha256=%s refusal_stdout_sha256=%s stderr_sha256=%s empty_environment=true empty_stderr=true shell_mediation=false python_executed=false rust_executed=false runtime_dependencies=clean host_material_grant=frozen material_grant=true material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(file_hash "$ROOT_DIR/tools/loom/process_witness_handshake_payload_main.sio")" \
  "$(file_hash "$PAYLOAD_ONE")" "$(file_hash "$TEST_ROOT/positive_one.stdout")" \
  "$(file_hash "$TEST_ROOT/eof.stdout")" "$(file_hash "$TEST_ROOT/positive_one.stderr")"
