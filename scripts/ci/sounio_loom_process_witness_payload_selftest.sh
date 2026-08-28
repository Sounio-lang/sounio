#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-process-witness-payload-selftest.XXXXXX")"
PAYLOAD_ONE="$TEST_ROOT/payload-one"
PAYLOAD_TWO="$TEST_ROOT/payload-two"
STDOUT_ONE="$TEST_ROOT/stdout-one"
STDOUT_TWO="$TEST_ROOT/stdout-two"
STDERR_ONE="$TEST_ROOT/stderr-one"
STDERR_TWO="$TEST_ROOT/stderr-two"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-process-witness-payload-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

file_hash() {
  local sum
  sum="$(sha256sum "$1")"
  printf '%s' "${sum%% *}"
}

for output in "$PAYLOAD_ONE" "$PAYLOAD_TWO"; do
  SOUNIO_LOOM_PROCESS_WITNESS_PAYLOAD_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_process_witness_payload.sh" >/dev/null
done
cmp "$PAYLOAD_ONE" "$PAYLOAD_TWO" || fail 'two Sounio payload builds differ'
[[ "$(stat -c '%a' "$PAYLOAD_ONE")" == 755 && ! -u "$PAYLOAD_ONE" && ! -g "$PAYLOAD_ONE" ]] ||
  fail 'payload executable mode is unsafe'

set +e
(
  cd "$TEST_ROOT"
  env -i "$PAYLOAD_ONE" > "$STDOUT_ONE" 2> "$STDERR_ONE"
)
status_one=$?
(
  cd "$TEST_ROOT"
  env -i "$PAYLOAD_TWO" > "$STDOUT_TWO" 2> "$STDERR_TWO"
)
status_two=$?
set -e

[[ "$status_one" == "$status_two" ]] || fail 'payload exit status is nondeterministic'
[[ "$status_one" == 0 ]] || fail "Sounio payload returned nonzero status: $status_one"
cmp "$STDOUT_ONE" "$STDOUT_TWO" || fail 'payload stdout is nondeterministic'
cmp "$STDERR_ONE" "$STDERR_TWO" || fail 'payload stderr is nondeterministic'
[[ -s "$STDOUT_ONE" && "$(wc -l < "$STDOUT_ONE")" == 1 && "$(wc -c < "$STDOUT_ONE")" -le 512 ]] ||
  fail 'payload stdout is empty, multiline, or unbounded'
[[ ! -s "$STDERR_ONE" ]] || fail 'payload stderr is not empty'

dependencies="$(ldd "$PAYLOAD_ONE" 2>&1 || true)"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'payload executable has a prohibited runtime dependency'
fi
if grep -Ev '^[[:space:]]*//!' "$ROOT_DIR/tools/loom/process_witness_payload_main.sio" | \
  grep -Eqi 'getenv|argv|clock|time\(|random|socket|connect|open\(|read\(|write\('; then
  fail 'payload source acquired a prohibited ambient input or effect'
fi

printf 'sounio-loom-process-witness-payload-selftest: PASS semantic_authority=Sounio producer=Sounio role=SEMANTIC_PAYLOAD action=9030 garden=preregistered builds=2 deterministic_binary=true deterministic_stdout=true deterministic_stderr=true source_sha256=%s executable_sha256=%s stdout_sha256=%s stderr_sha256=%s exit_status=%s output_lines=1 output_bounded=true empty_environment=true empty_stderr=true descendants_expected=0 write_set_expected=empty shell_mediation=false python_executed=false rust_executed=false runtime_dependencies=clean host_material_grant=frozen material_grant=true material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n' \
  "$(file_hash "$ROOT_DIR/tools/loom/process_witness_payload_main.sio")" \
  "$(file_hash "$PAYLOAD_ONE")" "$(file_hash "$STDOUT_ONE")" \
  "$(file_hash "$STDERR_ONE")" "$status_one"
