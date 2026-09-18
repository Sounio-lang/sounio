#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-operation-grant-fixture.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-exec-operation-grant-fixture-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

MANIFEST="$ROOT_DIR/tools/loom/kernel_exec_grant_cell_authority.freeze.v1"
FIXTURE_ONE="$TEST_ROOT/fixture-one"
FIXTURE_TWO="$TEST_ROOT/fixture-two"
AUTHORITY="$TEST_ROOT/action-9030-authority"
BUNDLE="$TEST_ROOT/fixtures.v1"

manifest_value() {
  local key="$1" count line
  count="$(grep -c "^${key}=" "$MANIFEST" || true)"
  [[ "$count" == 1 ]] || fail "authority field $key occurs $count times"
  line="$(grep -m1 "^${key}=" "$MANIFEST")"
  printf '%s' "${line#*=}"
}

hex_u32() {
  local digest="$1" offset
  for ((offset = 0; offset < 64; offset += 8)); do
    printf ' %u' "$((16#${digest:offset:8}))"
  done
}

frame_for() {
  local label="$1" prefix="FRAME $1 " line parents key digest
  line="$(grep -m1 "^FRAME ${label} " "$BUNDLE")"
  parents="$(grep -m1 '^PARENT_BINDINGS ' "$BUNDLE")"
  printf '%s %s' "${line#"$prefix"}" "${parents#PARENT_BINDINGS }"
  for key in grant_identity command_environment peer_vector transition_journal source_semantics_toolchain result_receipt; do
    digest="$(sed -n "s/^BINDING ${key} //p" "$BUNDLE")"
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || fail "binding hash $key is malformed"
    hex_u32 "$digest"
  done
}

for output in "$FIXTURE_ONE" "$FIXTURE_TWO"; do
  SOUNIO_LOOM_EXEC_OPERATION_GRANT_FIXTURE_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_operation_grant_fixture.sh" >/dev/null
done
cmp "$FIXTURE_ONE" "$FIXTURE_TWO" || fail 'two Sounio fixture builds differ'
SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_exec_grant_cell_authority.sh" >/dev/null
"$FIXTURE_ONE" > "$BUNDLE"
[[ "$(wc -l < "$BUNDLE")" == 15 ]] || fail 'fixture bundle line count diverged'

command="$(sed -n 's/^COMMAND //p' "$BUNDLE")"
event="$(sed -n 's/^EVENT //p' "$BUNDLE")"
command_sha256="$(printf '%s' "$command" | sha256sum | cut -d ' ' -f 1)"
[[ "$command" == 'loom-exec-cell-v2 sounio-check source=tests/verify-ir/call_b.sio' &&
   "$command_sha256" == b5566b9ef6aa68866db784bbb33792d0dda4506932fbb10240db96ce99e1a27d &&
   "$event" == 6017e4c6e745560696f78836f9cc07ec71a9106f13ad1bfdb16d7e342f0840a9 ]] ||
  fail 'typed command or event binding diverged'

check_fixture() {
  local label="$1" expected_field="$2" actual
  actual="$(printf '%s\n' "$(frame_for "$label")" | "$AUTHORITY" || true)"
  [[ "$actual" == "$(manifest_value "$expected_field")" ]] ||
    fail "Sounio authority disagreed with fixture $label: $actual"
}
check_fixture issue issue_decision
check_fixture consume consume_decision
check_fixture close close_decision
mismatch="$(printf '%s\n' "$(frame_for command_mismatch)" | "$AUTHORITY" || true)"
[[ "$mismatch" == 'SOUNIO_KERNEL_EXEC_GRANT_CELL_DENY code=492 '* ]] ||
  fail "command mismatch control diverged: $mismatch"

ORACLE_EXECUTED="$TEST_ROOT/oracle-executed"
for name in python3 rustc; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
PATH="$TEST_ROOT:$PATH" "$FIXTURE_ONE" >/dev/null
[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a prohibited oracle executed'
DEPENDENCIES="$(ldd "$FIXTURE_ONE" 2>&1 || true)"
printf '%s\n' "$DEPENDENCIES" | grep -Eqi 'python|rust' &&
  fail 'fixture executable has a prohibited runtime dependency'

printf 'sounio-loom-exec-operation-grant-fixture-selftest: PASS semantic_authority=Sounio producer=Sounio role=SEMANTIC_FIXTURE_PRODUCER action=9030 catalog_action=9035 result_action=9036 fixtures=4 treatment=issue+consume+close command_mismatch=DENY492 causal_sabotage=PASS command_sha256=%s event_sha256=%s intent_sha256=db5e4c791f346ee1e2248938ddc66a0a0bbfb635f12b2a6bb15fc60da8bf62cc source_sha256=%s executable_sha256=%s bundle_sha256=%s deterministic=true arbitrary_shell=false expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false runtime_dependencies=clean material_grant=false material_execution=false host_payload_selection_attached=false provider_lifecycle_attached=false production_activation=false parity_open=false claim_ready=false\n' \
  "$command_sha256" "$event" \
  "$(sha256sum "$ROOT_DIR/tools/loom/exec_operation_grant_fixture_main.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$FIXTURE_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)"
