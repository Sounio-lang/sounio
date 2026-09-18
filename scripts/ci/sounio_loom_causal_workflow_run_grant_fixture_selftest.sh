#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-causal-run-grant.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-causal-workflow-run-grant-fixture-selftest: FAIL: %s\n' "$*" >&2
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
  SOUNIO_LOOM_CAUSAL_RUN_GRANT_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_run_grant_fixture.sh" >/dev/null
done
cmp "$FIXTURE_ONE" "$FIXTURE_TWO" || fail 'two Sounio fixture builds differ'
SOUNIO_LOOM_KERNEL_EXEC_GRANT_CELL_OUTPUT="$AUTHORITY" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_kernel_exec_grant_cell_authority.sh" >/dev/null
"$FIXTURE_ONE" > "$BUNDLE"
[[ "$(wc -l < "$BUNDLE")" == 17 ]] || fail 'fixture bundle line count diverged'

command="$(sed -n 's/^COMMAND //p' "$BUNDLE")"
command_sha256="$(sed -n 's/^COMMAND_SHA256 //p' "$BUNDLE")"
intent_sha256="$(sed -n 's/^INTENT_SHA256 //p' "$BUNDLE")"
event_sha256="$(sed -n 's/^EVENT_SHA256 //p' "$BUNDLE")"
artifact_sha256="$(sed -n 's/^ARTIFACT_SHA256 //p' "$BUNDLE")"
[[ "$command" == 'loom-causal-cell-v1 RUN_EXACT artifact=eff2ac0ef28b34d6cc4f008cfb08a30ba18a0874c8654c06a3c62ec2f48a249c' &&
   "$(printf '%s' "$command" | sha256sum | cut -d ' ' -f 1)" == "$command_sha256" &&
   "$command_sha256" == 9ab40e390d585d45ce42127a063d8c7244670d13ef6955f0d3a279b964ce92b1 &&
   "$intent_sha256" == a6bd39a568166cd107f5fab4a68886d6e53e35816d9d27c930b9fa8d24bf853f &&
   "$event_sha256" == da096ff2b845df2ddcf80c62fe68786eafd4316758a210d3995b18605b830691 &&
   "$artifact_sha256" == eff2ac0ef28b34d6cc4f008cfb08a30ba18a0874c8654c06a3c62ec2f48a249c ]] ||
  fail 'typed RUN_EXACT command lineage diverged'

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

printf 'sounio-loom-causal-workflow-run-grant-fixture-selftest: PASS semantic_authority=Sounio producer=Sounio role=SEMANTIC_FIXTURE_PRODUCER launch_action=9030 workflow_action=9037 fixtures=4 treatment=issue+consume+close command_mismatch=DENY492 causal_sabotage=PASS command_sha256=%s intent_sha256=%s event_sha256=%s artifact_sha256=%s source_sha256=%s executable_sha256=%s bundle_sha256=%s deterministic=true arbitrary_shell=false expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false runtime_dependencies=clean material_grant=false material_execution=false host_launch_attached=false production_activation=false parity_open=false claim_ready=false\n' \
  "$command_sha256" "$intent_sha256" "$event_sha256" "$artifact_sha256" \
  "$(sha256sum "$ROOT_DIR/tools/loom/causal_workflow_run_grant_fixture_main.sio" | cut -d ' ' -f 1)" \
  "$(sha256sum "$FIXTURE_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BUNDLE" | cut -d ' ' -f 1)"
