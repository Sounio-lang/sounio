#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/causal_workflow_attest_grant_fixture.freeze.v1"

fail() {
  printf 'sounio-loom-causal-workflow-attest-grant-fixture-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

value() {
  local line
  line="$(grep -m1 "^$1=" "$MANIFEST")"
  [[ -n "$line" ]] || fail "manifest field $1 absent"
  printf '%s' "${line#*=}"
}

expect() { [[ "$(value "$1")" == "$2" ]] || fail "field $1 drifted"; }
expect_hash() {
  local path
  path="$(value "$1_path")"
  [[ -f "$ROOT_DIR/$path" && ! -L "$ROOT_DIR/$path" && \
     "$(sha256sum "$ROOT_DIR/$path" | cut -d ' ' -f 1)" == "$(value "$1_sha256")" ]] ||
    fail "$1 binding drifted"
}

expect schema loom-causal-workflow-attest-grant-fixture-freeze-v1
expect stage SEMANTICS_FROZEN
expect producing_language Sounio
expect language_role SEMANTIC_FIXTURE_PRODUCER
expect semantic_authority Sounio
expect launch_action 9030
expect workflow_action 9037
expect fixture_count 4
expect command_mismatch_result DENY492
expect causal_sabotage PASS
expect run_ticket_is_bearer false
expect run_ticket_is_execution_authority false
expect launch_authority action-9030
expect material_grant false
expect material_execution false
expect host_launch_attached false
expect production_activation false
expect parity_open false
expect claim_ready false
for key in source build_script selftest freeze_selftest first_manifest evidence \
  parent_9030_manifest parent_9037_manifest toolchain_wrapper toolchain_compiler; do
  expect_hash "$key"
done

source_commit="$(value source_commit)"
for key in source build_script selftest; do
  path="$(value "${key}_path")"
  [[ "$(git -C "$ROOT_DIR" show "${source_commit}:$path" | sha256sum | cut -d ' ' -f 1)" == \
     "$(value "${key}_sha256")" ]] || fail "$key is not source-commit bound"
done

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-causal-attest-grant-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_CAUSAL_ATTEST_GRANT_OUTPUT="$work/fixture-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_causal_workflow_attest_grant_fixture.sh" >/dev/null
  "$work/fixture-$ordinal" > "$work/bundle-$ordinal"
done
cmp "$work/fixture-one" "$work/fixture-two" || fail 'fixture rebuild differs'
cmp "$work/bundle-one" "$work/bundle-two" || fail 'fixture output differs'
[[ "$(sha256sum "$work/fixture-one" | cut -d ' ' -f 1)" == "$(value executable_sha256)" &&
   "$(sha256sum "$work/bundle-one" | cut -d ' ' -f 1)" == "$(value bundle_sha256)" ]] ||
  fail 'frozen executable or bundle drifted'
result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_causal_workflow_attest_grant_fixture_selftest.sh")"
[[ "$result" == "$(value result)" &&
   "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == "$(value result_sha256)" ]] ||
  fail 'frozen selftest result drifted'

printf 'sounio-loom-causal-workflow-attest-grant-fixture-freeze-selftest: PASS semantic_authority=Sounio launch_action=9030 workflow_action=9037 stage=SEMANTICS_FROZEN source_sha256=%s executable_sha256=%s bundle_sha256=%s command_sha256=%s intent_sha256=%s event_sha256=%s command_mismatch=DENY492 causal_sabotage=PASS arbitrary_shell=false python_executed=false rust_executed=false material_grant=false material_execution=false host_launch_attached=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(value source_sha256)" "$(value executable_sha256)" "$(value bundle_sha256)" \
  "$(value command_sha256)" "$(value intent_sha256)" "$(value event_sha256)"
