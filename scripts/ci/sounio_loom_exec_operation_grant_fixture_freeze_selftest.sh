#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/exec_operation_grant_fixture.freeze.v1"

fail() {
  printf 'sounio-loom-exec-operation-grant-fixture-freeze-selftest: FAIL: %s\n' "$*" >&2
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
  local key="$1" expected="$2" actual
  actual="$(manifest_value "$key")"
  [[ "$actual" == "$expected" ]] ||
    fail "$key expected $expected but found $actual"
}

expect_hash() {
  local path="$1" expected="$2" actual
  [[ -f "$ROOT_DIR/$path" && ! -L "$ROOT_DIR/$path" ]] ||
    fail "$path is absent or linked"
  actual="$(sha256sum "$ROOT_DIR/$path" | cut -d ' ' -f 1)"
  [[ "$actual" == "$expected" ]] || fail "$path hash drifted"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] ||
  fail 'freeze manifest is absent or linked'
expect_value schema loom-exec-operation-grant-fixture-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value producing_language Sounio
expect_value language_role SEMANTIC_FIXTURE_PRODUCER
expect_value semantic_authority Sounio
expect_value action 9030
expect_value catalog_action 9035
expect_value result_action 9036
expect_value fixture_count 4
expect_value command_mismatch_result DENY492
expect_value causal_sabotage PASS
expect_value arbitrary_shell false
expect_value expected_results_encoded_in_material_layer false
expect_value material_grant false
expect_value material_execution false
expect_value host_payload_selection_attached false
expect_value provider_lifecycle_attached false
expect_value production_activation false
expect_value parity_open false
expect_value claim_ready false

for key in garden source authority_manifest catalog_manifest result_manifest \
  build_script selftest freeze_selftest evidence toolchain_wrapper \
  toolchain_compiler; do
  expect_hash "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

source_commit="$(manifest_value source_commit)"
git -C "$ROOT_DIR" cat-file -e "${source_commit}^{commit}" ||
  fail 'source commit is unavailable'
for key in source build_script selftest; do
  path="$(manifest_value "${key}_path")"
  at_commit="$(git -C "$ROOT_DIR" show "${source_commit}:$path" | sha256sum | cut -d ' ' -f 1)"
  [[ "$at_commit" == "$(manifest_value "${key}_sha256")" ]] ||
    fail "$key is not bound to the Sounio executable commit"
done

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-operation-grant-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_EXEC_OPERATION_GRANT_FIXTURE_OUTPUT="$work/fixture-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_operation_grant_fixture.sh" \
      >/dev/null
  "$work/fixture-$ordinal" > "$work/bundle-$ordinal"
done
cmp "$work/fixture-one" "$work/fixture-two" ||
  fail 'fixture rebuild is nondeterministic'
cmp "$work/bundle-one" "$work/bundle-two" ||
  fail 'fixture output is nondeterministic'
[[ "$(sha256sum "$work/fixture-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] || fail 'fixture executable hash drifted'
[[ "$(sha256sum "$work/bundle-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value bundle_sha256)" ]] || fail 'fixture bundle hash drifted'
[[ "$(wc -l < "$work/bundle-one")" == "$(manifest_value bundle_lines)" ]] ||
  fail 'fixture bundle line count drifted'

command="$(sed -n 's/^COMMAND //p' "$work/bundle-one")"
event="$(sed -n 's/^EVENT //p' "$work/bundle-one")"
[[ "$command" == "$(manifest_value command)" ]] || fail 'typed command drifted'
[[ "$(printf '%s' "$command" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value command_sha256)" ]] || fail 'typed command hash drifted'
[[ "$event" == "$(manifest_value event_sha256)" ]] || fail 'semantic event drifted'

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_exec_operation_grant_fixture_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] || fail 'fixture result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] || fail 'fixture result hash drifted'
evidence_result="$(sed -n 's/^result=//p' "$ROOT_DIR/$(manifest_value evidence_path)")"
[[ "$evidence_result" == "$result" ]] || fail 'evidence result diverged'

printf 'sounio-loom-exec-operation-grant-fixture-freeze-selftest: PASS semantic_authority=Sounio action=9030 catalog_action=9035 result_action=9036 stage=SEMANTICS_FROZEN source_sha256=%s executable_sha256=%s bundle_sha256=%s command_sha256=%s event_sha256=%s intent_sha256=%s command_mismatch=DENY492 causal_sabotage=PASS arbitrary_shell=false python_executed=false rust_executed=false material_grant=false material_execution=false host_payload_selection_attached=false provider_lifecycle_attached=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(manifest_value source_sha256)" "$(manifest_value executable_sha256)" \
  "$(manifest_value bundle_sha256)" "$(manifest_value command_sha256)" \
  "$(manifest_value event_sha256)" "$(manifest_value intent_sha256)"
