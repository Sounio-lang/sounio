#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/exec_result_handle.freeze.v1"

fail() {
  printf 'sounio-loom-exec-result-handle-freeze-selftest: FAIL: %s\n' "$*" >&2
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

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'freeze manifest is absent or linked'
expect_value schema loom-exec-result-handle-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value producing_language Sounio
expect_value language_role SEMANTIC_AUTHORITY
expect_value action 9033
expect_value concept_id SOUNIO-LOOM-EXEC-RESULT-HANDLE
expect_value case_count 16
expect_value causal_sabotage PASS
expect_value load_bearing_rule command_sha256_equal
expect_value prewrite_validation true
expect_value handle_is_bearer false
expect_value handle_is_execution_authority false
expect_value handle_is_semantic_proof false
expect_value expected_results_encoded_in_material_layer false
expect_value material_execution false
expect_value exec_cell_attached false
expect_value result_store_attached false
expect_value provider_hook_switched false
expect_value production_activation false
expect_value parity_open false
expect_value claim_ready false

for key in garden contract source entrypoint build_script selftest evidence \
  parent_9030_manifest parent_9031_manifest fixture_manifest \
  toolchain_wrapper toolchain_compiler; do
  expect_hash "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

source_commit="$(manifest_value sounio_executable_commit)"
git -C "$ROOT_DIR" cat-file -e "${source_commit}^{commit}" ||
  fail 'Sounio executable commit is absent'
for key in garden source entrypoint build_script selftest; do
  path="$(manifest_value "${key}_path")"
  at_commit="$(git -C "$ROOT_DIR" show "${source_commit}:$path" | sha256sum | cut -d ' ' -f 1)"
  [[ "$at_commit" == "$(manifest_value "${key}_sha256")" ]] ||
    fail "$key is not bound to the Sounio executable commit"
done

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-result-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_EXEC_RESULT_HANDLE_OUTPUT="$work/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_handle_fixture.sh" >/dev/null
done
cmp "$work/runtime-one" "$work/runtime-two" || fail 'Sounio rebuild is nondeterministic'
[[ "$(sha256sum "$work/runtime-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] || fail 'Sounio executable hash drifted'

handle_fields="$(manifest_value handle_fields_schema)
$(manifest_value event_sha256)
$(manifest_value grant_generation)
$(manifest_value result_receipt_sha256)"
publish_frame="$(manifest_value wire_schema) $(manifest_value publish_word0) $(manifest_value common_word1) $(manifest_value common_word2)"
resolve_frame="$(manifest_value wire_schema) $(manifest_value resolve_word0) $(manifest_value common_word1) $(manifest_value common_word2)"
mismatch_frame="$(manifest_value wire_schema) $(manifest_value command_mismatch_word0) $(manifest_value common_word1) $(manifest_value common_word2)"
[[ "$(printf '%s\n' "$publish_frame" | "$work/runtime-one")" == \
   "$(manifest_value publish_decision)
$handle_fields" ]] || fail 'publish decision or result fields drifted'
[[ "$(printf '%s\n' "$resolve_frame" | "$work/runtime-one")" == \
   "$(manifest_value resolve_decision)
$handle_fields" ]] || fail 'resolve decision or result fields drifted'
set +e
mismatch_result="$(printf '%s\n' "$mismatch_frame" | "$work/runtime-one")"
mismatch_code=$?
set -e
[[ $mismatch_code -eq 42 && "$mismatch_result" == \
   "$(manifest_value command_mismatch_decision)" ]] ||
  fail 'command-binding sabotage drifted'

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_exec_result_handle_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] || fail 'Sounio fixture result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] || fail 'Sounio fixture result hash drifted'
printf '%s\n' "$result" | cmp - "$ROOT_DIR/$(manifest_value evidence_path)" ||
  fail 'evidence is not the exact executable result'

printf 'sounio-loom-exec-result-handle-freeze-selftest: PASS semantic_authority=Sounio action=9033 stage=SEMANTICS_FROZEN executable_sha256=%s treatment=PUBLISH+RESOLVE command_mismatch=DENY534 causal_sabotage=PASS handle=%s python_executed=false rust_executed=false material_execution=false exec_cell_attached=false result_store_attached=false provider_hook_switched=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(manifest_value executable_sha256)" "$(manifest_value canonical_handle)"
