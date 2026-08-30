#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/exec_result_record.freeze.v1"

fail() {
  printf 'sounio-loom-exec-result-record-freeze-selftest: FAIL: %s\n' "$*" >&2
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
expect_value schema loom-exec-result-record-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value producing_language Sounio
expect_value language_role SEMANTIC_AUTHORITY
expect_value action 9036
expect_value concept_id SOUNIO-LOOM-EXEC-RESULT-RECORD
expect_value case_count 9
expect_value handle_is_bearer false
expect_value handle_is_execution_authority false
expect_value artifact_executed false
expect_value expected_results_encoded_in_material_layer false
expect_value causal_sabotage PASS
expect_value load_bearing_rule artifact_sha256_equal_canonical_record_field
expect_value ocaml_record_projection_attached false
expect_value dynamic_user_host_attached false
expect_value provider_result_returned false
expect_value production_activation false
expect_value parity_open false
expect_value claim_ready false

for key in garden contract source entrypoint build_script selftest evidence \
  parent_9035_manifest toolchain_wrapper toolchain_compiler; do
  expect_hash "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

source_commit="$(manifest_value sounio_executable_commit)"
git -C "$ROOT_DIR" cat-file -e "${source_commit}^{commit}" ||
  fail 'Sounio executable commit is absent'
for key in garden contract source entrypoint build_script selftest evidence; do
  path="$(manifest_value "${key}_path")"
  at_commit="$(git -C "$ROOT_DIR" show "${source_commit}:$path" | sha256sum | cut -d ' ' -f 1)"
  [[ "$at_commit" == "$(manifest_value "${key}_sha256")" ]] ||
    fail "$key is not bound to the Sounio executable commit"
done

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-result-record-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_EXEC_RESULT_RECORD_OUTPUT="$work/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_result_record_fixture.sh" >/dev/null
done
cmp "$work/runtime-one" "$work/runtime-two" ||
  fail 'Sounio rebuild is nondeterministic'
[[ "$(sha256sum "$work/runtime-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] ||
  fail 'Sounio executable hash drifted'

fields="$(printf '%s\n' \
  "$(manifest_value fields_schema)" \
  "$(manifest_value record_schema)" \
  "$(manifest_value operation)" \
  "$(manifest_value catalog_sha256)" \
  "$(manifest_value catalog_result_schema_sha256)" \
  "$(manifest_value record_schema_sha256)" \
  "$(manifest_value canonical_fields)" \
  "$(manifest_value handle_recipe)" \
  "$(manifest_value record_hash_recipe)" \
  "handle_is_bearer=$(manifest_value handle_is_bearer)" \
  "handle_is_execution_authority=$(manifest_value handle_is_execution_authority)" \
  "artifact_executed=$(manifest_value artifact_executed)")"
positive="$(manifest_value wire_schema) $(manifest_value positive_word0) $(manifest_value positive_word1)"
[[ "$(printf '%s\n' "$positive" | "$work/runtime-one")" == \
   "$(manifest_value issue_decision)
$fields" ]] || fail 'positive record schema drifted'

for control in stage schema material binding runtime artifact_binding; do
  frame="$(manifest_value wire_schema) $(manifest_value "${control}_word0") $(manifest_value positive_word1)"
  set +e
  output="$(printf '%s\n' "$frame" | "$work/runtime-one")"
  code=$?
  set -e
  [[ $code -eq 42 && "$output" == "$(manifest_value "${control}_decision")" ]] ||
    fail "$control control drifted"
done
for control in authority receipt; do
  frame="$(manifest_value wire_schema) $(manifest_value positive_word0) $(manifest_value "${control}_word1")"
  set +e
  output="$(printf '%s\n' "$frame" | "$work/runtime-one")"
  code=$?
  set -e
  [[ $code -eq 42 && "$output" == "$(manifest_value "${control}_decision")" ]] ||
    fail "$control control drifted"
done

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_exec_result_record_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] ||
  fail 'Sounio result-record result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] ||
  fail 'Sounio result-record result hash drifted'
printf '%s\n' "$result" | cmp - "$ROOT_DIR/$(manifest_value evidence_path)" ||
  fail 'evidence is not the exact executable result'

printf 'sounio-loom-exec-result-record-freeze-selftest: PASS semantic_authority=Sounio action=9036 stage=SEMANTICS_FROZEN executable_sha256=%s artifact_binding=DENY577 causal_sabotage=PASS handle_is_bearer=false handle_is_execution_authority=false artifact_executed=false expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false ocaml_record_projection_attached=false dynamic_user_host_attached=false provider_result_returned=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(manifest_value executable_sha256)"
