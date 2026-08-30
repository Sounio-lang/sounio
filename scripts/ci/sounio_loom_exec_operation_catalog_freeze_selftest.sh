#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/exec_operation_catalog.freeze.v1"

fail() {
  printf 'sounio-loom-exec-operation-catalog-freeze-selftest: FAIL: %s\n' "$*" >&2
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
expect_value schema loom-exec-operation-catalog-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value producing_language Sounio
expect_value language_role SEMANTIC_AUTHORITY
expect_value action 9035
expect_value concept_id SOUNIO-LOOM-EXEC-OPERATION-CATALOG
expect_value case_count 11
expect_value catalog_entries calibration,sounio-check
expect_value unknown_operation DENY562
expect_value invalid_argument DENY563
expect_value write_effect DENY564
expect_value template_mismatch DENY567
expect_value causal_sabotage PASS
expect_value load_bearing_rule operation_specific_command_template_sha256_equal
expect_value arbitrary_shell false
expect_value expected_results_encoded_in_material_layer false
expect_value ocaml_catalog_projection_attached false
expect_value host_payload_selection_attached false
expect_value provider_lifecycle_attached false
expect_value general_exec_attached false
expect_value production_activation false
expect_value parity_open false
expect_value claim_ready false

for key in garden contract source entrypoint build_script selftest evidence \
  parent_9030_manifest parent_9031_manifest parent_9033_manifest \
  parent_9034_manifest toolchain_wrapper toolchain_compiler; do
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

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-exec-operation-catalog-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_EXEC_OPERATION_CATALOG_OUTPUT="$work/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_exec_operation_catalog_fixture.sh" >/dev/null
done
cmp "$work/runtime-one" "$work/runtime-two" ||
  fail 'Sounio rebuild is nondeterministic'
[[ "$(sha256sum "$work/runtime-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] ||
  fail 'Sounio executable hash drifted'

catalog_fields() {
  local prefix="$1"
  printf '%s\n' \
    "$(manifest_value fields_schema)" \
    "$(manifest_value catalog_schema)" \
    "$(manifest_value catalog_sha256)" \
    "$(manifest_value "${prefix}_name")" \
    "$(manifest_value "${prefix}_semantic_event_sha256")" \
    "$(manifest_value "${prefix}_command_template_sha256")" \
    "$(manifest_value "${prefix}_argument_schema_sha256")" \
    "$(manifest_value "${prefix}_result_schema_sha256")" \
    "$(manifest_value "${prefix}_sandbox_profile_sha256")"
}

calibration_frame="$(manifest_value wire_schema) $(manifest_value calibration_word0) $(manifest_value common_word1)"
sounio_check_frame="$(manifest_value wire_schema) $(manifest_value sounio_check_word0) $(manifest_value common_word1)"
mismatch_frame="$(manifest_value wire_schema) $(manifest_value template_mismatch_word0) $(manifest_value common_word1)"
[[ "$(printf '%s\n' "$calibration_frame" | "$work/runtime-one")" == \
   "$(manifest_value project_decision)
$(catalog_fields calibration)" ]] || fail 'calibration entry drifted'
[[ "$(printf '%s\n' "$sounio_check_frame" | "$work/runtime-one")" == \
   "$(manifest_value project_decision)
$(catalog_fields sounio_check)" ]] || fail 'sounio-check entry drifted'
set +e
mismatch_result="$(printf '%s\n' "$mismatch_frame" | "$work/runtime-one")"
mismatch_code=$?
set -e
[[ $mismatch_code -eq 42 && "$mismatch_result" == \
   "$(manifest_value template_mismatch_decision)" ]] ||
  fail 'template-binding sabotage drifted'

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_exec_operation_catalog_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] ||
  fail 'Sounio catalog result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] ||
  fail 'Sounio catalog result hash drifted'
printf '%s\n' "$result" |
  cmp - "$ROOT_DIR/$(manifest_value evidence_path)" ||
  fail 'evidence is not the exact executable result'

printf 'sounio-loom-exec-operation-catalog-freeze-selftest: PASS semantic_authority=Sounio action=9035 stage=SEMANTICS_FROZEN executable_sha256=%s entries=calibration+sounio-check unknown_operation=DENY562 invalid_argument=DENY563 write_effect=DENY564 template_mismatch=DENY567 causal_sabotage=PASS arbitrary_shell=false python_executed=false rust_executed=false ocaml_catalog_projection_attached=false host_payload_selection_attached=false provider_lifecycle_attached=false general_exec_attached=false production_activation=false parity_open=false claim_ready=false\n' \
  "$(manifest_value executable_sha256)"
