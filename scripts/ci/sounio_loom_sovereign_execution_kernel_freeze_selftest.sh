#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/sovereign_execution_kernel.freeze.v1"

fail() {
  printf 'sounio-loom-sovereign-execution-kernel-freeze-selftest: FAIL: %s\n' "$*" >&2
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

expect_commit_hash() {
  local commit="$1" path="$2" expected="$3" actual
  git -C "$ROOT_DIR" cat-file -e "${commit}^{commit}" ||
    fail "commit $commit is absent"
  actual="$(git -C "$ROOT_DIR" show "${commit}:$path" | sha256sum | cut -d ' ' -f 1)"
  [[ "$actual" == "$expected" ]] ||
    fail "$path is not bound to commit $commit"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'freeze manifest is absent or linked'
expect_value schema loom-sovereign-execution-kernel-freeze-v1
expect_value stage SEMANTICS_FROZEN
expect_value semantic_authority Sounio
expect_value producing_language Sounio
expect_value language_role SEMANTIC_AUTHORITY
expect_value action 9042
expect_value concept_id SOUNIO-LOOM-SOVEREIGN-EXECUTION-KERNEL
expect_value parent_actions 9025+9030+9031
expect_value refusal_range 602-612
expect_value causal_sabotage PASS
expect_value load_bearing_rule production_requires_same_uid_peer_isolation
expect_value grant_is_bearer false
expect_value exported_token false
expect_value exported_handle false
expect_value descriptor_is_execution_authority false
expect_value release_authority HostGuardian-only
expect_value interface_release_authority zero
expect_value guardian_loss fail-closed
expect_value expected_results_encoded_in_material_layer false
expect_value python_executed false
expect_value rust_executed false
expect_value material_execution false
expect_value same_uid_peer_isolation false
expect_value production_activation false
expect_value exec_attached false
expect_value parity_open false
expect_value claim_ready false

for key in garden contract concept_registry source entrypoint build_script selftest \
  freeze_selftest first_manifest first_evidence parent_9025 parent_9030 \
  parent_9031 toolchain_wrapper toolchain_compiler; do
  expect_hash "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

garden_commit="$(manifest_value garden_commit)"
expect_commit_hash "$garden_commit" "$(manifest_value garden_path)" \
  "$(manifest_value garden_sha256)"
source_commit="$(manifest_value sounio_executable_commit)"
for key in garden contract source entrypoint build_script selftest; do
  expect_commit_hash "$source_commit" "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done
first_commit="$(manifest_value first_receipt_commit)"
for key in first_manifest first_evidence concept_registry; do
  expect_commit_hash "$first_commit" "$(manifest_value "${key}_path")" \
    "$(manifest_value "${key}_sha256")"
done

work="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-sovereign-freeze.XXXXXX")"
trap 'rm -rf "$work"' EXIT
for ordinal in one two; do
  SOUNIO_LOOM_SOVEREIGN_OUTPUT="$work/runtime-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_execution_kernel.sh" >/dev/null
done
cmp "$work/runtime-one" "$work/runtime-two" || fail 'Sounio rebuild is nondeterministic'
[[ "$(sha256sum "$work/runtime-one" | cut -d ' ' -f 1)" == \
   "$(manifest_value executable_sha256)" ]] || fail 'Sounio executable hash drifted'

frame_output() {
  printf '%s %s %s %s %s %s\n' \
    "$(manifest_value wire_schema)" \
    "$(manifest_value "$1_mode")" \
    "$(manifest_value "$1_stage")" \
    "$(manifest_value "$1_word")" \
    "$(manifest_value sabotage_count)" \
    "$(manifest_value sabotage_required)" | "$work/runtime-one"
}

treatment="$(frame_output treatment)"
guardian_death="$(frame_output guardian_death)"
production="$(frame_output production)"
[[ "${treatment%%$'\n'*}" == "$(manifest_value treatment_decision)" ]] ||
  fail 'treatment decision drifted'
[[ "${guardian_death%%$'\n'*}" == "$(manifest_value guardian_death_decision)" ]] ||
  fail 'guardian-death decision drifted'
[[ "${production%%$'\n'*}" == "$(manifest_value production_decision)" ]] ||
  fail 'production decision drifted'
[[ "$(printf '%s' "$treatment" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value treatment_output_sha256)" ]] || fail 'treatment output drifted'
[[ "$(printf '%s' "$guardian_death" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value guardian_death_output_sha256)" ]] || fail 'guardian output drifted'
[[ "$(printf '%s' "$production" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value production_output_sha256)" ]] || fail 'production output drifted'

result="$(bash "$ROOT_DIR/scripts/ci/sounio_loom_sovereign_execution_kernel_selftest.sh")"
[[ "$result" == "$(manifest_value result)" ]] || fail 'first result drifted'
[[ "$(printf '%s' "$result" | sha256sum | cut -d ' ' -f 1)" == \
   "$(manifest_value result_sha256)" ]] || fail 'first result hash drifted'
printf '%s\n' "$result" | cmp - "$ROOT_DIR/$(manifest_value first_evidence_path)" ||
  fail 'first evidence is not the exact executable result'

freeze_result="sounio-loom-sovereign-execution-kernel-freeze-selftest: PASS semantic_authority=Sounio action=9042 stage=SEMANTICS_FROZEN semantics_sha256=$(manifest_value semantics_sha256) executable_sha256=$(manifest_value executable_sha256) treatment=EXEC_ADMIT guardian_death=GUARDIAN_REVOKE production=PRODUCTION_GATE_READY parents=9025+9030+9031 grant=resident-memory+non-bearer+single-use+atomic peer=SO_PEERCRED+pidfd+start-tick+harness-ancestry+executable+operation release_authority=HostGuardian-only interface_release_authority=zero same_uid_spoof=DENY609-before-exec guardian_loss=DENY610-fail-closed production_without_same_uid=DENY612 causal_sabotage=PASS expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false material_execution=false same_uid_peer_isolation=false production_activation=false exec_attached=false parity_open=false claim_ready=false"
printf '%s\n' "$freeze_result" | cmp - "$ROOT_DIR/$(manifest_value freeze_evidence_path)" ||
  fail 'freeze evidence is not the exact frozen result'
printf '%s\n' "$freeze_result"
